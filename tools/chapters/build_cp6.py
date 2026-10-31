#!/usr/bin/env python
"""Build the two notebooks of the mock exam of part VI from a single source (used by Claude).

    python tools/chapters/figures_cp6.py
    python tools/chapters/build_cp6.py
    python tools/run_all_notebooks.py checkpoints/partie_6/03_examen_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py checkpoints/partie_6/02_examen_notebook.ipynb

The exam notebook has two parts. Part A is done DURING the exam: the two code questions, CP6.8 (🔨 with a written
contract, PyTorch: the loss of a VAE, tested on a given batch) and CP6.9 (🐛, PyTorch: one step of a GAN with three
faulty lines), without mylearn and without any check. Part B is done AFTER the exam: it checks the code of part A
against oracles imported from ``wb.exams`` (the formulas of chapter 25; the GAN step of chapter 27, compared weight
by weight), then the answers written on paper (sub-IDs CP6.1a, CP6.2a, ...). Part B holds no solution in clear: the
oracles, and the hints that give the KL formula or name a faulty line of CP6.9 with its fix, come from ``wb.exams``
(audit P5 R3). Its cells only run once ``EXAM_OVER = True``; its tools come from ``wb.nbtools``. The solutions
notebook records every checked answer with ``wb.record``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import Paper, badge, bug_letters_mistakes, check_unique_keys, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "checkpoints/partie_6"
EXAM = f"{FOLDER}/02_examen_notebook.ipynb"
SOLUTIONS = f"{FOLDER}/03_examen_solutions.ipynb"

# ---------------------------------------------------------------------------
# Data and tools of the code questions
# ---------------------------------------------------------------------------
GIVEN = r'''# Data and tools of the code questions CP6.8 and CP6.9 (run this cell first)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CP6.8: a batch of 16 MNIST images in [0, 1] (N, 1, 28, 28), random logits (an untrained decoder), and the two heads
# of an encoder with d = 4 latents (mu, log_var); everything in float64
_X8, _ = wb.datasets.load_mnist("train", n=16, normalize=True, seed=68)
X_V = torch.from_numpy(_X8.astype(np.float64)).unsqueeze(1)                        # (16, 1, 28, 28)
_g8 = torch.Generator().manual_seed(68)
LOGITS_V = 2.0 * torch.randn(16, 1, 28, 28, generator=_g8, dtype=torch.float64)  # (16, 1, 28, 28)
MU_V = torch.randn(16, 4, generator=_g8, dtype=torch.float64)                     # (16, 4)
LOGVAR_V = 0.5 * torch.randn(16, 4, generator=_g8, dtype=torch.float64)           # (16, 4)
_ORIGINALS = {"X_V": X_V.clone(), "LOGITS_V": LOGITS_V.clone(), "MU_V": MU_V.clone(), "LOGVAR_V": LOGVAR_V.clone()}

# CP6.9: a 1D GAN; the data follow N(4, 1.25^2)
torch.set_num_threads(1)                                                          # tiny networks: one thread is fastest
bce = nn.BCEWithLogitsLoss()                                                      # the mean over the mini-batch


def make_gan(seed=69):
    """(G, D, opt_d, opt_g): a new generator 1-16-1 and discriminator 1-32-32-1 (it gives a logit), Adam(2e-3)."""
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
    D = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.5, 0.999))
    return G, D, opt_d, opt_g


def real_batch(n=64):
    """n samples of the data, N(4, 1.25^2), shape (n, 1)."""
    return 4.0 + 1.25 * torch.randn(n, 1)


print("X_V:", tuple(X_V.shape), "· LOGITS_V:", tuple(LOGITS_V.shape), "· MU_V and LOGVAR_V:", tuple(MU_V.shape),
      "· G parameters:", sum(p.numel() for p in make_gan()[0].parameters()))'''

CP68_MD = r'''### CP6.8 — Écrire la loss d'un VAE en PyTorch et la tester sur un batch fourni 🔨 ★★ ⏱️ 15 min · 2 points
**Objectif :** écrire de mémoire la loss d'un VAE avec ses deux termes à la bonne échelle.

Avec PyTorch. La cellule des données fournit un batch de 16 images MNIST `X_V` (valeurs dans $[0, 1]$, forme `(16, 1, 28, 28)`), des **logits** tirés au hasard, comme ceux d'un décodeur pas encore entraîné (`LOGITS_V`, même forme), et les deux têtes d'un encodeur à $d = 4$ latents (`MU_V`, `LOGVAR_V`, forme `(16, 4)`) ; tout est en `float64`. Avec ces logits, la reconstruction sera bien plus grande que celle d'un VAE entraîné.

a) `vae_loss(logits, x, mu, log_var, beta=1.0)` rend le triplet `(loss, rec, kl)` de tenseurs à 0 dimension, **par image** (moyennés sur le batch) : `rec`, la cross-entropy binaire entre `x` et `sigmoid(logits)`, **sommée** sur les pixels d'une image ; `kl`, la divergence KL entre $\mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(e^{\log\boldsymbol{\sigma}^2}))$ et $\mathcal{N}(\mathbf{0}, \mathbf{I})$, sommée sur les latents ; `loss = rec + beta * kl`. Elle marche pour toute forme d'images (`(N, 784)` comme `(N, 1, 28, 28)`) et garde le graphe (le gradient doit pouvoir remonter jusqu'à `logits`, `mu` et `log_var`). **[1,4]**
b) Avec le batch fourni : reporte sur ta feuille $[\text{rec}, \text{kl}]$ (2 décimales). **[0,3]**
c) La loss pour `beta=3` (2 décimales). **[0,3]**'''

CP68_TODO = r'''def vae_loss(logits, x, mu, log_var, beta=1.0):
    """(loss, rec, kl), 0-d tensors per example: rec = BCE summed over the pixels of an image, kl = Gaussian KL summed
    over the latents, both averaged over the batch; loss = rec + beta * kl; CP6.8 a)."""
    raise NotImplementedError("vae_loss")  # TODO CP6.8 a)'''

TRY_TOOL = r'''# The tries of part A show what your code does (no check during the exam)
import sys
import traceback


def try_code(question, run):
    """Run one try; a bug shows its traceback without stopping « Run all » (part B must still run)."""
    with wb.attempt(question):
        try:
            run()
        except NotImplementedError:
            raise
        except Exception as exc:
            if isinstance(exc, TypeError) and "NoneType" in str(exc):
                raise                                       # wb.attempt: « as-tu oublié le return ? »
            traceback.print_exc(file=sys.stdout)
            print(f"❌ {question} : ton code lève l'erreur ci-dessus (la suite du notebook s'exécute quand même).")


def try_8():
    loss, rec, kl = vae_loss(LOGITS_V, X_V, MU_V, LOGVAR_V)
    print("rec =", round(float(rec), 2), "· kl =", round(float(kl), 2), "· loss =", round(float(loss), 2))
    print("loss with beta = 3:", round(float(vae_loss(LOGITS_V, X_V, MU_V, LOGVAR_V, beta=3.0)[0]), 2))


try_code("CP6.8", try_8)'''

CP68_SOLUTION = r'''def vae_loss(logits, x, mu, log_var, beta=1.0):
    """(loss, rec, kl), 0-d tensors per example: rec = BCE summed over the pixels of an image, kl = Gaussian KL summed
    over the latents, both averaged over the batch; loss = rec + beta * kl; CP6.8 a)."""
    n = x.shape[0]
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / n        # sum over pixels, mean over images
    kl = 0.5 * (mu ** 2 + log_var.exp() - 1 - log_var).sum() / n                    # sum over latents, mean over images
    return rec + beta * kl, rec, kl


loss_8, rec_8, kl_8 = vae_loss(LOGITS_V, X_V, MU_V, LOGVAR_V)
loss3_8 = vae_loss(LOGITS_V, X_V, MU_V, LOGVAR_V, beta=3.0)[0]
print(f"rec = {rec_8.item():.4f} · kl = {kl_8.item():.4f} · loss = {loss_8.item():.4f} · loss (beta = 3) = {loss3_8.item():.4f}")'''

CP68_RECORD = r'''def variant_8(*, rec_mean=False, no_half=False, sigma=False, kl_mean=False, sigmoid_twice=False, no_batch_mean=False,
              beta=1.0):
    """[rec, kl] (or the loss for beta != 1) of CP6.8 written another (wrong) way."""
    n = X_V.shape[0]
    probs = torch.sigmoid(torch.sigmoid(LOGITS_V)) if sigmoid_twice else torch.sigmoid(LOGITS_V)
    bce_all = F.binary_cross_entropy(probs, X_V, reduction="none")
    rec = bce_all.mean() if rec_mean else bce_all.sum() / (1 if no_batch_mean else n)
    var = LOGVAR_V.exp() ** 2 if sigma else LOGVAR_V.exp()                          # log_var read as log(sigma)
    log_v = 2 * LOGVAR_V if sigma else LOGVAR_V
    terms = (MU_V ** 2 + var - 1 - log_v) * (1.0 if no_half else 0.5)
    kl = terms.mean() if kl_mean else terms.sum() / (1 if no_batch_mean else n)
    return [float(rec), float(kl)] if beta == 1.0 else float(rec + beta * kl)


wb.record("CP6.8b", [rec_8.item(), kl_8.item()], decimals=2, mistakes={
    "la reconstruction est SOMMÉE sur les 784 pixels de chaque image, puis moyennée sur les images": variant_8(rec_mean=True),
    "la KL d'une gaussienne contre N(0, 1) a un facteur 1/2 : ½(μ² + σ² − 1 − ln σ²)": variant_8(no_half=True),
    "log_var est le logarithme de la VARIANCE σ² : σ² = exp(log_var)": variant_8(sigma=True),
    "la KL est SOMMÉE sur les 4 latents, puis moyennée sur les images": variant_8(kl_mean=True),
    "les deux termes sont PAR IMAGE : divise les sommes par le nombre d'images": variant_8(no_batch_mean=True),
    "ce sont des logits : binary_cross_entropy_with_logits applique la sigmoïde une seule fois": variant_8(sigmoid_twice=True)})
wb.record("CP6.8c", loss3_8.item(), decimals=2, mistakes={
    "beta multiplie la KL, pas la reconstruction": float(3 * rec_8 + kl_8),
    "la loss vaut rec + beta * kl : avec beta = 3, la KL compte trois fois": loss_8.item(),
    "la KL est SOMMÉE sur les 4 latents, puis moyennée sur les images": variant_8(kl_mean=True, beta=3.0)})'''

CP69_MD = r'''### CP6.9 — Réparer un pas d'entraînement de GAN 🐛 ★★ ⏱️ 10 min · 1,5 point
**Objectif :** reconnaître les trois erreurs classiques d'un pas de GAN, et les corriger.

Avec PyTorch. `gan_step_buggy` (cellule suivante) doit faire **un** pas d'entraînement d'un GAN : d'abord le discriminateur (vrais exemples contre faux), puis le générateur (loss non saturante, comme au ch. 27). `make_gan()` rend un générateur, un discriminateur (qui rend un **logit**) et leurs deux optimiseurs ; `real_batch()` tire 64 vrais exemples. Elle contient **trois** lignes fautives ; une erreur peut se déclarer plus loin que sa cause.

a) Écris dans `faulty_9` les lettres des trois lignes **à corriger** (dans n'importe quel ordre). **[0,6]**
b) Écris `gan_step_fixed(G, D, real, opt_d, opt_g)`, la version corrigée : même signature, mêmes sorties, et le même tirage (un seul appel à `torch.randn`, ligne B, avant tout le reste). La cellule d'essai l'enchaîne 1 500 fois (quelques secondes) : la moyenne des exemples générés doit s'approcher de 4. **[0,9]**'''

CP69_GIVEN = r'''# CP6.9: the step to repair (do not change this cell: copy the function into the next one to fix it)
def gan_step_buggy(G, D, real, opt_d, opt_g):
    """One step of a GAN, the discriminator then the generator; returns (d_loss, g_loss) as floats."""
    ones, zeros = torch.ones(len(real), 1), torch.zeros(len(real), 1)          # A
    fake = G(torch.randn(len(real), 1))                                         # B
    d_loss = bce(D(real), ones) + bce(D(fake), zeros)                           # C
    opt_d.zero_grad()                                                           # D
    d_loss.backward()                                                           # E
    opt_d.step()                                                                # F
    g_loss = bce(D(fake), zeros)                                                # G
    opt_g.zero_grad()                                                           # H
    g_loss.backward()                                                           # I
    opt_d.step()                                                                # J
    return d_loss.item(), g_loss.item()                                         # K'''

CP69_TODO = r'''faulty_9 = ...   # CP6.9 a): the letters of the three faulty lines, e.g. "ABC"


def gan_step_fixed(G, D, real, opt_d, opt_g):
    """gan_step_buggy, fixed (same signature, same returns, one call to torch.randn at line B); CP6.9 b)."""
    raise NotImplementedError("gan_step_fixed")  # TODO CP6.9 b)'''

CP69_TRY = r'''# Try gan_step_fixed: 1 500 steps (a few seconds; no check during the exam)
def train_gan(step, n_steps=1500, seed=69):
    """Train make_gan(seed) with `step` on real_batch(); returns (G, the losses of the last 100 steps)."""
    G, D, opt_d, opt_g = make_gan(seed)
    torch.manual_seed(seed + 1)
    losses = [step(G, D, real_batch(), opt_d, opt_g) for _ in range(n_steps)]
    return G, np.mean(losses[-100:], axis=0)


def try_9():
    G_try, (d_try, g_try) = train_gan(gan_step_fixed)
    with torch.no_grad():
        sample = G_try(torch.randn(2000, 1))
    print(f"generated: mean {sample.mean().item():.2f}, std {sample.std().item():.2f} (data: 4 and 1.25) · "
          f"last 100 steps: d_loss {d_try:.3f}, g_loss {g_try:.3f}")


try_code("CP6.9", try_9)'''

CP69_SOLUTION = r'''faulty_9 = "CGJ"


def gan_step_fixed(G, D, real, opt_d, opt_g):
    """gan_step_buggy, fixed (same signature, same returns, one call to torch.randn at line B); CP6.9 b)."""
    ones, zeros = torch.ones(len(real), 1), torch.zeros(len(real), 1)          # A
    fake = G(torch.randn(len(real), 1))                                         # B
    d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)                  # C: D's step must not reach G
    opt_d.zero_grad()                                                           # D
    d_loss.backward()                                                           # E
    opt_d.step()                                                                # F
    g_loss = bce(D(fake), ones)                                                 # G: G wants D to answer "real"
    opt_g.zero_grad()                                                           # H
    g_loss.backward()                                                           # I
    opt_g.step()                                                                # J: the generator's optimizer
    return d_loss.item(), g_loss.item()                                         # K


def train_gan(step, n_steps=1500, seed=69):
    """Train make_gan(seed) with `step` on real_batch(); returns (G, the losses of the last 100 steps)."""
    G, D, opt_d, opt_g = make_gan(seed)
    torch.manual_seed(seed + 1)
    losses = [step(G, D, real_batch(), opt_d, opt_g) for _ in range(n_steps)]
    return G, np.mean(losses[-100:], axis=0)


G_9, (d_9, g_9) = train_gan(gan_step_fixed)
with torch.no_grad():
    sample_9 = G_9(torch.randn(2000, 1, generator=torch.Generator().manual_seed(9)))
print(f"generated: mean {sample_9.mean().item():.2f}, std {sample_9.std().item():.2f} (data: 4 and 1.25) · "
      f"last 100 steps: d_loss {d_9:.3f} (2 ln 2 = 1.386), g_loss {g_9:.3f} (ln 2 = 0.693)")'''

CP69_RECORD = r'''wb.record("CP6.9a", "CGJ", mistakes=''' + bug_letters_mistakes("CGJ", "ABDEFHIK", {
    "C": "le pas du discriminateur ne doit pas remonter jusqu'au générateur (pense à detach)",
    "G": "quel label le générateur veut-il que D donne à ses faux ?",
    "J": "quel optimiseur met à jour le générateur ?"},
    special={("C", "I"): "l'erreur se déclare en (I), mais la ligne à corriger est celle qui laisse le pas de D "
                         "libérer le graphe du générateur"}) + ")"

# ---------------------------------------------------------------------------
# Part B: checks after the exam
# ---------------------------------------------------------------------------
PART_B_INTRO = r'''## Partie B · Après l'examen : vérification automatique

**Ne commence cette partie qu'une fois ta copie terminée**, examen rendu. Elle vérifie le code de la partie A, puis les réponses que tu as écrites sur ta feuille, question par question : ✅ juste, ❌ faux (avec une piste), ⏳ pas rempli. Elle ne note pas : les démarches, les justifications et les questions rédigées se corrigent avec `03_examen_corrige.md`, qui donne aussi le barème.

1. Passe `EXAM_OVER` à `True` dans la cellule ci-dessous, puis exécute-la.
2. Exécute les vérifications du code (CP6.8 et CP6.9 ; la seconde entraîne le GAN deux fois, une dizaine de secondes).
3. Reporte ensuite chaque réponse de ta feuille : **la valeur** que tu as écrite, pas un nouveau calcul (sinon tu ne vérifies plus ta copie). En Python, le séparateur décimal est un **point** (`0.125`), une matrice s'écrit comme une liste de lignes (`[[1, 2], [3, 4]]`), une forme comme une chaîne (`"(2, 5, 7, 7)"`) et les lettres entre guillemets (`"XYZ"`).

Si tu as fermé le notebook depuis l'examen, exécute d'abord la cellule de configuration (tout en haut) et les cellules de la partie A, sauf les essais.'''

PART_B_FLAG = r'''EXAM_OVER = False   # set it to True once your exam is finished, then run the cells of part B

import re

from wb import exams
from wb.nbtools import StopCheck, as_numpy, fr, letters_text, shape_text, verdict

RESULTS = {}        # sub-ID -> result of wb.check
PROPERTIES = {}     # question -> [(property, passed)], reset each time its check cell runs
PENDING = set()     # the code questions not written yet (⏳)
_REMINDED = []      # the full reminder is printed once, then a short one


def exam_over():
    """True once the exam is finished; otherwise print a reminder (and the cell checks nothing)."""
    if not EXAM_OVER:
        print("⏸️ Rien n'est vérifié (EXAM_OVER vaut False)." if _REMINDED else
              "⏸️ Vérification après l'examen : passe EXAM_OVER à True dans la première cellule de la partie B.")
        _REMINDED.append(True)
        return False
    if "wb" not in globals():
        print("⚠️ Exécute d'abord la cellule de configuration, tout en haut du notebook.")
        return False
    return True


def ready(question, *names):
    """True when the cells of part A that define `names` have run (after a kernel restart, run them again)."""
    missing = [name for name in names if name not in globals()]
    if missing:
        verb = "n'existe" if len(missing) == 1 else "n'existent"
        print(f"⚠️ {question} : exécute d'abord les cellules de la partie A ({', '.join(missing)} {verb} pas encore).")
    return not missing


def prop(question, ok, success, failure):
    """One property of the code of part A: ✅ or ❌ (wb.nbtools.verdict), remembered for the summary."""
    PROPERTIES.setdefault(question, []).append((success or failure, bool(ok)))
    verdict(question, ok, success, failure)
    return bool(ok)


def safe(function, *args, **kwargs):
    """(True, result) or (False, the text of the error), or (None, None) while the code is not written:
    a bug of the exam code must not stop part B."""
    try:
        return True, function(*args, **kwargs)
    except NotImplementedError:
        return None, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"


_LIBRARY_ADVICE = " ; si elle fait partie de `mylearn`, ses tests (`pytest`) disent quel cas pose problème."


def code_check(ex_id, value, reminder=""):
    """wb.check of a value computed by your exam function (no advice about mylearn: it is not used here)."""
    result = wb.check(ex_id, value, computed=True, quiet=True)
    message = result.message.replace(_LIBRARY_ADVICE, ".")
    if not result and reminder and not message.startswith("Erreur classique"):
        message += f" Rappel : {reminder}"
    print(f"{'✅' if result else '❌'} Ex {ex_id} : {message}")
    RESULTS[ex_id] = result
    return result


def sorted_letters(value):
    """A set of letters typed in any order ("K, A et E") -> its letters in alphabetical order ("AEK")."""
    return letters_text(value, sort=True)


def restore_data(question, names):
    """Put back the data of part A if a try of part A changed them in place (a function that writes into its input)."""
    changed = [name for name in names
               if name not in globals() or not torch.equal(globals()[name], _ORIGINALS[name])]
    for name in changed:
        globals()[name] = _ORIGINALS[name].clone()
    if changed:
        print(f"⚠️ {question} : {', '.join(changed)} avai{'en' if len(changed) > 1 else ''}t changé depuis la cellule des "
              f"données (une fonction de la partie A écrit-elle dans ce qu'elle reçoit ?) : la vérification repart "
              f"des données d'origine.")'''

CODE_CHECK_8 = r'''# Checks of vae_loss (CP6.8), against the formulas of chapter 25
if exam_over() and ready("CP6.8", "np", "torch", "F", "X_V", "LOGITS_V", "MU_V", "LOGVAR_V", "_ORIGINALS", "vae_loss"):
    PROPERTIES["CP6.8"] = []
    PENDING.discard("CP6.8")
    for key in ("CP6.8b", "CP6.8c"):
        RESULTS.pop(key, None)
    restore_data("CP6.8", ["X_V", "LOGITS_V", "MU_V", "LOGVAR_V"])
    oracle_8 = exams.cp6_vae_loss                       # (loss, rec, kl) as floats, the formulas of chapter 25
    hints_8 = exams.HINTS["CP6.8"]

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    inputs = [t.clone() for t in (LOGITS_V, X_V, MU_V, LOGVAR_V)]
    ok, result = safe(vae_loss, *inputs)
    if ok is None:
        print("⏳ Ex CP6.8 : vae_loss n'est pas encore écrite.")
        PENDING.add("CP6.8")
        for key in ("CP6.8b", "CP6.8c"):
            RESULTS[key] = wb.check(key, ..., quiet=True)
    elif not ok:
        prop("CP6.8", False, "", f"vae_loss lève une erreur : {result}")
    elif result is None:
        prop("CP6.8", False, "", "vae_loss renvoie None : as-tu oublié le return ?")
    elif not isinstance(result, (tuple, list)) or len(result) != 3:
        prop("CP6.8", False, "", f"vae_loss doit rendre le triplet (loss, rec, kl) ; reçu {type(result).__name__}"
             + (f" de longueur {len(result)}" if isinstance(result, (tuple, list)) else "") + ".")
    else:
        got = [as_float(v) for v in result]
        want = oracle_8(LOGITS_V, X_V, MU_V, LOGVAR_V)
        prop("CP6.8", all(isinstance(v, torch.Tensor) and v.dim() == 0 for v in result),
             "loss, rec et kl sont des tenseurs à 0 dimension.",
             "loss, rec et kl doivent être des tenseurs à 0 dimension (pas des float : le gradient doit pouvoir "
             "remonter ; pas un tenseur par image : la moyenne sur le batch).")
        if None not in got:
            prop("CP6.8", abs(got[1] - want[1]) < 1e-6 * abs(want[1]),
                 "rec : la cross-entropy binaire sur les logits, sommée sur les pixels, moyennée sur les images.",
                 f"rec vaut {fr(got[1], 4)}, attendu {fr(want[1], 4)} : la cross-entropy binaire sur les LOGITS (une seule "
                 "sigmoïde), une SOMME sur les pixels de chaque image, divisée par le nombre d'images.")
            prop("CP6.8", abs(got[2] - want[2]) < 1e-9 + 1e-6 * abs(want[2]), hints_8["kl_ok"],
                 hints_8["kl_wrong"].format(got=fr(got[2], 4), want=fr(want[2], 4)))
            prop("CP6.8", abs(got[0] - want[0]) < 1e-6 * abs(want[0]), "loss = rec + kl avec beta = 1.",
                 f"loss vaut {fr(got[0], 4)}, attendu {fr(want[0], 4)} = rec + beta * kl.")
        prop("CP6.8", all(torch.equal(a, b) for a, b in zip(inputs, (LOGITS_V, X_V, MU_V, LOGVAR_V))),
             "vae_loss ne modifie pas ses entrées.", "vae_loss modifie un tenseur qu'elle reçoit : elle doit le laisser intact.")
        restore_data("CP6.8", ["X_V", "LOGITS_V", "MU_V", "LOGVAR_V"])
        want4 = oracle_8(LOGITS_V, X_V, MU_V, LOGVAR_V, beta=3.0)[0]
        ok4, res4 = safe(vae_loss, *[t.clone() for t in (LOGITS_V, X_V, MU_V, LOGVAR_V)], beta=3.0)
        got4 = as_float(res4[0]) if ok4 and isinstance(res4, (tuple, list)) and len(res4) == 3 else None
        prop("CP6.8", got4 is not None and abs(got4 - want4) < 1e-6 * abs(want4),
             "beta multiplie la KL (beta = 3).", f"avec beta = 3, attendu loss = {fr(want4, 4)}, reçu {got4} : "
             "beta multiplie la KL seulement, et vient de l'argument.")
        # another shape (N, 784), other sizes, and the gradient
        g_o = torch.Generator().manual_seed(681)
        lo = torch.randn(3, 784, generator=g_o, dtype=torch.float64, requires_grad=True)
        xo = torch.rand(3, 784, generator=g_o, dtype=torch.float64)
        mo = torch.randn(3, 5, generator=g_o, dtype=torch.float64, requires_grad=True)
        vo = torch.randn(3, 5, generator=g_o, dtype=torch.float64, requires_grad=True)
        ok_o, res_o = safe(vae_loss, lo, xo, mo, vo, beta=0.5)
        want_o = oracle_8(lo.detach(), xo, mo.detach(), vo.detach(), beta=0.5)
        try:
            same_o = bool(ok_o and abs(float(torch.as_tensor(res_o[0]).detach()) - want_o[0]) < 1e-6 * abs(want_o[0]))
        except (TypeError, ValueError, IndexError, RuntimeError):
            same_o = False
        main_ok = None not in got and all(abs(g - w) < 1e-6 * abs(w) + 1e-9 for g, w in zip(got, want))
        if not ok_o:
            prop("CP6.8", False, "", f"avec des images aplaties (3, 784) et 5 latents, vae_loss lève une erreur : {res_o} "
                 "(utilise les tailles de ses arguments, et ne modifie pas un tenseur qui demande un gradient).")
        else:
            prop("CP6.8", same_o, "avec des images aplaties (3, 784), 5 latents et beta = 0,5, la loss du contrat.",
                 "avec des images aplaties (3, 784), 5 latents et beta = 0,5, la loss n'est pas celle du contrat : "
                 + ("utilise les tailles des arguments (pas 16 images, 784 pixels ou 4 latents écrits en dur)."
                    if main_ok else "c'est la conséquence des erreurs ci-dessus."))
            try:
                res_o[0].backward()
                flows = all(t.grad is not None and float(t.grad.abs().sum()) > 0 for t in (lo, mo, vo))
            except Exception:
                flows = False
            prop("CP6.8", flows, "le gradient remonte jusqu'aux logits, à mu et à log_var.",
                 "loss.backward() ne donne pas de gradient à logits, mu et log_var : pas de .item(), de .detach() ni de "
                 "torch.no_grad() dans vae_loss.")
        if None not in got:
            code_check("CP6.8b", [got[1], got[2]], "[rec, kl] avec le batch fourni et beta = 1.")
        if got4 is not None:
            code_check("CP6.8c", got4, "la loss avec beta = 3.")'''

CODE_CHECK_9 = r'''# Checks of gan_step_fixed (CP6.9), against the step of chapter 27, weight by weight (about ten seconds)
if exam_over() and ready("CP6.9", "torch", "nn", "bce", "make_gan", "real_batch", "gan_step_fixed", "faulty_9"):
    PROPERTIES["CP6.9"] = []
    PENDING.discard("CP6.9")
    RESULTS.pop("CP6.9a", None)
    RESULTS["CP6.9a"] = wb.check("CP6.9a", ... if faulty_9 is ... else sorted_letters(faulty_9))
    reference_step = exams.cp6_gan_step                 # the GAN step of chapter 27
    hints_9 = exams.HINTS["CP6.9"]

    def run_steps(step, n=3, seed=69):
        """(ok, losses or error, G weights, D weights) after n steps from make_gan(seed)."""
        G, D, opt_d, opt_g = make_gan(seed)
        torch.manual_seed(seed + 1)
        losses = []
        for _ in range(n):
            ok, out = safe(step, G, D, real_batch(), opt_d, opt_g)
            if not ok:
                return ok, out, None, None
            losses.append(out)
        return True, losses, [p.detach().clone() for p in G.parameters()], [p.detach().clone() for p in D.parameters()]

    ok, mine, mine_g, mine_d = run_steps(gan_step_fixed)
    if ok is None:
        print("⏳ Ex CP6.9 : gan_step_fixed n'est pas encore écrite.")
        PENDING.add("CP6.9")
    elif not ok:
        prop("CP6.9", False, "", f"gan_step_fixed lève une erreur : {mine}" + (
            hints_9["freed_graph"] if "backward through the graph" in str(mine) else ""))
    else:
        _, ref, ref_g, ref_d = run_steps(reference_step)
        try:
            pairs_ok = all(len(m) == 2 and isinstance(m[0], float) and isinstance(m[1], float) for m in mine)
        except TypeError:
            pairs_ok = False
        prop("CP6.9", pairs_ok, "gan_step_fixed rend le couple (d_loss, g_loss) de float.",
             "gan_step_fixed doit rendre le couple (d_loss, g_loss), deux float Python (.item()).")
        same_d = all(torch.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(mine_d, ref_d))
        same_g = all(torch.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(mine_g, ref_g))
        _, _, _, mine_d1 = run_steps(gan_step_fixed, n=1)
        _, _, _, ref_d1 = run_steps(reference_step, n=1)
        first_d = mine_d1 is not None and all(torch.allclose(a, b, rtol=1e-5, atol=1e-7) for a, b in zip(mine_d1, ref_d1))
        start_g = [p.detach().clone() for p in make_gan(69)[0].parameters()]
        still_g = mine_g is not None and all(torch.equal(a, b) for a, b in zip(mine_g, start_g))   # G never moved
        why_d = hints_9["d_after_first" if first_d else "d_still_g" if still_g else "d_other"]
        prop("CP6.9", same_d, "après trois pas, les poids du discriminateur sont ceux du pas de référence.",
             "après trois pas, les poids du discriminateur diffèrent de ceux du pas de référence : " + why_d)
        prop("CP6.9", same_g, "après trois pas, les poids du générateur sont ceux du pas de référence.",
             "après trois pas, les poids du générateur diffèrent : " + hints_9["g_still" if still_g else "g_other"])
        if pairs_ok and len(mine) == len(ref):
            first_losses = np.allclose(np.array(mine[0], dtype=float), np.array(ref[0], dtype=float), rtol=1e-5)
            prop("CP6.9", np.allclose(np.array(mine, dtype=float), np.array(ref, dtype=float), rtol=1e-5),
                 "les losses rendues sont celles du pas de référence.",
                 "les losses du premier pas sont justes, celles des pas suivants diffèrent : c'est la conséquence de "
                 "l'écart des poids ci-dessus." if first_losses else hints_9["losses_wrong"])
        if pairs_ok and same_d and same_g:
            G_c, (d_c, g_c) = train_gan(gan_step_fixed)
            with torch.no_grad():
                sample_c = G_c(torch.randn(2000, 1, generator=torch.Generator().manual_seed(9)))
            prop("CP6.9", abs(sample_c.mean().item() - 4.0) < 0.5,
                 f"après 1 500 pas, les exemples générés ont une moyenne de {sample_c.mean().item():.2f} (les données : 4)."
                 .replace(".", ",", 1),
                 f"après 1 500 pas, la moyenne générée vaut {sample_c.mean().item():.2f}, loin de 4.".replace(".", ",", 1))'''

PAPER_CONTEXT = r'''# The paper questions only use the numbers of their statements (01_examen_sujet.md)
import math

import numpy as np

# CP6.1
RET_1 = 0.9 ** 2 * 1.0

# CP6.2
MU_2, LOGVAR_2, EPS_2 = np.array([0.5, -1.0]), np.array([0.0, math.log(0.25)]), np.array([1.2, -0.4])
SIGMA_2 = np.exp(0.5 * LOGVAR_2)
Z_2 = MU_2 + SIGMA_2 * EPS_2
KL_2 = 0.5 * (MU_2 ** 2 + np.exp(LOGVAR_2) - 1 - LOGVAR_2)
DZ_2 = 0.5 * SIGMA_2[1] * EPS_2[1]                    # dz2 / d(log sigma2^2)
KL_SMALL_2 = 0.5 * (0.0 + math.exp(-10.0) - 1 + 10.0)

# CP6.3
QA_3, QB_3, QC_3 = [0.2, 0.4, 0.1], [0.5, 0.3, 0.8], [0.6, 0.0, 0.0]
ALPHA_3, GAMMA_3, EPS_3 = 0.5, 0.9, 0.2


def update_3(q, target, alpha=ALPHA_3):
    return q + alpha * (target - q)


QL_A_3 = update_3(QA_3[1], 0 + GAMMA_3 * max(QB_3))
QL_B_3 = update_3(QB_3[2], 1.0)                       # C is a winning board: the episode ends
SARSA_A_3 = update_3(QA_3[1], 0 + GAMMA_3 * QB_3[0])
P_GREEDY_3 = 1 - EPS_3 + EPS_3 / 3

# CP6.4
D_EQ_4, G_EQ_4 = 2 * math.log(2), math.log(2)
D_STAR_4 = 0.3 / (0.3 + 0.1)

# CP6.5
F_5 = np.array([[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1]], dtype=float)
RAW_5 = F_5 @ F_5.T
G_5 = RAW_5 / F_5.size                                # C * H * W = 3 * 2 * 2

# CP6.10
TABLE_10 = 3 ** 9 * 9 * 4
REACH_10 = 5478 * 9 * 4
BIG_10 = 3 ** 16 * 16 * 4
NET_10 = (48 * 128 + 128) + (128 * 128 + 128) + (128 * 16 + 16)

# CP6.13


def conv_out(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1


def adam_13(grads, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, w=1.0):
    m = v = 0.0
    for t, g in enumerate(grads, start=1):
        m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g * g
        w -= lr * (m / (1 - b1 ** t)) / (math.sqrt(v / (1 - b2 ** t)) + eps)
    return w'''


def _letters_mistakes(answer: str, messages: dict) -> str:
    """mistakes={message: letter} for every other choice of a lettered question."""
    pairs = ",\n                     ".join(f'"{message}": "{letter}"' for letter, message in messages.items()
                                            if letter != answer)
    return "mistakes={" + pairs + "}"


TF = "True or False"
PAPER = [
    Paper("CP6.1", "Questions flash : dix réponses courtes sur la partie VI", [
        ("a", "a number, 1 decimal", "784 / 32",
         r'''decimals=1, mistakes={"le facteur de compression divise la taille de l'entrée par celle du code": 32 / 784}'''),
        ("b", TF, "False",
         r'''mistakes={"un autoencodeur ordinaire n'oblige pas ses codes à remplir N(0, I) : un code tiré au hasard tombe souvent dans un trou de l'espace latent": True}'''),
        ("c", 'a letter, e.g. "B"', '"B"', _letters_mistakes("B", {
            "A": "N(μ, σ²) est le posterior approché, celui que rend l'encodeur pour une image ; le prior est fixé d'avance",
            "C": "le prior du VAE est une gaussienne, pas une loi uniforme"})),
        ("d", "a number, 2 decimals", "RET_1",
         r'''decimals=2, mistakes={"la récompense arrive au pas 2 : elle est multipliée par γ²": 0.9,
                     "sans γ, c'est le retour non actualisé": 1.0}'''),
        ("e", 'a shape like "(1, 2)"', '"(16, 4)"',
         r'''mistakes={"une ligne par état, une colonne par action": "(4, 16)",
                     "FrozenLake 4 × 4 a 16 états (les cases) et 4 actions (les directions)": "(4, 4)"}'''),
        ("f", "a number", "0.5",
         r'''decimals=2, mistakes={"à l'équilibre, D ne peut plus distinguer : il répond 1/2 partout, pas 1": 1.0,
                     "1,386 = 2 ln 2 est la loss de D à l'équilibre ; on demande sa SORTIE D(x)": 1.386}'''),
        ("g", TF, "True",
         r'''mistakes={"l'effondrement de mode, c'est un générateur qui ne couvre qu'une partie des modes (des pics) des données": False}'''),
        ("h", 'a shape like "(1, 2)"', '"(64, 64)"',
         r'''mistakes={"la matrice de Gram croise les cartes deux à deux : une ligne et une colonne par canal": "(196, 196)",
                     "la matrice de Gram ne garde pas les positions : elle est C × C": "(64, 196)"}'''),
        ("i", "a whole number", "70000",
         r'''mistakes={"ajoute les 10 000 images de test": 60000}'''),
        ("j", TF, "True",
         r'''mistakes={"CC BY n'interdit pas l'usage commercial : elle demande de citer la source (c'est NC qui l'interdit)": False}'''),
    ]),
    Paper("CP6.2", "VAE : un tirage reparamétrisé et sa divergence KL", [
        ("a", "[z1, z2], 4 decimals", "Z_2.tolist()",
         r'''decimals=4, mistakes={"σ = exp(½ log σ²) : log_var n'est pas l'écart-type": (MU_2 + LOGVAR_2 * EPS_2).tolist(),
                     "on multiplie ε par l'écart-type σ, pas par la variance σ²": (MU_2 + np.exp(LOGVAR_2) * EPS_2).tolist()}'''),
        ("b", "[KL1, KL2], 4 decimals", "KL_2.tolist()",
         r'''decimals=4, mistakes={"la KL a un facteur ½ : ½(μ² + σ² − 1 − ln σ²)": (2 * KL_2).tolist(),
                     "n'oublie pas le − 1 : la KL de N(0, 1) contre elle-même doit être nulle": (KL_2 + 0.5).tolist(),
                     "ln σ² est log_var lui-même, déjà un logarithme": (0.5 * (MU_2 ** 2 + np.exp(LOGVAR_2) - 1)).tolist()}'''),
        ("c", "a number, 4 decimals", "DZ_2",
         r'''decimals=4, mistakes={"z = μ + exp(½ log σ²) ε : la dérivée par rapport à log σ² garde le facteur ½": 2 * DZ_2,
                     "c'est la dérivée par rapport à μ qui vaut 1 ; on demande celle par rapport à log σ²": 1.0}'''),
        ("d", "a number, 4 decimals", "KL_SMALL_2",
         r'''decimals=4, mistakes={"le terme − ln σ² vaut + 10 ici : une variance minuscule coûte cher": 0.5 * (math.exp(-10) - 1)}'''),
    ]),
    Paper("CP6.3", "Deux transitions : mise à jour Q-learning puis SARSA", [
        ("a", "a number, 4 decimals", "QL_A_3",
         r'''decimals=4, mistakes={"le Q-learning prend le MAX de la ligne de B, pas la valeur de l'action jouée ensuite": SARSA_A_3,
                     "la cible actualise la valeur de B par γ : r + γ max Q(B, ·)": update_3(QA_3[1], max(QB_3)),
                     "on mélange la cible avec l'ancienne valeur dans la proportion α : Q + α (cible − Q)": 0 + GAMMA_3 * max(QB_3)}'''),
        ("b", "a number, 4 decimals", "QL_B_3",
         r'''decimals=4, mistakes={"C est un plateau gagnant : l'épisode s'arrête, la cible est r seul (pas d'amorçage sur sa ligne)": update_3(QB_3[2], 1 + GAMMA_3 * max(QC_3)),
                     "on mélange la cible avec l'ancienne valeur dans la proportion α : Q + α (cible − Q)": 1.0}'''),
        ("c", "a number, 4 decimals", "SARSA_A_3",
         r'''decimals=4, mistakes={"SARSA prend la valeur de l'action réellement choisie ensuite (a' = 0), pas le max": QL_A_3,
                     "la cible actualise la valeur de B par γ : r + γ Q(B, a')": update_3(QA_3[1], QB_3[0])}'''),
        ("d", "a probability, 4 decimals", "P_GREEDY_3",
         r'''decimals=4, mistakes={"l'exploration tire une action au hasard parmi les TROIS, la meilleure comprise : elle aussi peut tomber sur l'action 2": 1 - EPS_3,
                     "c'est la probabilité d'une action non gloutonne donnée : on demande celle de l'action gloutonne": EPS_3 / 3}'''),
    ]),
    Paper("CP6.4", "Discriminateur optimal et losses à l'équilibre", [
        ("b", "[loss of D, loss of G], 3 decimals", "[D_EQ_4, G_EQ_4]",
         r'''decimals=3, mistakes={"la loss de D additionne deux cross-entropies, chacune vaut ln 2 à l'équilibre": [G_EQ_4, G_EQ_4],
                     "la loss non saturante de G est − ln D(G(z)), un seul terme : ln 2": [D_EQ_4, D_EQ_4],
                     "c'est la sortie de D à l'équilibre ; on demande les losses": [0.5, 0.5]}'''),
        ("c", "a number, 4 decimals", "D_STAR_4",
         r'''decimals=4, mistakes={"D* = p_data / (p_data + p_g), pas p_g / (p_data + p_g)": 0.1 / 0.4,
                     "D* divise par la SOMME des deux densités": 0.3 / 0.1}'''),
    ]),
    Paper("CP6.5", "Matrice de Gram de trois cartes d'activation", [
        ("a", "F F^T as a list of rows", "RAW_5.tolist()",
         r'''decimals=1, mistakes={"F F^T : produit des cartes mises à plat ligne par ligne, une ligne et une colonne par canal (C × C)": (F_5.T @ F_5).tolist()}'''),
        ("b", "[G00, G02], 4 decimals", "[G_5[0, 0], G_5[0, 2]]",
         r'''decimals=4, mistakes={"la fiche divise par C·H·W = 12, pas par H·W seulement": [RAW_5[0, 0] / 4, RAW_5[0, 2] / 4],
                     "la fiche divise par C·H·W = 12, pas par C seulement": [RAW_5[0, 0] / 3, RAW_5[0, 2] / 3],
                     "la matrice de Gram normalisée divise F F^T par C·H·W": [RAW_5[0, 0], RAW_5[0, 2]]}'''),
    ]),
    Paper("CP6.6", "Prédire : γ → 0 puis ε = 0 pour un agent Q-learning sur Flippers", [
        ("a", 'a letter, e.g. "B"', '"C"', _letters_mistakes("C", {
            "A": "avec γ = 0, une case Q ne voit que la récompense immédiate : loin de la victoire, toutes les valeurs restent nulles",
            "B": "avec γ = 0, seuls les plateaux à UN coup de la victoire apprennent quelque chose : regarde quelle part ils représentent"})),
        ("b", 'a letter, e.g. "B"', '"A"', _letters_mistakes("A", {
            "B": "tant que la table est nulle, les égalités se départagent AU HASARD pendant l'entraînement : l'agent explore quand même",
            "C": "tant que la table est nulle, les égalités se départagent AU HASARD pendant l'entraînement : l'agent explore quand même ; γ = 0,9 propage ensuite les victoires"})),
    ]),
    Paper("CP6.7", "Trois courbes, trois diagnostics : VAE, GAN, apprentissage par renforcement", [
        ("a", "a letter A to F", '"A"', _letters_mistakes("A", {
            "B": "regarde la KL (axe de droite) : elle reste collée à 0 et la reconstruction plafonne vers 208",
            "C": "les courbes ① sont celles d'un VAE (une reconstruction et une KL), pas d'un GAN",
            "D": "un VAE n'a ni discriminateur ni générateur à mettre en équilibre : lis la KL",
            "E": "les courbes ① sont celles d'un VAE, pas d'un agent",
            "F": "un VAE n'a pas de récompense : lis la KL, sur l'axe de droite"})),
        ("b", "a letter A to F", '"C"', _letters_mistakes("C", {
            "A": "les courbes ② sont celles d'un GAN (une loss de D, une loss de G), pas d'un VAE",
            "B": "un GAN n'a ni reconstruction ni KL : compare les deux losses",
            "D": "à l'équilibre, la loss de D vaudrait 2 ln 2 ≈ 1,39 ; ici elle est presque nulle et celle de G dépasse 6",
            "E": "les courbes ② sont celles d'un GAN, pas d'un agent",
            "F": "un GAN n'a pas de récompense : compare les deux losses"})),
        ("c", "a letter A to F", '"F"', _letters_mistakes("F", {
            "A": "les courbes ③ sont celles d'un agent (un retour par épisode), pas d'un VAE",
            "B": "un agent n'a ni reconstruction ni KL : lis le niveau du retour",
            "C": "les courbes ③ sont celles d'un agent, pas d'un GAN",
            "D": "un agent n'a pas de discriminateur : lis le niveau du retour",
            "E": "le retour d'un épisode vaut 1 pour une victoire : après 3 000 épisodes, il reste sous 0,1, loin d'une bonne politique"})),
    ]),
    Paper("CP6.10", "Table Q ou réseau : estimer la mémoire nécessaire", [
        ("a", "megabytes, 2 decimals", "TABLE_10 / 1e6",
         r'''decimals=2, mistakes={"un float32 occupe 4 octets": 3 ** 9 * 9 / 1e6,
                     "une ligne par codage de plateau (3⁹), une colonne par case (9)": 3 ** 9 * 4 / 1e6,
                     "chaque case a 3 états possibles : 3⁹ codages, pas 2⁹": 2 ** 9 * 9 * 4 / 1e6}'''),
        ("b", "megabytes, 2 decimals", "REACH_10 / 1e6",
         r'''decimals=2, mistakes={"un float32 occupe 4 octets": 5478 * 9 / 1e6}'''),
        ("c", "gigabytes, 3 decimals", "BIG_10 / 1e9",
         r'''decimals=3, mistakes={"le morpion 4 × 4 a 16 cases, donc 16 actions par ligne": 3 ** 16 * 9 * 4 / 1e9,
                     "un float32 occupe 4 octets": 3 ** 16 * 16 / 1e9}'''),
        ("d", "a whole number", "NET_10",
         r'''mistakes={"compte aussi les biais, un par neurone": NET_10 - (128 + 128 + 16),
                     "l'entrée a 16 cases × 3 états (codage one-hot), soit 48 valeurs": (16 * 128 + 128) + (128 * 128 + 128) + (128 * 16 + 16)}'''),
    ]),
    Paper("CP6.13", "Retour sur les parties IV et V : convolution et pas d'Adam", [
        ("a", 'a shape like "(1, 2, 3, 4)"', '"(1, 32, 32, 32)"',
         r'''mistakes={"le padding de 2 s'ajoute des deux côtés : (64 + 4 − 5) / 2 + 1, arrondi vers le bas": "(1, 32, 30, 30)",
                     "le stride de 2 divise la taille par deux": "(1, 32, 64, 64)",
                     "la sortie a autant de canaux que de filtres": "(1, 3, 32, 32)"}'''),
        ("b", "a whole number", "32 * 3 * 5 * 5 + 32",
         r'''mistakes={"un filtre couvre les 3 canaux d'entrée": 32 * 5 * 5 + 32,
                     "compte aussi les biais, un par filtre": 32 * 3 * 5 * 5}'''),
        ("c", 'a shape like "(1, 2, 3, 4)"', '"(1, 16, 64, 64)"',
         r'''mistakes={"c'est la formule d'une convolution ordinaire : la convolution transposée agrandit": "(1, 16, 16, 16)",
                     "le padding d'une convolution transposée retire 2p à la taille : (32 − 1) × 2 − 2 + 4": "(1, 16, 66, 66)"}'''),
        ("d", "w1, 4 decimals", "adam_13([0.5])",
         r'''decimals=4, mistakes={"la correction du biais divise m₁ par 1 − β₁ : le premier pas vaut environ lr, quelle que soit la taille du gradient": 1.0 - 0.01 * 0.05 / (math.sqrt(0.00025) + 1e-8),
                     "c'est le pas de la descente de gradient simple, w − lr g ; Adam normalise par la racine de v": 1.0 - 0.01 * 0.5}'''),
        ("e", "w2, 4 decimals", "adam_13([0.5, -0.1])",
         r'''decimals=4, mistakes={"au 2ᵉ pas, la correction du biais divise par 1 − β₁² et 1 − β₂², pas par 1 − β₁ et 1 − β₂": adam_13([0.5]) - 0.01 * (0.035 / 0.1) / (math.sqrt(0.00025975 / 0.001) + 1e-8),
                     "Adam garde la mémoire de ses moyennes : le 2ᵉ pas n'est pas un nouveau premier pas de taille lr": 0.99 + 0.01}'''),
    ]),
]

WRAP = {"CP6.1e": "shape_text", "CP6.1h": "shape_text", "CP6.13a": "shape_text", "CP6.13c": "shape_text"}
WRAP.update({f"CP6.7{letter}": "letters_text" for letter in "abc"})

SUMMARY = r'''# Summary of the automatic checks
import re


def natural(sub_id):
    """CP6.2a before CP6.10a: the question number as a number, then the letter."""
    number, letter = re.match(r"CP6\.(\d+)(.*)", sub_id).groups()
    return int(number), letter


if exam_over():
    done = {k: r for k, r in RESULTS.items() if r.status != "pending"}
    right = sorted((k for k, r in done.items() if r), key=natural)
    wrong = sorted((k for k, r in done.items() if not r), key=natural)
    pending = sorted((k for k, r in RESULTS.items() if r.status == "pending"), key=natural)
    print(f"Réponses vérifiées : {len(right)} juste(s) sur {len(done)} ; pas remplies : {len(pending)}.")
    if wrong:
        print("À revoir avec le corrigé :", ", ".join(wrong))
    for question in ("CP6.8", "CP6.9"):
        checked_props = [ok for _, ok in PROPERTIES.get(question, [])]
        state = " (pas encore écrite : ⏳)" if question in PENDING else ""
        print(f"Propriétés du code de {question} : {sum(checked_props)} sur {len(checked_props)}{state}.")
    print("Note ensuite ta copie avec le barème de 03_examen_corrige.md : il donne aussi les points des démarches, "
          "des justifications et des questions rédigées, que cette vérification ne voit pas.")'''


def paper_cell(kind: str, paper: Paper) -> list:
    """One paper question of part B: answers + guarded wb.check (exam), or wb.record (solutions)."""
    cells = [md(f"**{paper.id} — {paper.title}**")]
    stem = f"answer_{paper.id.replace('.', '_')}"
    if kind == "exercise":
        lines = [f"# {paper.id}: the values written on your answer sheet"]
        lines += [f"{stem}{letter} = ...  # {letter}) {hint}" for letter, hint, _, _ in paper.subs]
        names = ", ".join(f"{WRAP[paper.id + letter]}({stem}{letter})" if paper.id + letter in WRAP
                          else f"{stem}{letter}" for letter, *_ in paper.subs)
        lines += ["", "if exam_over():",
                  f"    for letter, answer in zip(\"{paper.letters}\", [{names}]):",
                  f"        RESULTS[f\"{paper.id}{{letter}}\"] = wb.check(f\"{paper.id}{{letter}}\", answer)"]
        cells.append(code("\n".join(lines)))
    else:
        lines = []
        for letter, _, expr, options in paper.subs:
            args = f", {options}" if options else ""
            lines.append(f'wb.record("{paper.id}{letter}", {expr}{args})')
            check_unique_keys(lines[-1], f"{paper.id}{letter}")
        cells.append(code("\n".join(lines), tags=["answer"]))
    return cells


def header_cells(kind: str) -> list:
    rel = EXAM if kind == "exercise" else SOLUTIONS
    if kind == "exercise":
        title = "# Checkpoint VI · Examen blanc — notebook de l'examen"
        how = ("Ce notebook accompagne le sujet `01_examen_sujet.md`.\n\n"
               "- **Partie A, pendant l'examen** : les deux questions de code, CP6.8 (🔨, 2 points : la loss d'un VAE) "
               "et CP6.9 (🐛, 1,5 point : un pas de GAN à réparer), avec PyTorch. Écris ton code dans les cellules à "
               "compléter, sans `mylearn`, sans les corrigés et sans assistant IA ; `help()` et la documentation "
               "officielle de PyTorch sont permis. Rien n'est vérifié pendant l'examen : les cellules « essai » montrent "
               "seulement ce que fait ton code. Tes réponses chiffrées et tes explications vont sur ta feuille "
               "(`04_mes_reponses.md`).\n"
               "- **Partie B, après l'examen** : la vérification automatique de ton code et des réponses de ta "
               "feuille. Ses cellules ne vérifient rien tant que `EXAM_OVER` vaut `False`.\n\n"
               "« Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche ⏳ "
               "(partie A) ou ⏸️ (partie B).\n\n"
               "> Travaille dans **ta copie** (`mon_travail/checkpoints/partie_6/02_examen_notebook.ipynb`, créée "
               "par `python tools/start_chapter.py CP6`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# Checkpoint VI · Examen blanc — solutions (notebook exécuté)"
        how = ("Les solutions des deux questions de code (CP6.8, la loss d'un VAE ; CP6.9, le pas de GAN réparé et "
               "entraîné), exécutées, puis l'enregistrement des réponses que vérifie la partie B du notebook de "
               "l'examen (cellules marquées `answer`, récoltées par `tools/build_answers.py`). Le corrigé détaillé et le "
               "barème sont dans `03_examen_corrige.md`.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell("demo"))
    cells.append(md("## Partie A · Pendant l'examen : les questions de code" if kind == "exercise"
                    else "## Partie A · Les questions de code"))
    cells.append(code(GIVEN))
    cells.append(md(CP68_MD))
    if kind == "exercise":
        cells += [code(CP68_TODO), code(TRY_TOOL)]
        cells.append(md(CP69_MD))
        cells += [code(CP69_GIVEN), code(CP69_TODO), code(CP69_TRY)]
        cells.append(md("---\n\n### 🛑 Fin de la partie A\n\nL'examen s'arrête ici. Rends ta copie (enregistre ce "
                        "notebook et ta feuille), puis, seulement après, passe à la partie B."))
        cells.append(md(PART_B_INTRO))
        cells.append(code(PART_B_FLAG))
        cells.append(md("### Le code de la partie A"))
        cells += [code(CODE_CHECK_8), code(CODE_CHECK_9)]
        cells.append(md("### Les réponses de ta feuille"))
        for paper in PAPER:
            cells += paper_cell(kind, paper)
        cells.append(md("### Bilan"))
        cells.append(code(SUMMARY))
        cells.append(md("**Ce que cette vérification ne voit pas** : les démarches et les justifications ; les "
                        "explications (CP6.2 e, CP6.3 e, CP6.4 a, d et e, CP6.5 c, CP6.6 c, CP6.10 e, CP6.11, CP6.12, "
                        "CP6.13 f) ; et, pour CP6.8, les nombres recopiés sur ta feuille (seule ta fonction est "
                        "vérifiée). Tout cela se corrige avec `03_examen_corrige.md`. Ensuite : la remédiation du "
                        "corrigé (relis la synthèse, `05_synthese.md`, pour les notions ratées), puis le mini-projet."))
    else:
        cells += [code(CP68_SOLUTION), code(CP68_RECORD, tags=["answer"])]
        cells.append(md(CP69_MD))
        cells += [code(CP69_GIVEN), code(CP69_SOLUTION), code(CP69_RECORD, tags=["answer"])]
        cells.append(md("## Partie B · Les réponses vérifiées après l'examen\n\n"
                        "Chaque réponse est calculée à partir des nombres du sujet ; les erreurs classiques reçoivent "
                        "un message qui dit où chercher."))
        cells.append(code(PAPER_CONTEXT))
        for paper in PAPER:
            cells += paper_cell(kind, paper)
    return cells


def main() -> int:
    write_notebook(EXAM, build("exercise"))
    write_notebook(SOLUTIONS, build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER)
    print(f"✅ {EXAM} and {SOLUTIONS} written ({len(PAPER)} paper questions, {n_checks} paper checks, "
          "2 code questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
