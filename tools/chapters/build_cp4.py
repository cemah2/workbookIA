#!/usr/bin/env python
"""Build the two notebooks of the mock exam of part IV from a single source (used by Claude).

    python tools/chapters/figures_cp4.py
    python tools/chapters/build_cp4.py
    python tools/run_all_notebooks.py checkpoints/partie_4/03_examen_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py checkpoints/partie_4/02_examen_notebook.ipynb

The exam notebook has two parts. Part A is done DURING the exam: the code question CP4.9 (🔨, NumPy
only: an epoch of training with AdamW and gradient clipping), without mylearn, without PyTorch and
without any check. Part B is done AFTER the exam: it checks the code of part A against a NumPy
transcription of torch.optim.AdamW (and against PyTorch itself when it is installed), then the answers
written on paper (sub-IDs CP4.1a, CP4.2a, ...). Its cells only run once ``EXAM_OVER = True``. The
solutions notebook records every checked answer with ``wb.record``.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import Paper, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "checkpoints/partie_4"
EXAM = f"{FOLDER}/02_examen_notebook.ipynb"
SOLUTIONS = f"{FOLDER}/03_examen_solutions.ipynb"

# ---------------------------------------------------------------------------
# Data and tools of the code question
# ---------------------------------------------------------------------------
GIVEN = r'''# Data and tools of the code question CP4.9 (run this cell first)
import numpy as np

# 2 048 images of Fashion-MNIST, flattened to 784 values in [0, 1] (float64), and their labels 0-9
X_F, Y_F = wb.datasets.load_fashion_mnist("train", n=2048, flatten=True, normalize=True, seed=49)
X_F = X_F.astype(np.float64)
LR, BATCH, MAX_NORM = 1e-2, 100, 1.0          # the settings of the try (CP4.9 d)


def init_params(seed=49):
    """The starting parameters [W, b] of the softmax regression 784 -> 10 (new arrays at each call)."""
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 0.01, size=(784, 10)), np.zeros(10)]


def loss_and_grads(params, X, y):
    """(mean cross-entropy of the mini-batch, [dW, db]) of the softmax regression z = X @ W + b."""
    W, b = params
    z = X @ W + b
    z = z - z.max(axis=1, keepdims=True)                       # log-sum-exp trick (ch. 17)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y]).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1.0                                  # dL/dz = (p - one_hot(y)) / n  (ch. 18)
    d /= n
    return float(loss), [X.T @ d, d.sum(axis=0)]


def accuracy(params, X, y):
    """Share of the images whose largest logit is the right class."""
    return float(((X @ params[0] + params[1]).argmax(axis=1) == y).mean())


_ORIGINALS = {"X_F": X_F.copy(), "Y_F": Y_F.copy()}         # part B restores them if a try changed them
print("X_F:", X_F.shape, "· Y_F:", Y_F.shape, "· starting loss:", round(loss_and_grads(init_params(), X_F, Y_F)[0], 4))'''

CP49_MD = r'''### CP4.9 — Écrire de mémoire une epoch d'entraînement avec AdamW et clipping, en NumPy 🔨 ★★ ⏱️ 15 min · 2 points
**Objectif :** écrire sans aide le cœur d'une boucle d'entraînement : clipping par la norme globale, un pas d'AdamW, une epoch en mini-batches.

Avec NumPy seulement. Une régression softmax (784 → 10) sur les 2 048 images de `X_F` et leurs labels `Y_F` ; ses paramètres sont la liste `params = [W, b]` (`init_params()` en donne une nouvelle à chaque appel). `loss_and_grads(params, X, y)` rend la cross-entropy moyenne du mini-batch et la liste `[dW, db]` de ses gradients ; `accuracy(params, X, y)` rend l'accuracy.

a) `clip_by_global_norm(grads, max_norm)` rend `(clipped, norm)` : `norm`, la norme globale (racine de la somme des carrés de **tous** les gradients) ; `clipped`, une liste de **nouveaux** tableaux, les gradients multipliés par $\min\left(1, \frac{\text{max\_norm}}{\text{norm} + 10^{-6}}\right)$. Elle ne modifie pas `grads`. **[0,5]**
b) `adamw_step(params, grads, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)` fait un pas d'AdamW **sur place**, avec les conventions de `torch.optim.AdamW` : $t \leftarrow t + 1$ ; $w \leftarrow w\,(1 - \eta\lambda)$ ; $m \leftarrow \beta_1 m + (1 - \beta_1)\,g$ ; $v \leftarrow \beta_2 v + (1 - \beta_2)\,g^2$ ; $w \leftarrow w - \eta\,\frac{m / (1 - \beta_1^t)}{\sqrt{v / (1 - \beta_2^t)} + \epsilon}$. `state`, un dictionnaire vide au premier appel, garde ce qui sert d'un pas à l'autre. **[0,7]**
c) `train_epoch(params, X, y, state, lr, batch_size, max_norm, rng)` fait une epoch : `order = rng.permutation(len(X))`, puis les mini-batches de `batch_size` exemples dans cet ordre (le dernier peut être plus petit) ; pour chacun, `loss_and_grads`, le clipping, puis `adamw_step` (les autres réglages par défaut). Elle rend la loss moyenne de l'epoch par **exemple** (chaque loss de mini-batch pesée par son nombre d'exemples), un `float`. **[0,5]**
d) La cellule d'essai entraîne une epoch avec `LR`, `BATCH` et `MAX_NORM`, et `np.random.default_rng(49)` : reporte sur ta feuille $[\text{loss}, \text{accuracy}]$ (4 et 3 décimales). **[0,3]**'''

CP49_TODO = r'''def clip_by_global_norm(grads, max_norm):
    """(clipped, norm): new arrays scaled by min(1, max_norm / (norm + 1e-6)), and the global norm; CP4.9 a)."""
    raise NotImplementedError("clip_by_global_norm")  # TODO CP4.9 a)


def adamw_step(params, grads, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """One AdamW step, in place on the arrays of params, like torch.optim.AdamW; state is a dict; CP4.9 b)."""
    raise NotImplementedError("adamw_step")  # TODO CP4.9 b)


def train_epoch(params, X, y, state, lr, batch_size, max_norm, rng):
    """One epoch in the order rng.permutation(len(X)); returns the mean loss per example, a float; CP4.9 c)."""
    raise NotImplementedError("train_epoch")  # TODO CP4.9 c)'''

CP49_TRY = r'''# Try your functions (no check during the exam: compare with what you expect)
with wb.attempt("CP4.9"):
    clipped, norm = clip_by_global_norm([np.array([3.0, 4.0]), np.array([12.0])], 6.5)
    print("clip of [3, 4], [12] at 6.5:", [c.round(4).tolist() for c in clipped], "· norm", norm)
with wb.attempt("CP4.9"):
    params_try, state_try = init_params(), {}
    loss_try = train_epoch(params_try, X_F, Y_F, state_try, LR, BATCH, MAX_NORM, np.random.default_rng(49))
    print(f"one epoch: mean loss {loss_try:.4f} · accuracy {accuracy(params_try, X_F, Y_F):.3f}")'''

CP49_SOLUTION = r'''def clip_by_global_norm(grads, max_norm):
    """(clipped, norm): new arrays scaled by min(1, max_norm / (norm + 1e-6)), and the global norm; CP4.9 a)."""
    norm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads)))
    coef = min(1.0, max_norm / (norm + 1e-6))
    return [g * coef for g in grads], norm                     # g * coef is a new array: grads stay intact


def adamw_step(params, grads, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """One AdamW step, in place on the arrays of params, like torch.optim.AdamW; state is a dict; CP4.9 b)."""
    beta1, beta2 = betas
    if not state:                                              # first call: t = 0 and zero moments
        state["t"] = 0
        state["m"] = [np.zeros_like(p) for p in params]
        state["v"] = [np.zeros_like(p) for p in params]
    state["t"] += 1
    t = state["t"]
    for p, g, m, v in zip(params, grads, state["m"], state["v"]):
        p *= 1 - lr * weight_decay                             # decoupled weight decay, before the Adam step
        m *= beta1
        m += (1 - beta1) * g
        v *= beta2
        v += (1 - beta2) * g * g
        p -= lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)


def train_epoch(params, X, y, state, lr, batch_size, max_norm, rng):
    """One epoch in the order rng.permutation(len(X)); returns the mean loss per example, a float; CP4.9 c)."""
    order = rng.permutation(len(X))
    total = 0.0
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        loss, grads = loss_and_grads(params, X[idx], y[idx])
        grads, _ = clip_by_global_norm(grads, max_norm)
        adamw_step(params, grads, state, lr)
        total += loss * len(idx)                               # weighted: the last mini-batch has 48 images
    return total / len(X)


params_9, state_9 = init_params(), {}
loss_9 = train_epoch(params_9, X_F, Y_F, state_9, LR, BATCH, MAX_NORM, np.random.default_rng(49))
acc_9 = accuracy(params_9, X_F, Y_F)
print(f"one epoch ({state_9['t']} steps): mean loss {loss_9:.4f} · accuracy {acc_9:.4f}")
try:                                                           # the same epoch with PyTorch, as a cross-check
    import torch
    W_t, b_t = (torch.tensor(a, requires_grad=True) for a in init_params())
    opt_t = torch.optim.AdamW([W_t, b_t], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    order_t = np.random.default_rng(49).permutation(len(X_F))
    X_t, Y_t = torch.from_numpy(X_F), torch.from_numpy(Y_F)
    for start in range(0, len(X_F), BATCH):
        idx = torch.from_numpy(order_t[start:start + BATCH])
        loss_t = torch.nn.functional.cross_entropy(X_t[idx] @ W_t + b_t, Y_t[idx])
        opt_t.zero_grad()
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_([W_t, b_t], MAX_NORM)
        opt_t.step()
    gap_t = max(np.abs(W_t.detach().numpy() - params_9[0]).max(), np.abs(b_t.detach().numpy() - params_9[1]).max())
    print(f"torch.optim.AdamW + clip_grad_norm_ on the same mini-batches: largest gap {gap_t:.1e}")
except ImportError:
    print("(PyTorch is not installed: no cross-check)")'''

CP49_RECORD = r'''def variant_9(*, clip=True, weighted=True, coupled=False, bias_correction=True, decay_after=False):
    """[loss, accuracy] of the epoch of CP4.9 written another (wrong) way."""
    params, state = init_params(), {"t": 0, "m": [np.zeros((784, 10)), np.zeros(10)], "v": [np.zeros((784, 10)), np.zeros(10)]}
    order = np.random.default_rng(49).permutation(len(X_F))
    losses, sizes = [], []
    for start in range(0, len(X_F), BATCH):
        idx = order[start:start + BATCH]
        loss, grads = loss_and_grads(params, X_F[idx], Y_F[idx])
        if clip:
            grads, _ = clip_by_global_norm(grads, MAX_NORM)
        state["t"] += 1
        t = state["t"]
        for p, g, m, v in zip(params, grads, state["m"], state["v"]):
            if coupled:
                g = g + 0.01 * p
            elif not decay_after:
                p *= 1 - LR * 0.01
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            m_hat, v_hat = (m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)) if bias_correction else (m, v)
            p -= LR * m_hat / (np.sqrt(v_hat) + 1e-8)
            if decay_after:
                p *= 1 - LR * 0.01
        losses.append(loss)
        sizes.append(len(idx))
    mean = np.average(losses, weights=sizes) if weighted else float(np.mean(losses))
    return [float(mean), accuracy(params, X_F, Y_F)]


wb.record("CP4.9d", [loss_9, acc_9], decimals=4, mistakes={
    "le clipping n'est pas appliqué : chaque gradient passe par clip_by_global_norm avant le pas d'AdamW":
    variant_9(clip=False),
    "la loss moyenne doit peser chaque mini-batch par son nombre d'exemples (le dernier n'en a que 48)":
    variant_9(weighted=False),
    "c'est le weight decay couplé d'Adam (λw ajouté au gradient) : AdamW multiplie w par (1 − ηλ), à part":
    variant_9(coupled=True),
    "les moments ne sont pas corrigés du biais : divise m par 1 − β₁ᵗ et v par 1 − β₂ᵗ":
    variant_9(bias_correction=False)})'''

# ---------------------------------------------------------------------------
# Part B: checks after the exam
# ---------------------------------------------------------------------------
PART_B_INTRO = r'''## Partie B · Après l'examen : vérification automatique

**Ne commence cette partie qu'une fois ta copie terminée**, examen rendu. Elle vérifie le code de la partie A, puis les réponses que tu as écrites sur ta feuille, question par question : ✅ juste, ❌ faux (avec une piste), ⏳ pas rempli. Elle ne note pas : les démarches, les justifications et les questions rédigées se corrigent avec `03_examen_corrige.md`, qui donne aussi le barème.

1. Passe `EXAM_OVER` à `True` dans la cellule ci-dessous, puis exécute-la.
2. Exécute la vérification du code (CP4.9).
3. Reporte ensuite chaque réponse de ta feuille : **la valeur** que tu as écrite, pas un nouveau calcul (sinon tu ne vérifies plus ta copie). En Python, le séparateur décimal est un **point** (`0.125`), une matrice s'écrit comme une liste de lignes (`[[1, 2], [3, 4]]`), et les lettres s'écrivent entre guillemets (`"XYZ"`).

Si tu as fermé le notebook depuis l'examen, exécute d'abord la cellule de configuration (tout en haut) et les cellules de la partie A, sauf les essais.'''

PART_B_FLAG = r'''EXAM_OVER = False   # set it to True once your exam is finished, then run the cells of part B

import re

RESULTS = {}        # sub-ID -> result of wb.check
PROPERTIES = {}     # question -> [(property, passed)], reset each time its check cell runs
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


def verdict(question, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message, and remember the result."""
    PROPERTIES.setdefault(question, []).append((success or failure, bool(ok)))
    print(f"✅ {question} : {success}" if ok else f"❌ {question} : {failure}")
    return bool(ok)


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


def as_letters(value):
    """Letters typed as "ACE", "A, C et E" or ["A", "C", "E"] -> "ACE" (`...` stays `...`)."""
    if value is ... or value is None:
        return value
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    value = re.sub(r"(?i)\b(et|and|ou|or)\b", " ", str(value))
    return "".join(ch for ch in value if ch.isalpha()).upper()


def as_sorted_letters(value):
    """A set of letters typed in any order -> its letters in alphabetical order."""
    letters = as_letters(value)
    return "".join(sorted(letters)) if isinstance(letters, str) else letters


def as_shape(value):
    """A shape typed as "(16, 32)", "[16, 32]", "16, 32", torch.Size([16, 32]) or a tuple -> "(16, 32)"."""
    if isinstance(value, (tuple, list)) and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return str(tuple(value))
    if isinstance(value, str) and re.fullmatch(r"\s*(torch\.Size)?[\s(\[]*\d+(\s*,\s*\d+)*\s*,?[\s)\]]*\s*", value):
        return str(tuple(int(n) for n in re.findall(r"\d+", value)))
    return value


def restore_data(question, names):
    """Put back the data of part A if a try of part A changed them in place (a function that writes into its input)."""
    changed = [name for name in names
               if name not in globals() or not np.array_equal(globals()[name], _ORIGINALS[name])]
    for name in changed:
        globals()[name] = _ORIGINALS[name].copy()
    if changed:
        print(f"⚠️ {question} : {', '.join(changed)} avai{'en' if len(changed) > 1 else ''}t changé depuis la cellule des "
              f"données (une fonction de la partie A écrit-elle dans ce qu'elle reçoit ?) : la vérification repart "
              f"des données d'origine.")'''

CODE_CHECK_9 = r'''# Checks of clip_by_global_norm, adamw_step and train_epoch (CP4.9)
if exam_over() and ready("CP4.9", "np", "X_F", "Y_F", "init_params", "loss_and_grads", "accuracy", "_ORIGINALS",
                         "clip_by_global_norm", "adamw_step", "train_epoch"):
    PROPERTIES["CP4.9"] = []
    RESULTS.pop("CP4.9d", None)
    restore_data("CP4.9", ["X_F", "Y_F"])

    def oracle_step(params, grads, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """torch.optim.AdamW transcribed in NumPy (its pseudo-code, PyTorch 2.11), in place."""
        if not state:
            state.update(t=0, m=[np.zeros_like(p) for p in params], v=[np.zeros_like(p) for p in params])
        state["t"] += 1
        for p, g, m, v in zip(params, grads, state["m"], state["v"]):
            p *= 1 - lr * weight_decay
            m[...] = betas[0] * m + (1 - betas[0]) * g
            v[...] = betas[1] * v + (1 - betas[1]) * g * g
            p -= lr * (m / (1 - betas[0] ** state["t"])) / (np.sqrt(v / (1 - betas[1] ** state["t"])) + eps)

    def safe(function, *args, **kwargs):
        """(True, result) or (False, the text of the error): a bug of the exam code must not stop part B."""
        try:
            return True, function(*args, **kwargs)
        except NotImplementedError:
            return None, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    # a) clip_by_global_norm
    grads_a = [np.array([3.0, 4.0]), np.array([12.0])]
    ok, out = safe(clip_by_global_norm, grads_a, 6.5)
    if ok is None:
        print("⏳ CP4.9 a) : clip_by_global_norm n'est pas encore écrite.")
    elif not ok:
        verdict("CP4.9", False, "", f"clip_by_global_norm lève une erreur : {out}")
    elif out is None:
        verdict("CP4.9", False, "", "clip_by_global_norm ne renvoie rien : as-tu oublié le return ?")
    elif not (isinstance(out, (tuple, list)) and len(out) == 2):
        verdict("CP4.9", False, "", f"clip_by_global_norm doit renvoyer le couple (clipped, norm), pas {out!r}"[:300])
    else:
        clipped, norm = out
        verdict("CP4.9", np.array_equal(grads_a[0], [3.0, 4.0]) and np.array_equal(grads_a[1], [12.0]),
                "clip_by_global_norm ne modifie pas les gradients qu'elle reçoit.",
                "clip_by_global_norm modifie grads (par exemple avec *=) : elle doit rendre de nouveaux tableaux.")
        try:
            norm_ok = abs(float(norm) - 13.0) < 1e-9
        except (TypeError, ValueError):
            norm_ok = False
        verdict("CP4.9", norm_ok, "la norme globale de [3, 4] et [12] vaut 13.",
                f"la norme rendue vaut {norm!r} : la norme GLOBALE est la racine de la somme des carrés de tous les "
                "gradients (3² + 4² + 12² = 169, donc 13), pas une norme par tableau.")
        coef = 6.5 / (13 + 1e-6)
        try:
            values_ok = (len(clipped) == 2 and np.allclose(clipped[0], [3 * coef, 4 * coef], rtol=1e-9, atol=0)
                         and np.allclose(clipped[1], [12 * coef], rtol=1e-9, atol=0))
        except (TypeError, ValueError):
            values_ok = False
        verdict("CP4.9", values_ok, "au-dessus de max_norm, les gradients sont multipliés par max_norm / (norm + 1e-6).",
                "avec max_norm = 6,5 et une norme de 13, chaque gradient doit être multiplié par 6,5 / (13 + 1e-6), "
                f"le même coefficient pour tous ; reçu {clipped!r}"[:300])
        ok_low, low = safe(clip_by_global_norm, [np.array([3.0, 4.0]), np.array([12.0])], 20.0)
        verdict("CP4.9", ok_low and isinstance(low, (tuple, list)) and len(low) == 2
                and np.array_equal(low[0][0], [3.0, 4.0]) and np.array_equal(low[0][1], [12.0]),
                "sous max_norm, les gradients ne changent pas (coefficient 1).",
                "avec max_norm = 20 > 13, les gradients ne doivent pas changer : le coefficient est min(1, ...).")

    # b) adamw_step against the transcription of torch.optim.AdamW
    rng_b = np.random.default_rng(491)
    start_b = [rng_b.normal(size=(3, 2)), rng_b.normal(size=2)]
    grads_b = [[rng_b.normal(size=(3, 2)), rng_b.normal(size=2)] for _ in range(3)]
    mine_b, want_b, state_mine, state_want = [p.copy() for p in start_b], [p.copy() for p in start_b], {}, {}
    objects_b = [id(p) for p in mine_b]
    ok, out = True, None
    for step_b, g in enumerate(grads_b):
        ok, out = safe(adamw_step, mine_b, [x.copy() for x in g], state_mine, 0.01, betas=(0.8, 0.95), eps=1e-6,
                       weight_decay=0.1)
        oracle_step(want_b, [x.copy() for x in g], state_want, 0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.1)
        if not ok:
            break
    if ok is None:
        print("⏳ CP4.9 b) : adamw_step n'est pas encore écrite.")
    elif not ok:
        verdict("CP4.9", False, "", f"adamw_step lève une erreur : {out}")
    else:
        verdict("CP4.9", [id(p) for p in mine_b] == objects_b and not np.array_equal(mine_b[0], start_b[0]),
                "adamw_step modifie les paramètres sur place.",
                "les tableaux de params n'ont pas changé : adamw_step doit les modifier SUR PLACE (p -= ..., p *= ...), "
                "pas créer de nouveaux tableaux.")
        gap_b = max(np.abs(a - b).max() for a, b in zip(mine_b, want_b))
        verdict("CP4.9", gap_b < 1e-10,
                "trois pas d'AdamW (betas, eps et weight_decay reçus en arguments) : les valeurs de torch.optim.AdamW.",
                f"après trois pas (lr 0,01, betas (0,8 ; 0,95), eps 1e-6, weight_decay 0,1), l'écart à torch.optim.AdamW "
                f"vaut {gap_b:.2e} : vérifie le compteur t (incrémenté avant de corriger le biais), le weight decay "
                "découplé (w × (1 − ηλ) avant le pas), la correction de biais, et que tu utilises les betas et eps reçus.")
        frozen = [np.array([2.0, -4.0])]
        ok_d, out_d = safe(adamw_step, frozen, [np.zeros(2)], {}, 0.1, weight_decay=0.5)
        verdict("CP4.9", ok_d and np.allclose(frozen[0], [1.9, -3.8], rtol=1e-12, atol=0),
                "avec un gradient nul, seul le weight decay agit : w × (1 − ηλ).",
                f"avec un gradient nul, lr 0,1 et weight_decay 0,5, w = [2, −4] doit devenir [1,9 ; −3,8] (w × (1 − ηλ), "
                f"le pas d'Adam étant nul) ; reçu {frozen[0]!r}.")

    # c) train_epoch against the same epoch with the oracle
    params_c = init_params()
    ok, loss_c = safe(train_epoch, params_c, X_F.copy(), Y_F.copy(), {}, LR, BATCH, MAX_NORM, np.random.default_rng(49))
    if ok is None:
        print("⏳ CP4.9 c) : train_epoch (ou une fonction qu'elle appelle) n'est pas encore écrite.")
    elif not ok:
        verdict("CP4.9", False, "", f"train_epoch lève une erreur : {loss_c}")
    elif loss_c is None:
        verdict("CP4.9", False, "", "train_epoch ne renvoie rien : as-tu oublié le return ?")
    else:
        want_c, state_c = init_params(), {}
        order_c = np.random.default_rng(49).permutation(len(X_F))
        total_c = 0.0
        for start in range(0, len(X_F), BATCH):
            idx = order_c[start:start + BATCH]
            loss_i, grads_i = loss_and_grads(want_c, X_F[idx], Y_F[idx])
            norm_i = np.sqrt(sum(np.sum(g ** 2) for g in grads_i))
            oracle_step(want_c, [g * min(1.0, MAX_NORM / (norm_i + 1e-6)) for g in grads_i], state_c, LR)
            total_c += loss_i * len(idx)
        gap_c = max(np.abs(a - b).max() for a, b in zip(params_c, want_c))
        verdict("CP4.9", isinstance(loss_c, float), "train_epoch rend un float.",
                f"train_epoch rend {type(loss_c).__name__} : attendu un float Python (return float(...)).")
        verdict("CP4.9", gap_c < 1e-10,
                "après l'epoch, les paramètres sont ceux de torch.optim.AdamW et clip_grad_norm_ sur les mêmes mini-batches.",
                f"après l'epoch, l'écart aux paramètres attendus vaut {gap_c:.2e} : l'ordre rng.permutation(len(X)), des "
                "mini-batches consécutifs de batch_size, le clipping de CHAQUE gradient avant le pas, un seul state pour "
                "toute l'epoch, et les réglages par défaut d'adamw_step.")
        try:
            weighted_ok = abs(float(loss_c) - total_c / len(X_F)) < 1e-9
        except (TypeError, ValueError):
            weighted_ok = False
        verdict("CP4.9", weighted_ok, "la loss rendue est la moyenne par exemple (le dernier mini-batch, de 48 images, pèse 48).",
                f"train_epoch rend {loss_c!r} ; la moyenne par exemple vaut {total_c / len(X_F):.6f} : pèse chaque loss de "
                "mini-batch par son nombre d'exemples.")
        try:
            import torch                                    # the same check against PyTorch itself, when installed
            W_t, b_t = (torch.tensor(a, requires_grad=True) for a in init_params())
            opt_t = torch.optim.AdamW([W_t, b_t], lr=LR, weight_decay=0.01)
            X_t, Y_t = torch.from_numpy(X_F), torch.from_numpy(Y_F)
            for start in range(0, len(X_F), BATCH):
                idx = torch.from_numpy(order_c[start:start + BATCH])
                loss_t = torch.nn.functional.cross_entropy(X_t[idx] @ W_t + b_t, Y_t[idx])
                opt_t.zero_grad()
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_([W_t, b_t], MAX_NORM)
                opt_t.step()
            gap_t = max(np.abs(W_t.detach().numpy() - params_c[0]).max(), np.abs(b_t.detach().numpy() - params_c[1]).max())
            print(f"   (PyTorch : écart de tes paramètres à torch.optim.AdamW + clip_grad_norm_ : {gap_t:.1e})")
        except ImportError:
            pass
        code_check("CP4.9d", [loss_c, accuracy(params_c, X_F, Y_F)],
                   "une epoch dans l'ordre rng.permutation, clipping puis AdamW à chaque mini-batch, loss pesée par "
                   "le nombre d'exemples.")'''

PAPER_CONTEXT = r'''# The paper questions only use the numbers of their statements (01_examen_sujet.md)
import math

import numpy as np

# CP4.1
LAYERS_1 = [20, 64, 32, 3]
PARAMS_1 = sum(a * b + b for a, b in zip(LAYERS_1[:-1], LAYERS_1[1:]))
SOFTMAX_1 = math.exp(2) / (math.exp(2) + 2)

# CP4.2
X_2, Y_2 = np.array([2.0, -1.0]), 2.0
W1_2, B1_2 = np.array([[1.0, -0.5], [0.5, 1.0]]), np.array([-0.5, 0.0])
W2_2, B2_2 = np.array([0.75, -1.0]), 0.5


def forward_2(W1, b1, W2, b2, relu=True):
    z = X_2 @ W1 + b1
    h = np.maximum(z, 0) if relu else z
    return z, h, float(h @ W2 + b2)


Z_2, H_2, YHAT_2 = forward_2(W1_2, B1_2, W2_2, B2_2)
LOSS_2 = 0.5 * (YHAT_2 - Y_2) ** 2
DELTA_2 = YHAT_2 - Y_2
DW2_2, DB2_2 = H_2 * DELTA_2, DELTA_2
DH_2 = DELTA_2 * W2_2
DZ_2 = DH_2 * (Z_2 > 0)
DW1_2 = np.outer(X_2, DZ_2)
ETA_2 = 0.1
W1N_2, B1N_2, W2N_2, B2N_2 = W1_2 - ETA_2 * DW1_2, B1_2 - ETA_2 * DZ_2, W2_2 - ETA_2 * DW2_2, B2_2 - ETA_2 * DB2_2


def loss_after_2(W1, b1, W2, b2):
    return 0.5 * (forward_2(W1, b1, W2, b2)[2] - Y_2) ** 2


# CP4.3


def momentum_3(dampening=0.0):
    w, b, history = 1.0, None, []
    for g in (2.0, 1.0):
        b = g if b is None else 0.9 * b + (1 - dampening) * g
        w -= 0.1 * b
        history.append((b, w))
    return history


def adam_3(steps=2, weight_decay=0.0, decoupled=True, bias_correction=True, decay_after=False):
    w, m, v, out = 1.0, 0.0, 0.0, []
    for t, g in enumerate((2.0, 1.0)[:steps], start=1):
        if weight_decay and not decoupled:
            g = g + weight_decay * w
        if weight_decay and decoupled and not decay_after:
            w *= 1 - 0.01 * weight_decay
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        m_hat, v_hat = (m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)) if bias_correction else (m, v)
        w -= 0.01 * m_hat / math.sqrt(v_hat)
        if weight_decay and decoupled and decay_after:
            w *= 1 - 0.01 * weight_decay
        out.append((w, m_hat, v_hat, m, v))
    return out


# CP4.4
XD_4, U_4, P_4 = np.array([2.0, 4.0, 6.0, 8.0]), np.array([0.1, 0.6, 0.3, 0.9]), 0.25
DROP_4 = XD_4 * (U_4 >= P_4) / (1 - P_4)
XB_4, GAMMA_4, BETA_4, RHO_4 = np.array([2.0, 6.0, 8.0, 12.0]), 0.5, -1.0, 0.1
MEAN_4, VAR_4, UNBIASED_4 = XB_4.mean(), XB_4.var(), XB_4.var(ddof=1)
RUN_MEAN_4, RUN_VAR_4 = (1 - RHO_4) * 0 + RHO_4 * MEAN_4, (1 - RHO_4) * 1 + RHO_4 * UNBIASED_4


def bn_train_4(x, var):
    return GAMMA_4 * (x - MEAN_4) / math.sqrt(var) + BETA_4


def bn_eval_4(x, mean=RUN_MEAN_4, var=RUN_VAR_4):
    return GAMMA_4 * (x - mean) / math.sqrt(var) + BETA_4


# CP4.5
L_5 = min(n for n in range(1, 100) if 0.25 ** n < 1e-6)

# CP4.6
PARAMS_6 = 784 * 1024 + 1024 + 1024 * 1024 + 1024 + 1024 * 10 + 10
STEPS_6 = math.ceil(60000 / 256) * 10

# CP4.12
P_12 = 0.7
TRUE_12 = [0.7, 0.5, 0.9]
XR_12, YR_12 = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 7.0])
FLIPPERS_12, TEST_12 = np.array([182.0, 191.0, 197.0, 210.0]), 216.0
WITH_TEST_12 = np.append(FLIPPERS_12, TEST_12)'''


def _letters_mistakes(answer: str, messages: dict) -> str:
    """mistakes={message: letter} for every other choice of a lettered question."""
    pairs = ",\n                     ".join(f'"{message}": "{letter}"' for letter, message in messages.items()
                                            if letter != answer)
    return "mistakes={" + pairs + "}"


def _bug_letters_7() -> str:
    """mistakes for the sets of letters of CP4.7 a) that differ from DIKM by one letter (missing, extra, swapped)."""
    answer, lines = set("DIKM"), []
    lines.append('"il manque une ligne fautive : une ligne crée à chaque epoch ce qui devrait vivre tout l\'entraînement": "IKM"')
    lines.append('"il manque une ligne fautive : regarde ce que reçoit loss_fn, des logits ou des probabilités ?": "DKM"')
    lines.append('"il manque une ligne fautive : dans quel ordre doivent venir zero_grad, backward et step ?": "DIM"')
    lines.append('"il manque une ligne fautive : que garde en mémoire la somme d\'un tenseur qui a un grad_fn ?": "DIK"')
    for extra in "ABCEFGHJLNOPQ":
        letters = "".join(sorted(answer | {extra}))
        lines.append(f'"une ligne de trop : la ligne ({extra}) est correcte ; relis ce qu\'elle fait": "{letters}"')
    for removed in sorted(answer):
        for extra in "ABCEFGHJLNOPQ":
            letters = "".join(sorted((answer - {removed}) | {extra}))
            if (removed, extra) == ("K", "J"):   # the same bug, blamed on the other line of the pair
                lines.append('"(J) et (K) sont bien dans le mauvais ordre, mais l\'énoncé demande la ligne qui arrive trop '
                             'tard": "DIJM"')
                continue
            lines.append(f'"trois des quatre lignes sont justes : la ligne ({extra}) est correcte, et une ligne fautive '
                         f'manque": "{letters}"')
    return ",\n                     ".join(lines)


PAPER = [
    Paper("CP4.1", "Questions flash", [
        ("a", "number of parameters", "PARAMS_1",
         r'''mistakes={"compte aussi les biais : un par neurone, dans chaque couche sauf l'entrée": PARAMS_1 - (64 + 32 + 3)}'''),
        ("b", 'a shape, as a string like "(1, 2)"', '"(16, 32)"',
         r'''mistakes={"la forme d'une sortie est (nombre d'exemples, nombre de neurones), dans cet ordre": "(32, 16)",
                     "c'est la sortie de la première couche cachée : on demande celle de la deuxième": "(16, 64)",
                     "la forme d'une sortie garde l'axe des exemples": "(32,)"}'''),
        ("c", "a list of four values", "[0.0, 0.5, 1.0, -0.2]",
         r'''decimals=2, mistakes={"tanh′(z) = 1 − tanh²(z) : en 0, la pente vaut 1": [0.0, 0.5, 0.0, -0.2],
                     "la leaky ReLU de l'énoncé a une pente de 0,1, pas 0,01": [0.0, 0.5, 1.0, -0.02],
                     "ReLU(z) = max(0, z) : une valeur négative donne 0": [-1.5, 0.5, 1.0, -0.2],
                     "σ(0) = 1 / (1 + e⁰)": [0.0, 1.0, 1.0, -0.2]}'''),
        ("d", "a probability, 3 decimals", "SOFTMAX_1",
         r'''decimals=3, mistakes={"le dénominateur somme les exponentielles des TROIS logits, e² + e⁰ + e⁰": math.exp(2) / (math.exp(2) + 1),
                     "c'est la probabilité d'un des deux autres logits : on demande la première": 1 / (math.exp(2) + 2),
                     "la softmax passe par les exponentielles : ce n'est pas 2 / (2 + 0 + 0)": 1.0}'''),
        ("e", "number of learned parameters", "128",
         r'''mistakes={"running_mean et running_var sont des buffers, pas des paramètres appris : il reste γ et β": 256,
                     "γ ET β : deux vecteurs de 64 valeurs": 64, "la batchnorm de PyTorch a bien des paramètres appris": 0}'''),
        ("f", "number of parameters", "8 * (3 * 3 * 3) + 8",
         r'''mistakes={"compte aussi les biais, un par filtre": 8 * 27,
                     "un filtre couvre les 3 canaux d'entrée : 3 × 3 × 3 poids": 8 * 9 + 8}'''),
    ]),
    Paper("CP4.2", "Passe avant, rétropropagation et un pas de SGD", [
        ("a", "[z1, z2]", "Z_2.tolist()",
         r'''decimals=4, mistakes={"n'oublie pas les biais de la couche cachée": (X_2 @ W1_2).tolist(),
                     "dans la convention z = x W, la colonne k de W porte les poids du neurone k : lis W par colonnes": (X_2 @ W1_2.T + B1_2).tolist()}'''),
        ("b", "[y_hat, L], 5 decimals", "[YHAT_2, LOSS_2]",
         r'''decimals=5, mistakes={"la loss de l'énoncé a un facteur ½ : L = ½ (ŷ − y)²": [YHAT_2, (YHAT_2 - Y_2) ** 2],
                     "un neurone caché à ReLU de somme négative sort 0, pas sa somme": [forward_2(W1_2, B1_2, W2_2, B2_2, relu=False)[2], 0.5 * (forward_2(W1_2, B1_2, W2_2, B2_2, relu=False)[2] - Y_2) ** 2]}'''),
        ("c", "delta = dL/dy_hat", "DELTA_2",
         r'''decimals=4, mistakes={"avec le facteur ½, la dérivée de ½ (ŷ − y)² est ŷ − y, sans le 2": 2 * DELTA_2}'''),
        ("d", "[dW2_1, dW2_2, db2], 4 decimals", "[*DW2_2.tolist(), DB2_2]",
         r'''decimals=4, mistakes={"le gradient d'un poids de sortie est δ × la SORTIE du neurone caché (après la ReLU), pas sa somme": [*(Z_2 * DELTA_2).tolist(), DB2_2],
                     "le biais de sortie reçoit δ × 1": [*DW2_2.tolist(), 0.0]}'''),
        ("e", "[dL/dz1, dL/dz2], 4 decimals", "DZ_2.tolist()",
         r'''decimals=4, mistakes={"n'oublie pas le facteur ReLU′(z) : 0 pour un neurone de somme négative": DH_2.tolist()}'''),
        ("f", "dL/dW1 as a list of two rows, 4 decimals", "DW1_2.tolist()",
         r'''decimals=4, mistakes={"range le gradient comme W(1) : ligne j = l'entrée x_j, colonne k = le neurone k": DW1_2.T.tolist(),
                     "n'oublie pas le facteur ReLU′(z) du neurone 2": np.outer(X_2, DH_2).tolist()}'''),
        ("g", "new W1 as a list of two rows, 5 decimals", "W1N_2.tolist()",
         r'''decimals=5, mistakes={"on descend le gradient : w ← w − η ∂L/∂w": (W1_2 + ETA_2 * DW1_2).tolist(),
                     "c'est un pas de learning rate 1 : l'énoncé prend η = 0,1": (W1_2 - DW1_2).tolist()}'''),
        ("h", "loss after the step, 5 decimals", "loss_after_2(W1N_2, B1N_2, W2N_2, B2N_2)",
         r'''decimals=5, mistakes={"le pas met à jour TOUS les paramètres, la couche de sortie comprise": loss_after_2(W1N_2, B1N_2, W2_2, B2_2),
                     "c'est la loss d'avant le pas": LOSS_2,
                     "les biais aussi font un pas": loss_after_2(W1N_2, B1_2, W2N_2, B2_2)}'''),
    ]),
    Paper("CP4.3", "Momentum, Adam et AdamW", [
        ("a", "b2", "momentum_3()[1][0]",
         r'''decimals=4, mistakes={"la convention de PyTorch n'a pas de facteur (1 − μ) : b ← μ b + g": momentum_3(dampening=0.9)[1][0],
                     "relis la convention : la vitesse du deuxième pas part de b₁ = g₁, pas de 0": 1.0}'''),
        ("b", "w2", "momentum_3()[1][1]",
         r'''decimals=4, mistakes={"le premier pas a déjà déplacé w : w₂ = w₁ − η b₂": 1 - 0.1 * momentum_3()[1][0],
                     "la convention de PyTorch n'a pas de facteur (1 − μ) : b ← μ b + g": momentum_3(dampening=0.9)[1][1]}'''),
        ("c", "w1, 5 decimals", "adam_3(1)[0][0]",
         r'''decimals=5, mistakes={"corrige m et v du biais : divise par 1 − β₁ et 1 − β₂ au premier pas": adam_3(1, bias_correction=False)[0][0]}'''),
        ("d", "[m_hat_2, v_hat_2], 3 decimals", "[adam_3()[1][1], adam_3()[1][2]]",
         r'''decimals=3, mistakes={"ce sont les moments NON corrigés : divise par 1 − β₁² et 1 − β₂²": [adam_3()[1][3], adam_3()[1][4]],
                     "au deuxième pas, t = 2 : la correction divise par 1 − β², pas par 1 − β": [adam_3()[1][3] / 0.1, adam_3()[1][4] / 0.001]}'''),
        ("e", "w2, 5 decimals", "adam_3()[1][0]",
         r'''decimals=5, mistakes={"corrige m et v du biais à chaque pas": adam_3(bias_correction=False)[1][0]}'''),
        ("f", "w1 with AdamW, 5 decimals", "adam_3(1, weight_decay=0.1)[0][0]",
         r'''decimals=5, mistakes={"AdamW multiplie w par (1 − ηλ) AVANT le pas d'Adam, comme torch.optim.AdamW": adam_3(1, weight_decay=0.1, decay_after=True)[0][0],
                     "le facteur est 1 − ηλ, avec le learning rate : 1 − 0,01 × 0,1": 0.9 * 1 - 0.01,
                     "c'est le weight decay couplé d'Adam : AdamW le découple": adam_3(1, weight_decay=0.1, decoupled=False)[0][0]}'''),
        ("g", "w1 with Adam(weight_decay=0.1), 5 decimals", "adam_3(1, weight_decay=0.1, decoupled=False)[0][0]",
         r'''decimals=5, mistakes={"c'est le weight decay découplé d'AdamW : Adam(weight_decay) ajoute λw au GRADIENT, avant la normalisation": adam_3(1, weight_decay=0.1)[0][0]}'''),
    ]),
    Paper("CP4.4", "Dropout et batchnorm", [
        ("a", "a list, 3 decimals", "DROP_4.tolist()",
         r'''decimals=3, mistakes={"le dropout INVERSÉ divise les valeurs gardées par 1 − p": (XD_4 * (U_4 >= P_4)).tolist(),
                     "une valeur est gardée quand u ≥ p, et mise à zéro sinon": (XD_4 * (U_4 < P_4) / (1 - P_4)).tolist(),
                     "on divise par la probabilité de GARDER une valeur, 1 − p": (XD_4 * (U_4 >= P_4) / P_4).tolist()}'''),
        ("b", "a list", "XD_4.tolist()",
         r'''decimals=3, mistakes={"en évaluation, le dropout inversé ne fait rien : il rend son entrée": (XD_4 / (1 - P_4)).tolist(),
                     "c'est le dropout « non inversé » du livre : PyTorch multiplie déjà en entraînement, et ne touche à rien en évaluation": (XD_4 * (1 - P_4)).tolist(),
                     "en évaluation, aucune valeur n'est mise à zéro": DROP_4.tolist()}'''),
        ("c", "the mean of a)", "float(DROP_4.mean())",
         r'''decimals=3, mistakes={"c'est la moyenne de x : on demande celle des quatre valeurs de a)": float(XD_4.mean())}'''),
        ("d", "[mean, biased variance]", "[MEAN_4, VAR_4]",
         r'''decimals=4, mistakes={"la batchnorm normalise avec la variance BIAISÉE (diviser par m)": [MEAN_4, UNBIASED_4],
                     "c'est l'écart-type : on demande la variance": [MEAN_4, math.sqrt(VAR_4)]}'''),
        ("e", "output for 12, 4 decimals", "bn_train_4(12.0, VAR_4)",
         r'''decimals=4, mistakes={"la normalisation utilise la variance biaisée du mini-batch": bn_train_4(12.0, UNBIASED_4),
                     "applique ensuite γ et β : y = γ x̂ + β": (12.0 - MEAN_4) / math.sqrt(VAR_4)}'''),
        ("f", "[running mean, running var], 4 decimals", "[RUN_MEAN_4, RUN_VAR_4]",
         r'''decimals=4, mistakes={"la variance glissante se met à jour avec la variance NON biaisée du mini-batch": [RUN_MEAN_4, (1 - RHO_4) * 1 + RHO_4 * VAR_4],
                     "la mise à jour garde 1 − ρ = 0,9 de l'ancienne valeur et prend ρ = 0,1 du mini-batch": [RHO_4 * 0 + (1 - RHO_4) * MEAN_4, RHO_4 * 1 + (1 - RHO_4) * UNBIASED_4],
                     "la couche est neuve : sa variance glissante part de 1, pas de 0": [RUN_MEAN_4, RHO_4 * UNBIASED_4]}'''),
        ("g", "eval output for 7, 4 decimals", "bn_eval_4(7.0)",
         r'''decimals=4, mistakes={"en évaluation, la batchnorm utilise ses statistiques GLISSANTES, pas celles d'un mini-batch": BETA_4,
                     "applique ensuite γ et β : y = γ x̂ + β": (7.0 - RUN_MEAN_4) / math.sqrt(RUN_VAR_4)}'''),
    ]),
    Paper("CP4.5", "Le gradient à travers des sigmoïdes empilées (c)", [
        ("c", "the smallest number of layers", "L_5",
         r'''fractional="on demande un nombre de couches", mistakes={"calcule (1/4)⁹ : la borne dépasse encore 10⁻⁶": 9,
                     "la pente de la sigmoïde est au plus 1/4, pas 1/2": min(n for n in range(1, 100) if 0.5 ** n < 1e-6)}'''),
    ]),
    Paper("CP4.6", "Fermi : paramètres et mémoire", [
        ("a", "parameters in millions, 1 decimal", "PARAMS_6 / 1e6",
         r'''decimals=1, mistakes={"en millions : divise le nombre de paramètres par 10⁶": PARAMS_6 / 1e5}'''),
        ("b", "memory in MB (an integer)", "round(PARAMS_6 * 4 * 4 / 1e6)",
         r'''fractional="arrondis à l'entier", mistakes={"ce sont les paramètres seuls : ajoute les gradients et les deux moments d'AdamW": round(PARAMS_6 * 4 / 1e6),
                     "paramètres et gradients seulement : AdamW garde aussi ses deux moments": round(PARAMS_6 * 4 * 2 / 1e6),
                     "il manque un des quatre tableaux de même taille (paramètres, gradients, m, v)": round(PARAMS_6 * 4 * 3 / 1e6),
                     "un mégaoctet vaut 10⁶ octets ici, pas 2²⁰": round(PARAMS_6 * 16 / 2 ** 20)}'''),
        ("c", "the factor", "2",
         r'''fractional="un rapport entier", mistakes={"compte les tableaux de la taille des paramètres : 4 pour AdamW, 2 sans moments": 4,
                     "AdamW garde DEUX moments par paramètre, m et v : quatre tableaux contre deux": 3}'''),
        ("d", "number of updates", "STEPS_6",
         r'''fractional="un nombre de mises à jour", mistakes={"le dernier mini-batch, incomplet, est aussi une mise à jour : arrondis au-dessus": (60000 // 256) * 10,
                     "c'est une seule epoch : on en demande 10": STEPS_6 // 10}'''),
    ]),
    Paper("CP4.12", "Parties I à III", [
        ("a", "log loss in nats, 3 decimals", "-math.log(P_12)",
         r'''decimals=3, mistakes={"la cross-entropy ne regarde que la probabilité de la VRAIE classe : −ln 0,7": -(0.2 * math.log(0.2) + 0.1 * math.log(0.1) + 0.7 * math.log(0.7)),
                     "en nats : le logarithme népérien, pas log₁₀": -math.log10(P_12),
                     "c'est en bits : on la demande en nats (ln)": -math.log2(P_12)}'''),
        ("b", "log loss in bits, 3 decimals", "-math.log2(P_12)",
         r'''decimals=3, mistakes={"c'est en nats : en bits, log₂ (diviser par ln 2)": -math.log(P_12),
                     "en bits : log₂, pas log₁₀": -math.log10(P_12)}'''),
        ("c", "mean log loss, 4 decimals", "float(np.mean([-math.log(q) for q in TRUE_12]))",
         r'''decimals=4, mistakes={"on fait la moyenne des −ln p, pas −ln de la moyenne des p": -math.log(np.mean(TRUE_12)),
                     "une moyenne, pas une somme": float(np.sum([-math.log(q) for q in TRUE_12]))}'''),
        ("d", "[w for alpha = 0, w for alpha = 7], 3 decimals", "[float(XR_12 @ YR_12 / (XR_12 @ XR_12)), float(XR_12 @ YR_12 / (XR_12 @ XR_12 + 7))]",
         r'''decimals=3, mistakes={"la pénalité α w² s'ajoute au dénominateur : w = Σxy / (Σx² + α)": [float(XR_12 @ YR_12 / (XR_12 @ XR_12)), float((XR_12 @ YR_12 - 7) / (XR_12 @ XR_12))],
                     "sans ordonnée à l'origine, w = Σxy / Σx², pas le rapport des moyennes": [float(YR_12.mean() / XR_12.mean()), float(XR_12 @ YR_12 / (XR_12 @ XR_12 + 7))]}'''),
        ("e", "z-score, 3 decimals", "float((TEST_12 - FLIPPERS_12.mean()) / FLIPPERS_12.std())",
         r'''decimals=3, mistakes={"le scaler apprend sur l'entraînement SEUL : le manchot de test n'entre pas dans la moyenne": float((TEST_12 - WITH_TEST_12.mean()) / WITH_TEST_12.std()),
                     "StandardScaler divise par n, pas par n − 1": float((TEST_12 - FLIPPERS_12.mean()) / FLIPPERS_12.std(ddof=1))}'''),
    ]),
]

CODE_PAPER = [
    Paper("CP4.7", "Les lettres des quatre lignes fautives (a)", [
        ("a", 'four letters, for example "XYZW"', '"DIKM"', "mistakes={" + _bug_letters_7() + "}"),
    ]),
    Paper("CP4.8", "Lire les quatre paires de courbes", [
        ("a", "a letter A to F", '"A"', _letters_mistakes("A", {
            "B": "un learning rate trop petit fait baisser la loss lentement et régulièrement : celle-ci saute et ne baisse pas",
            "C": "l'overfitting suppose une loss d'entraînement qui baisse : regarde la courbe d'entraînement",
            "D": "un entraînement sain fait baisser la loss : regarde l'échelle",
            "E": "une fuite donne des losses basses et collées, pas des sauts",
            "F": "eval() oublié laisse la loss d'entraînement baisser normalement : est-ce le cas ici ?"})),
        ("b", "a letter A to F", '"C"', _letters_mistakes("C", {
            "A": "la loss d'entraînement baisse régulièrement jusqu'à presque 0 : le pas n'est pas trop grand",
            "B": "la loss d'entraînement baisse vite : le pas n'est pas trop petit",
            "D": "regarde la loss de validation après son minimum, et l'écart avec l'entraînement",
            "E": "une fuite ferait suivre l'entraînement à la validation : ici elles s'écartent",
            "F": "eval() oublié rend la validation bruitée et proche de l'entraînement : ici elle s'en écarte nettement"})),
        ("c", "a letter A to F", '"B"', _letters_mistakes("B", {
            "A": "un pas trop grand fait sauter la loss : ici elle baisse à peine, sans un seul saut",
            "C": "l'overfitting suppose une loss d'entraînement basse : où en est-elle après 30 epochs ?",
            "D": "un entraînement sain aurait fait baisser la loss bien plus vite : relis l'échelle",
            "E": "une fuite donne des losses BASSES : celles-ci sont encore près de ln 10",
            "F": "eval() oublié n'empêche pas l'entraînement d'avancer : ici rien n'avance"})),
        ("d", "a letter A to F", '"D"', _letters_mistakes("D", {
            "A": "la loss baisse régulièrement, sans saut : le pas n'est pas trop grand",
            "B": "la loss baisse vite les premières epochs : le pas n'est pas trop petit",
            "C": "la loss de validation remonte-t-elle à la fin ?",
            "E": "une fuite donnerait des losses presque nulles et identiques : ce n'est pas le cas",
            "F": "eval() oublié rendrait la validation bruitée : elle est ici régulière"})),
    ]),
    Paper("CP4.13", "Prédire ce que font cinq extraits PyTorch", [
        ("a", "a letter A, B, C or D", '"B"', _letters_mistakes("B", {
            "A": "le graphe est reconstruit à chaque tour : que fait backward() de la valeur déjà présente dans .grad ?",
            "C": "chaque tour recalcule (2 * x).sum() : un graphe neuf, qui peut être rétropropagé",
            "D": "x a requires_grad=True : backward() remplit x.grad"})),
        ("b", "a letter A, B, C or D", '"C"', _letters_mistakes("C", {
            "A": "en mode évaluation, le dropout ne tire plus de masque",
            "B": "PyTorch fait un dropout INVERSÉ : il ne multiplie rien par 1 − p en évaluation",
            "D": "en mode évaluation, le dropout ne met plus rien à zéro"})),
        ("c", "a letter A, B, C or D", '"C"', _letters_mistakes("C", {
            "A": "une couche neuve est en mode entraînement : avec un seul exemple, que vaut la variance du mini-batch ?",
            "B": "une couche neuve est en mode entraînement : elle normalise avec les statistiques du mini-batch",
            "D": "PyTorch vérifie la taille du mini-batch avant de calculer quoi que ce soit"})),
        ("d", "a letter A, B, C or D", '"A"', _letters_mistakes("A", {
            "B": "PyTorch range le poids de nn.Linear en (n_out, n_in), la transposée de la convention du workbook",
            "C": "le poids est une matrice, pas un vecteur aplati",
            "D": "c'est la forme du biais : on demande celle du poids"})),
        ("e", "a letter A, B, C or D", '"D"', _letters_mistakes("D", {
            "A": "des logits nuls donnent une softmax uniforme : la probabilité de la bonne classe vaut 1/4",
            "B": "CrossEntropyLoss rend −ln de la probabilité de la bonne classe, en moyenne : pas la probabilité elle-même",
            "C": "CrossEntropyLoss attend des logits : des zéros sont des logits parfaitement valables"})),
    ]),
]

WRAP = {"CP4.1b": "as_shape", "CP4.7a": "as_sorted_letters"}
WRAP.update({f"CP4.8{letter}": "as_letters" for letter in "abcd"})
WRAP.update({f"CP4.13{letter}": "as_letters" for letter in "abcde"})

SUMMARY = r'''# Summary of the automatic checks
import re


def natural(sub_id):
    """CP4.2a before CP4.13a: the question number as a number, then the letter."""
    number, letter = re.match(r"CP4\.(\d+)(.*)", sub_id).groups()
    return int(number), letter


if exam_over():
    done = {k: r for k, r in RESULTS.items() if r.status != "pending"}
    right = sorted((k for k, r in done.items() if r), key=natural)
    wrong = sorted((k for k, r in done.items() if not r), key=natural)
    pending = sorted((k for k, r in RESULTS.items() if r.status == "pending"), key=natural)
    print(f"Réponses vérifiées : {len(right)} juste(s) sur {len(done)} ; pas remplies : {len(pending)}.")
    if wrong:
        print("À revoir avec le corrigé :", ", ".join(wrong))
    checked = [ok for props in PROPERTIES.values() for _, ok in props]
    print(f"Propriétés du code (CP4.9) : {sum(checked)} sur {len(checked)}.")
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
        cells.append(code("\n".join(lines), tags=["answer"]))
    return cells


def header_cells(kind: str) -> list:
    rel = EXAM if kind == "exercise" else SOLUTIONS
    if kind == "exercise":
        title = "# Checkpoint IV · Examen blanc — notebook de l'examen"
        how = ("Ce notebook accompagne le sujet `01_examen_sujet.md`.\n\n"
               "- **Partie A, pendant l'examen** : la question de code CP4.9 (🔨, 2 points, avec NumPy seulement). "
               "Écris ton code dans la cellule à compléter, sans `mylearn`, sans PyTorch, sans les corrigés et sans "
               "assistant IA ; `help()` et la documentation officielle de NumPy sont permis. Rien n'est vérifié pendant "
               "l'examen : la cellule « essai » montre seulement ce que fait ton code. Tes réponses chiffrées et tes "
               "explications vont sur ta feuille (`04_mes_reponses.md`).\n"
               "- **Partie B, après l'examen** : la vérification automatique de ton code et des réponses de ta "
               "feuille. Ses cellules ne vérifient rien tant que `EXAM_OVER` vaut `False`.\n\n"
               "« Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche ⏳ "
               "(partie A) ou ⏸️ (partie B).\n\n"
               "> Travaille dans **ta copie** (`mon_travail/checkpoints/partie_4/02_examen_notebook.ipynb`, créée "
               "par `python tools/start_chapter.py CP4`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# Checkpoint IV · Examen blanc — solutions (notebook exécuté)"
        how = ("La solution de la question de code (CP4.9), exécutée et comparée à PyTorch, puis l'enregistrement des "
               "réponses que vérifie la partie B du notebook de l'examen (cellules marquées `answer`, récoltées "
               "par `tools/build_answers.py`). Le corrigé détaillé et le barème sont dans `03_examen_corrige.md`.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell("demo"))
    cells.append(md("## Partie A · Pendant l'examen : la question de code" if kind == "exercise"
                    else "## Partie A · La question de code"))
    cells.append(code(GIVEN))
    cells.append(md(CP49_MD))
    if kind == "exercise":
        cells += [code(CP49_TODO), code(CP49_TRY)]
        cells.append(md("---\n\n### 🛑 Fin de la partie A\n\nL'examen s'arrête ici. Rends ta copie (enregistre ce "
                        "notebook et ta feuille), puis, seulement après, passe à la partie B."))
        cells.append(md(PART_B_INTRO))
        cells.append(code(PART_B_FLAG))
        cells.append(md("### Le code de la partie A"))
        cells.append(code(CODE_CHECK_9))
        cells.append(md("### Les réponses de ta feuille"))
        for paper in PAPER[:6] + CODE_PAPER[:2] + PAPER[6:] + CODE_PAPER[2:]:
            cells += paper_cell(kind, paper)
        cells.append(md("### Bilan"))
        cells.append(code(SUMMARY))
        cells.append(md("**Ce que cette vérification ne voit pas** : les démarches et les justifications ; les "
                        "explications et les démonstrations (CP4.2 i, CP4.3 h, CP4.4 c (le pourquoi) et h, CP4.5 a, b "
                        "et d, CP4.7 b, CP4.10, CP4.11, CP4.12 f, CP4.14) ; et, pour CP4.9, les nombres recopiés sur "
                        "ta feuille (seules tes fonctions sont vérifiées). Tout cela se corrige avec "
                        "`03_examen_corrige.md`. Ensuite : la remédiation du corrigé (relis la synthèse, "
                        "`05_synthese.md`, pour les notions ratées), puis le mini-projet."))
    else:
        cells += [code(CP49_SOLUTION), code(CP49_RECORD, tags=["answer"])]
        cells.append(md("## Partie B · Les réponses vérifiées après l'examen\n\n"
                        "Chaque réponse est calculée à partir des nombres du sujet ; les erreurs classiques reçoivent "
                        "un message qui dit où chercher."))
        cells.append(code(PAPER_CONTEXT))
        for paper in PAPER[:6] + CODE_PAPER[:2] + PAPER[6:] + CODE_PAPER[2:]:
            cells += paper_cell(kind, paper)
    return cells


def main() -> int:
    write_notebook(EXAM, build("exercise"))
    write_notebook(SOLUTIONS, build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER + CODE_PAPER)
    print(f"✅ {EXAM} and {SOLUTIONS} written ({len(PAPER) + len(CODE_PAPER)} paper questions, {n_checks} paper "
          "checks, 1 code question)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
