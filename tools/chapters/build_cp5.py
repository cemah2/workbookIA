#!/usr/bin/env python
"""Build the two notebooks of the mock exam of part V from a single source (used by Claude).

    python tools/chapters/figures_cp5.py
    python tools/chapters/build_cp5.py
    python tools/run_all_notebooks.py checkpoints/partie_5/03_examen_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py checkpoints/partie_5/02_examen_notebook.ipynb

The exam notebook has two parts. Part A is done DURING the exam: the two code questions, CP5.5 (🔨, NumPy only:
unroll a simple RNN over a sequence from a given cell) and CP5.6 (🔨, PyTorch: a training loop with early
stopping that keeps the best weights), without mylearn and without any check. Part B is done AFTER the exam: it
checks the code of part A against oracles (nn.RNN, and the contract of CP5.6 written once more), then the answers
written on paper (sub-IDs CP5.1a, CP5.2a, ...). Its cells only run once ``EXAM_OVER = True``; its tools come from
``wb.nbtools``. The solutions notebook records every checked answer with ``wb.record``.

CP5.6 trains in float64: the best epoch and its validation loss then do not depend on the machine (a perturbation
of 1e-12 of the starting weights leaves the ten first digits of every validation loss unchanged, session 52),
whereas in float32 the trajectory changes from the third epoch on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import Paper, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "checkpoints/partie_5"
EXAM = f"{FOLDER}/02_examen_notebook.ipynb"
SOLUTIONS = f"{FOLDER}/03_examen_solutions.ipynb"

# ---------------------------------------------------------------------------
# Data and tools of the code questions
# ---------------------------------------------------------------------------
GIVEN = r'''# Data and tools of the code questions CP5.5 and CP5.6 (run this cell first)
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# CP5.5: a mini-batch of 3 sequences of 5 steps with 2 features, and the parameters of a simple RNN of 4 neurons
rng = np.random.default_rng(55)
X_S = rng.normal(size=(3, 5, 2))                       # (N, T, D): batch, time, features
H0_S = np.zeros((3, 4))                                # (N, H): the starting state
WX_S = rng.normal(0.0, 0.8, size=(2, 4))               # (D, H)
WH_S = rng.normal(0.0, 0.8, size=(4, 4))               # (H, H)
B_S = rng.normal(0.0, 0.3, size=4)                     # (H,)


def rnn_cell(x_t, h, Wx, Wh, b):
    """One step of a simple RNN for a mini-batch: x_t (N, D), h (N, H) -> tanh(x_t @ Wx + h @ Wh + b), (N, H)."""
    return np.tanh(x_t @ Wx + h @ Wh + b)


# CP5.6: 3 072 images of Fashion-MNIST in [0, 1], float64 (2 048 to train, 1 024 to validate)
_X6, _Y6 = wb.datasets.load_fashion_mnist("train", n=3072, normalize=True, seed=56)
X_IMG = torch.from_numpy(_X6.astype(np.float64)).unsqueeze(1)      # (3072, 1, 28, 28)
Y_IMG = torch.from_numpy(_Y6.astype(np.int64))                     # (3072,)


def make_cnn(seed=56):
    """A new small CNN with dropout, in float64 (the same starting weights for the same seed)."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),       # (N, 8, 14, 14)
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # (N, 16, 7, 7)
        nn.Flatten(), nn.Dropout(0.25), nn.Linear(16 * 7 * 7, 10)).double()


def make_loaders(seed=56, n_train=2048, n_val=1024):
    """(train_loader, val_loader): mini-batches of 64 shuffled images, then of 300 images (the last one is smaller)."""
    train = TensorDataset(X_IMG[:n_train], Y_IMG[:n_train])
    val = TensorDataset(X_IMG[2048:2048 + n_val], Y_IMG[2048:2048 + n_val])
    return (DataLoader(train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed)),
            DataLoader(val, batch_size=300))


CHECKPOINT = Path(tempfile.gettempdir()) / "cp5_6_best.pt"     # where the try of CP5.6 saves the best weights
_ORIGINALS = {"X_S": X_S.copy(), "H0_S": H0_S.copy()}          # part B restores them if a try changed them
print("X_S:", X_S.shape, "· images:", tuple(X_IMG.shape), "· labels:", tuple(Y_IMG.shape),
      "· CNN parameters:", sum(p.numel() for p in make_cnn().parameters()))'''

CP55_MD = r'''### CP5.5 — Écrire `rnn_forward` à partir d'une cellule fournie 🔨 ★ ⏱️ 10 min · 1,5 point
**Objectif :** dérouler une couche récurrente simple sur toute une séquence, pour un mini-batch.

Avec NumPy seulement. `rnn_cell(x_t, h, Wx, Wh, b)` (fournie) fait un pas : $h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$, avec `x_t` de forme `(N, D)` et `h` de forme `(N, H)`.

a) `rnn_forward(X, h0, Wx, Wh, b)` reçoit `X` de forme `(N, T, D)` (batch, temps, features : la convention `batch_first=True`) et l'état de départ `h0`, de forme `(N, H)`. Elle rend le couple `(H_all, h_last)` : `H_all`, de forme `(N, T, H)`, où `H_all[:, t]` est l'état **après** le pas `t` (`h0` n'y figure pas), et `h_last`, l'état après le dernier pas, de forme `(N, H)`. Elle ne modifie pas `X`. **[1,2]**
b) Avec les données de la cellule (`X_S`, `H0_S`, `WX_S`, `WH_S`, `B_S`), reporte sur ta feuille `H_all[1, 4, 0]` (4 décimales). **[0,3]**'''

CP55_TODO = r'''def rnn_forward(X, h0, Wx, Wh, b):
    """(H_all (N, T, H), h_last (N, H)): the states after each step of rnn_cell over X (N, T, D); CP5.5 a)."""
    raise NotImplementedError("rnn_forward")  # TODO CP5.5 a)'''

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


def try_5():
    H_all, h_last = rnn_forward(X_S, H0_S, WX_S, WH_S, B_S)
    print("H_all:", np.shape(H_all), "· h_last:", np.shape(h_last))
    print("H_all[1, 4, 0] =", round(float(H_all[1, 4, 0]), 4))


try_code("CP5.5", try_5)'''

CP55_SOLUTION = r'''def rnn_forward(X, h0, Wx, Wh, b):
    """(H_all (N, T, H), h_last (N, H)): the states after each step of rnn_cell over X (N, T, D); CP5.5 a)."""
    h, states = h0, []
    for t in range(X.shape[1]):                        # the time axis is axis 1 (batch_first)
        h = rnn_cell(X[:, t], h, Wx, Wh, b)            # the same weights at every step
        states.append(h)
    return np.stack(states, axis=1), h                 # (N, T, H): h0 is not one of the states


H_all_5, h_last_5 = rnn_forward(X_S, H0_S, WX_S, WH_S, B_S)
print("H_all:", H_all_5.shape, "· h_last:", h_last_5.shape, "· H_all[1, 4, 0] =", round(H_all_5[1, 4, 0], 4))
rnn_5 = nn.RNN(2, 4, batch_first=True).double()        # the same layer in PyTorch: W_ih = Wx.T, W_hh = Wh.T
with torch.no_grad():
    rnn_5.weight_ih_l0.copy_(torch.from_numpy(WX_S.T))
    rnn_5.weight_hh_l0.copy_(torch.from_numpy(WH_S.T))
    rnn_5.bias_ih_l0.copy_(torch.from_numpy(B_S))
    rnn_5.bias_hh_l0.zero_()                           # PyTorch has two biases: their sum plays the role of b
    out_5, h_n_5 = rnn_5(torch.from_numpy(X_S), torch.from_numpy(H0_S).unsqueeze(0))
print(f"nn.RNN: largest gap {np.abs(out_5.numpy() - H_all_5).max():.1e} ; h_n = the last output: "
      f"{torch.equal(h_n_5[0], out_5[:, -1])}")'''

CP55_RECORD = r'''def variant_5(*, recurrent=True, transpose_wh=False, with_h0=False, bias=True):
    """H_all[1, 4, 0] of CP5.5 written another (wrong) way."""
    h, states = H0_S, [H0_S] if with_h0 else []
    for t in range(X_S.shape[1]):
        z = X_S[:, t] @ WX_S + (B_S if bias else 0.0)
        if recurrent:
            z = z + h @ (WH_S.T if transpose_wh else WH_S)
        h = np.tanh(z)
        states.append(h)
    return float(np.stack(states, axis=1)[1, 4, 0])


wb.record("CP5.5b", H_all_5[1, 4, 0], decimals=4, mistakes={
    "l'état précédent n'entre pas dans le calcul : chaque pas reçoit h, l'état rendu par le pas d'avant": variant_5(recurrent=False),
    "H_all contient h0 en première case : la case t est l'état APRÈS le pas t, sans h0": variant_5(with_h0=True),
    "Wh est appliquée transposée : la cellule calcule h @ Wh, avec Wh de forme (H, H)": variant_5(transpose_wh=True),
    "le biais b manque": variant_5(bias=False)})'''

CP56_MD = r'''### CP5.6 — Entraîner avec early stopping et garder le meilleur modèle 🔨 ★★ ⏱️ 15 min · 2 points
**Objectif :** écrire de mémoire une boucle d'entraînement PyTorch complète : modes entraînement et évaluation, loss de validation par exemple, sauvegarde des meilleurs poids, arrêt après `patience` epochs sans progrès.

Avec PyTorch. `make_cnn()` rend un petit CNN neuf (avec dropout, en `float64`) et `make_loaders()` le couple `(train_loader, val_loader)` sur 2 048 et 1 024 images de Fashion-MNIST ; les mini-batches de validation ont 300 images, sauf le dernier (124).

a) `fit_early_stopping(model, train_loader, val_loader, max_epochs, patience, path, lr=1e-3)` : **[1,6]**
- crée **un** optimiseur, `torch.optim.Adam(model.parameters(), lr=lr)` ;
- à chaque epoch (numérotées à partir de 1, au plus `max_epochs`) : en mode entraînement, un passage sur `train_loader` (cross-entropy moyenne du mini-batch sur les logits, `zero_grad`, `backward`, `step`) ; la loss d'entraînement de l'epoch est la moyenne **par exemple** des losses des mini-batches (chacune pesée par son nombre d'exemples) ; puis, en mode évaluation et sans gradient, la loss de validation : la cross-entropy moyenne **par exemple** sur tout `val_loader` ;
- si la loss de validation est **strictement** plus petite que la meilleure vue jusque-là, sauvegarde `model.state_dict()` dans `path` (`torch.save`) et retient l'epoch ; sinon, compte une epoch sans progrès ; l'entraînement s'arrête dès que `patience` epochs **de suite** n'ont pas progressé ;
- à la fin, recharge dans `model` les poids sauvegardés (`torch.load(path, weights_only=True)`) et rend le dictionnaire `{"train_loss": [...], "val_loss": [...], "best_epoch": k}` : une valeur (un `float`) par epoch faite, et l'epoch des poids gardés, numérotée à partir de 1.

b) La cellule d'essai entraîne `make_cnn()` sur `make_loaders()` avec `max_epochs=30`, `patience=3` et `lr=3e-3` (une dizaine de secondes) : reporte sur ta feuille $[\text{best\_epoch}, \text{loss de validation à cette epoch}]$ (4 décimales). **[0,4]**'''

CP56_TODO = r'''def fit_early_stopping(model, train_loader, val_loader, max_epochs, patience, path, lr=1e-3):
    """Adam, early stopping on the validation loss (mean per example), best state_dict saved in path and reloaded;
    returns {"train_loss": [...], "val_loss": [...], "best_epoch": k}; CP5.6 a)."""
    raise NotImplementedError("fit_early_stopping")  # TODO CP5.6 a)'''

CP56_TRY = r'''# Try fit_early_stopping (about ten seconds; no check during the exam)
def try_6():
    model_try = make_cnn()
    history_try = fit_early_stopping(model_try, *make_loaders(), 30, 3, CHECKPOINT, lr=3e-3)
    print("epochs done:", len(history_try["val_loss"]), "· best_epoch:", history_try["best_epoch"])
    print("validation losses:", [round(float(v), 4) for v in history_try["val_loss"]])


try_code("CP5.6", try_6)'''

CP56_SOLUTION = r'''def fit_early_stopping(model, train_loader, val_loader, max_epochs, patience, path, lr=1e-3):
    """Adam, early stopping on the validation loss (mean per example), best state_dict saved in path and reloaded;
    returns {"train_loss": [...], "val_loss": [...], "best_epoch": k}; CP5.6 a)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)          # one optimizer: its moments live all along
    history = {"train_loss": [], "val_loss": [], "best_epoch": None}
    best, waited = float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()                                                # dropout on
        total, count = 0.0, 0
        for X, y in train_loader:
            loss = nn.functional.cross_entropy(model(X), y)          # logits: the softmax is inside
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        history["train_loss"].append(total / count)
        model.eval()                                                 # dropout off
        with torch.no_grad():
            total, count = 0.0, 0
            for X, y in val_loader:
                total += nn.functional.cross_entropy(model(X), y, reduction="sum").item()
                count += len(y)
        val = total / count                                          # per example: the last mini-batch has 124
        history["val_loss"].append(val)
        if val < best:                                               # strictly better: save
            best, waited, history["best_epoch"] = val, 0, epoch
            torch.save(model.state_dict(), path)
        else:
            waited += 1
            if waited == patience:
                break
    model.load_state_dict(torch.load(path, weights_only=True))       # back to the best weights
    return history


history_6 = fit_early_stopping(make_cnn(), *make_loaders(), 30, 3, CHECKPOINT, lr=3e-3)
best_6 = history_6["best_epoch"]
print("epochs done:", len(history_6["val_loss"]), "· best_epoch:", best_6,
      "· its validation loss:", round(history_6["val_loss"][best_6 - 1], 4))
print("validation losses:", [round(v, 4) for v in history_6["val_loss"]])'''

CP56_RECORD = r'''def variant_6(*, per_batch=False, train_mode=False, extra_wait=0, lr=3e-3, zero_based=False, last=False):
    """[best_epoch, its validation loss] of CP5.6 b) written another (wrong) way."""
    model = make_cnn()
    train_loader, val_loader = make_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_epoch, waited, val_losses = float("inf"), None, 0, []
    for epoch in range(1, 31):
        model.train()
        for X, y in train_loader:
            loss = nn.functional.cross_entropy(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if not train_mode:
            model.eval()
        with torch.no_grad():
            losses = [(nn.functional.cross_entropy(model(X), y).item(), len(y)) for X, y in val_loader]
        val = float(np.mean([l for l, _ in losses])) if per_batch else sum(l * n for l, n in losses) / 1024
        val_losses.append(val)
        if val < best:
            best, best_epoch, waited = val, epoch, 0
        else:
            waited += 1
            if waited == 3 + extra_wait:
                break
    return [best_epoch - 1 if zero_based else best_epoch, val_losses[-1] if last else best]


wb.record("CP5.6b", [best_6, history_6["val_loss"][best_6 - 1]], decimals=4, mistakes={
    "la loss de validation est la moyenne des moyennes des mini-batches : le dernier, de 124 images, pèse alors "
    "autant qu'un mini-batch de 300 ; la moyenne est PAR EXEMPLE": variant_6(per_batch=True),
    "la validation tourne en mode entraînement : le dropout éteint des neurones pendant la mesure (model.eval() "
    "avant la validation)": variant_6(train_mode=True),
    "l'arrêt vient une epoch trop tard : il a lieu dès que patience epochs DE SUITE n'ont pas progressé":
    variant_6(extra_wait=1),
    "le learning rate reçu (lr=3e-3) n'est pas celui de l'optimiseur": variant_6(lr=1e-3),
    "les epochs se numérotent à partir de 1": variant_6(zero_based=True),
    "c'est la loss de validation de la dernière epoch faite : on demande celle de l'epoch des poids gardés":
    variant_6(last=True)})'''

# ---------------------------------------------------------------------------
# Part B: checks after the exam
# ---------------------------------------------------------------------------
PART_B_INTRO = r'''## Partie B · Après l'examen : vérification automatique

**Ne commence cette partie qu'une fois ta copie terminée**, examen rendu. Elle vérifie le code de la partie A, puis les réponses que tu as écrites sur ta feuille, question par question : ✅ juste, ❌ faux (avec une piste), ⏳ pas rempli. Elle ne note pas : les démarches, les justifications et les questions rédigées se corrigent avec `03_examen_corrige.md`, qui donne aussi le barème.

1. Passe `EXAM_OVER` à `True` dans la cellule ci-dessous, puis exécute-la.
2. Exécute les vérifications du code (CP5.5 et CP5.6 ; la seconde entraîne le CNN plusieurs fois, une trentaine de secondes).
3. Reporte ensuite chaque réponse de ta feuille : **la valeur** que tu as écrite, pas un nouveau calcul (sinon tu ne vérifies plus ta copie). En Python, le séparateur décimal est un **point** (`0.125`), une matrice s'écrit comme une liste de lignes (`[[1, 2], [3, 4]]`), une forme comme une chaîne (`"(8, 16, 28, 28)"`) et les lettres entre guillemets (`"XYZ"`).

Si tu as fermé le notebook depuis l'examen, exécute d'abord la cellule de configuration (tout en haut) et les cellules de la partie A, sauf les essais.'''

PART_B_FLAG = r'''EXAM_OVER = False   # set it to True once your exam is finished, then run the cells of part B

import re

from wb.nbtools import StopCheck, as_numpy, letters_text, pair, shape_text, verdict

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


def as_outcome(value):
    """CP5.1: a shape typed as "(8, 16)", "[8, 16]", "8 x 16" or a tuple -> "(8, 16)"; « erreur » for an error."""
    if isinstance(value, str) and re.search(r"(?i)err|exception|runtime|plante", value):
        return "erreur"
    return shape_text(value)


def sorted_letters(value):
    """A set of letters typed in any order ("D, B, I et F") -> its letters in alphabetical order ("BDFI")."""
    return letters_text(value, sort=True)


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

CODE_CHECK_5 = r'''# Checks of rnn_forward (CP5.5), against nn.RNN
if exam_over() and ready("CP5.5", "np", "torch", "nn", "X_S", "H0_S", "WX_S", "WH_S", "B_S", "rnn_cell", "_ORIGINALS",
                         "rnn_forward"):
    PROPERTIES["CP5.5"] = []
    PENDING.discard("CP5.5")
    RESULTS.pop("CP5.5b", None)
    restore_data("CP5.5", ["X_S", "H0_S"])

    def oracle_5(X, h0, Wx, Wh, b):
        """(output, h_n) of nn.RNN(batch_first=True) in float64 with W_ih = Wx.T, W_hh = Wh.T, b_ih = b, b_hh = 0."""
        layer = nn.RNN(Wx.shape[0], Wx.shape[1], batch_first=True).double()
        with torch.no_grad():
            layer.weight_ih_l0.copy_(torch.from_numpy(Wx.T.copy()))
            layer.weight_hh_l0.copy_(torch.from_numpy(Wh.T.copy()))
            layer.bias_ih_l0.copy_(torch.from_numpy(b.copy()))
            layer.bias_hh_l0.zero_()
            out, h_n = layer(torch.from_numpy(X.copy()), torch.from_numpy(h0.copy()).unsqueeze(0))
        return out.numpy(), h_n[0].numpy()

    X_in = X_S.copy()
    ok, result = safe(rnn_forward, X_in, H0_S.copy(), WX_S, WH_S, B_S)
    if ok is None:
        print("⏳ Ex CP5.5 : rnn_forward n'est pas encore écrite.")
        PENDING.add("CP5.5")
    elif not ok:
        prop("CP5.5", False, "", f"rnn_forward lève une erreur : {result}")
    else:
        try:
            H_all, h_last = pair("CP5.5", "rnn_forward", result, names=("H_all", "h_last"))
            want_all, want_last = oracle_5(X_S, H0_S, WX_S, WH_S, B_S)
            got_all, got_last = as_numpy(H_all), as_numpy(h_last)
            prop("CP5.5", got_all is not None and got_all.shape == (3, 5, 4) and got_last is not None
                 and got_last.shape == (3, 4),
                 "H_all a la forme (N, T, H) = (3, 5, 4) et h_last la forme (N, H) = (3, 4).",
                 f"attendu H_all de forme (3, 5, 4) et h_last de forme (3, 4), reçu {np.shape(H_all)}"
                 f"{' (une liste)' if isinstance(H_all, list) else ''} et "
                 f"{np.shape(h_last)} : H_all empile les T états après chaque pas, sur l'axe 1, sans h0.")
            if got_all is not None and got_all.shape == (3, 5, 4):
                gap = float(np.abs(got_all - want_all).max())
                prop("CP5.5", gap < 1e-10, "les états de chaque pas sont ceux de nn.RNN (mêmes poids, batch_first=True).",
                     f"l'écart à nn.RNN vaut {gap:.2e} : chaque pas reçoit X[:, t] (l'axe du temps est l'axe 1) et "
                     "l'état rendu par le pas précédent, en partant de h0.")
            if got_last is not None and got_last.shape == (3, 4):
                prop("CP5.5", np.abs(got_last - want_last).max() < 1e-10,
                     "h_last est l'état après le dernier pas (le h_n de nn.RNN).",
                     "h_last doit être l'état après le DERNIER pas, H_all[:, -1].")
            prop("CP5.5", np.array_equal(X_in, X_S), "rnn_forward ne modifie pas X.",
                 "rnn_forward modifie le X qu'elle reçoit : elle doit le laisser intact.")
            # other sizes, and a starting state that is not zero: are the arguments used?
            rng_o = np.random.default_rng(551)
            X_o, h0_o = rng_o.normal(size=(2, 3, 5)), rng_o.normal(size=(2, 6))
            Wx_o, Wh_o, b_o = rng_o.normal(0, 0.5, (5, 6)), rng_o.normal(0, 0.5, (6, 6)), rng_o.normal(0, 0.5, 6)
            ok_o, out_o = safe(rnn_forward, X_o, h0_o, Wx_o, Wh_o, b_o)
            want_o = oracle_5(X_o, h0_o, Wx_o, Wh_o, b_o)[0]
            try:
                same_o = bool(ok_o and np.allclose(as_numpy(out_o[0]), want_o, rtol=1e-10, atol=1e-12))
            except (TypeError, ValueError, IndexError):
                same_o = False
            prop("CP5.5", same_o, "avec d'autres tailles (N = 2, T = 3, D = 5, H = 6) et un h0 non nul, les états de nn.RNN.",
                 "avec N = 2, T = 3, D = 5, H = 6 et un h0 non nul, rnn_forward ne rend pas les états de nn.RNN : "
                 "utilise-t-elle h0 et les tailles de ses arguments (pas 5 pas ni 4 neurones écrits en dur) ?")
            ok_1, out_1 = safe(rnn_forward, X_o[:, :1], h0_o, Wx_o, Wh_o, b_o)
            try:
                one_ok = bool(ok_1 and as_numpy(out_1[0]).shape == (2, 1, 6)
                              and np.allclose(as_numpy(out_1[1]), want_o[:, 0], rtol=1e-10, atol=1e-12))
            except (TypeError, ValueError, IndexError):
                one_ok = False
            prop("CP5.5", one_ok, "une séquence d'un seul pas donne H_all de forme (N, 1, H).",
                 "avec T = 1, attendu H_all de forme (2, 1, 6) et h_last = rnn_cell(X[:, 0], h0, ...).")
            if got_all is not None and got_all.shape == (3, 5, 4):
                code_check("CP5.5b", got_all[1, 4, 0], "H_all[1, 4, 0] : séquence 1, pas 4 (le dernier), neurone 0.")
        except StopCheck:                                   # pair() printed why
            PROPERTIES["CP5.5"].append(("rnn_forward rend le couple (H_all, h_last)", False))'''

CODE_CHECK_6 = r'''# Checks of fit_early_stopping (CP5.6), against the contract written once more (about thirty seconds)
if exam_over() and ready("CP5.6", "torch", "nn", "make_cnn", "make_loaders", "fit_early_stopping"):
    PROPERTIES["CP5.6"] = []
    PENDING.discard("CP5.6")
    RESULTS.pop("CP5.6b", None)

    def reference_fit(model, train_loader, val_loader, max_epochs, patience, path, lr=1e-3):
        """The contract of CP5.6 a), the oracle of these checks."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history, best, waited = {"train_loss": [], "val_loss": [], "best_epoch": None}, float("inf"), 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            total, count = 0.0, 0
            for X, y in train_loader:
                loss = nn.functional.cross_entropy(model(X), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total, count = total + loss.item() * len(y), count + len(y)
            history["train_loss"].append(total / count)
            model.eval()
            with torch.no_grad():
                total = sum(nn.functional.cross_entropy(model(X), y, reduction="sum").item() for X, y in val_loader)
            history["val_loss"].append(total / len(val_loader.dataset))
            if history["val_loss"][-1] < best:
                best, waited, history["best_epoch"] = history["val_loss"][-1], 0, epoch
                torch.save(model.state_dict(), path)
            else:
                waited += 1
                if waited == patience:
                    break
        model.load_state_dict(torch.load(path, weights_only=True))
        return history

    def run(function, max_epochs, patience, lr, **sizes):
        """(ok, history or error, final weights, saved weights) of one training from a new CNN and new loaders."""
        path = Path(tempfile.mkdtemp()) / "best.pt"
        model = make_cnn()
        ok, history = safe(function, model, *make_loaders(**sizes), max_epochs, patience, path, lr=lr)
        try:
            saved = torch.load(path, weights_only=True) if ok and path.exists() else None
        except Exception:                                   # not a state_dict (a whole model, a dict of other things)
            saved = None
        return ok, history, {k: v.detach().clone() for k, v in model.state_dict().items()}, saved

    def same_weights(a, b):
        return (a is not None and b is not None and set(a) == set(b)
                and all(torch.allclose(a[k].double(), b[k].double(), rtol=1e-7, atol=1e-10) for k in b))

    def losses(history, key):
        """The list history[key] as floats, or None (missing key, not a list of numbers)."""
        try:
            values = as_numpy([float(v) for v in history[key]])
        except (TypeError, ValueError, KeyError):
            return None
        return values

    ok, mine, mine_final, mine_saved = run(fit_early_stopping, 30, 3, 3e-3)
    if ok is None:
        print("⏳ Ex CP5.6 : fit_early_stopping n'est pas encore écrite.")
        PENDING.add("CP5.6")
    elif not ok:
        prop("CP5.6", False, "", f"fit_early_stopping lève une erreur : {mine}")
    elif mine is None:
        prop("CP5.6", False, "", "fit_early_stopping renvoie None : as-tu oublié le return ?")
    elif not isinstance(mine, dict) or not {"train_loss", "val_loss", "best_epoch"} <= set(mine):
        prop("CP5.6", False, "", "fit_early_stopping doit rendre un dictionnaire aux clés \"train_loss\", \"val_loss\" et "
             f"\"best_epoch\" ; reçu {type(mine).__name__}" + (f" de clés {sorted(mine)}" if isinstance(mine, dict) else "") + ".")
    else:
        _, ref, ref_final, ref_saved = run(reference_fit, 30, 3, 3e-3)
        mine_tr, mine_va = losses(mine, "train_loss"), losses(mine, "val_loss")
        if mine_tr is None or mine_va is None:
            prop("CP5.6", False, "", "history[\"train_loss\"] et history[\"val_loss\"] doivent être des listes de nombres "
                 "(un float par epoch : loss.item()).")
        else:
            kinds = sorted({type(v).__name__ for key in ("train_loss", "val_loss") for v in mine[key]
                            if not isinstance(v, float)})
            prop("CP5.6", not kinds, "les losses rendues sont des float Python.",
                 f"history contient des {', '.join(kinds)} : range chaque loss comme un float Python (loss.item()), "
                 "pas comme un tenseur.")
            ref_tr, ref_va = as_numpy(ref["train_loss"]), as_numpy(ref["val_loss"])
            n = min(len(mine_tr), len(ref_tr), len(mine_va))
            bad_tr = next((k for k in range(n) if not np.isclose(mine_tr[k], ref_tr[k], rtol=1e-6, atol=0)), None)
            bad_va = next((k for k in range(n) if not np.isclose(mine_va[k], ref_va[k], rtol=1e-6, atol=0)), None)
            if len(mine_tr) != len(mine_va):
                prop("CP5.6", False, "", f"{len(mine_tr)} losses d'entraînement pour {len(mine_va)} de validation : une "
                     "valeur de chaque par epoch faite.")
            elif bad_tr is not None and (bad_va is None or bad_tr <= bad_va):
                prop("CP5.6", False, "", f"la loss d'entraînement de l'epoch {bad_tr + 1} vaut {mine_tr[bad_tr]:.6f}, "
                     f"attendu {ref_tr[bad_tr]:.6f} : un seul optimiseur Adam(lr=lr) pour tout l'entraînement, "
                     "model.train(), la cross-entropy moyenne sur les logits, et la moyenne PAR EXEMPLE des mini-batches "
                     "de l'epoch.")
            elif bad_va is not None:
                prop("CP5.6", False, "", f"la loss de validation de l'epoch {bad_va + 1} vaut {mine_va[bad_va]:.6f}, "
                     f"attendu {ref_va[bad_va]:.6f} : model.eval() avant la validation (sinon le dropout éteint des "
                     "neurones), torch.no_grad(), tout val_loader, et la moyenne PAR EXEMPLE (le dernier mini-batch a "
                     "124 images, pas 300).")
            else:
                prop("CP5.6", True, "à chaque epoch, les losses d'entraînement et de validation sont celles du contrat "
                     "(mode entraînement puis évaluation, moyennes par exemple).", "")
                prop("CP5.6", len(mine_va) == len(ref_va),
                     f"l'entraînement s'arrête après {len(ref_va)} epochs : 3 epochs de suite sans progrès.",
                     f"{len(mine_va)} epochs faites, attendu {len(ref_va)} : l'arrêt a lieu dès que patience = 3 epochs "
                     "DE SUITE n'ont pas fait strictement mieux que la meilleure loss de validation.")
                prop("CP5.6", mine["best_epoch"] == ref["best_epoch"],
                     f"best_epoch = {ref['best_epoch']}, l'epoch de la plus petite loss de validation, comptée à partir de 1.",
                     f"best_epoch = {mine['best_epoch']!r}, attendu {ref['best_epoch']} : l'epoch (numérotée à partir de 1) "
                     "de la plus petite loss de validation.")
                prop("CP5.6", same_weights(mine_saved, ref_saved), "le fichier path contient le state_dict de la meilleure epoch.",
                     "le fichier path ne contient pas les poids de la meilleure epoch : torch.save(model.state_dict(), path) "
                     "à chaque progrès strict, et seulement alors.")
                prop("CP5.6", same_weights(mine_final, ref_final), "à la fin, le modèle a repris les poids de la meilleure epoch.",
                     "à la fin, le modèle n'a pas les poids de la meilleure epoch : model.load_state_dict(torch.load(path, "
                     "weights_only=True)) avant le return.")
            if bad_tr is not None or bad_va is not None or len(mine_tr) != len(mine_va):
                print("   (l'arrêt, best_epoch et les poids gardés ne se vérifient qu'une fois les losses de chaque epoch justes)")
        # a learning rate of 0: the validation loss never changes, so it never improves STRICTLY after epoch 1
        ok_0, flat, _, _ = run(fit_early_stopping, 6, 2, 0.0, n_train=256, n_val=128)
        try:
            n_flat, best_flat = len(flat["val_loss"]), flat["best_epoch"]
            moved = len({round(float(v), 12) for v in flat["val_loss"]}) > 1
        except (TypeError, KeyError, ValueError):
            n_flat, best_flat, moved = None, None, False
        if best_flat == 0:
            why = "les epochs se numérotent à partir de 1."
        elif moved:
            why = "la loss de validation a changé : l'optimiseur utilise-t-il le lr reçu ?"
        elif isinstance(best_flat, (int, np.integer)) and best_flat > 1:
            why = "seul un progrès STRICT compte : une loss égale à la meilleure n'en est pas un."
        else:
            why = "l'arrêt vient dès que patience (le patience reçu) epochs de suite n'ont pas progressé."
        prop("CP5.6", ok_0 and n_flat == 3 and best_flat == 1,
             "avec lr = 0 (une loss de validation qui ne bouge plus), best_epoch = 1 et l'arrêt après 1 + patience epochs.",
             "avec lr = 0, patience = 2 et max_epochs = 6, la loss de validation est la même à chaque epoch : attendu "
             "best_epoch = 1 et 3 epochs faites" + (f", reçu {n_flat} epochs et best_epoch = {best_flat!r} : {why}"
                                                    if n_flat is not None else " : " + why))
        # max_epochs and lr come from the arguments
        ok_2, short, _, _ = run(fit_early_stopping, 2, 5, 1e-3, n_train=256, n_val=128)
        _, short_ref, _, _ = run(reference_fit, 2, 5, 1e-3, n_train=256, n_val=128)
        try:
            short_ok = bool(ok_2 and len(short["val_loss"]) == 2
                            and np.allclose([float(v) for v in short["val_loss"]], short_ref["val_loss"], rtol=1e-8, atol=0))
        except (TypeError, KeyError, ValueError):
            short_ok = False
        prop("CP5.6", short_ok, "avec max_epochs = 2 et lr = 1e-3 (256 images), les deux epochs du contrat.",
             "avec max_epochs = 2, patience = 5 et lr = 1e-3 sur 256 images, attendu les deux epochs du contrat : "
             "fit_early_stopping utilise-t-elle ses arguments (pas 30 epochs ni lr = 3e-3 écrits en dur) ?")
        if mine_tr is not None and mine_va is not None and isinstance(mine.get("best_epoch"), (int, np.integer)) \
                and 1 <= mine["best_epoch"] <= len(mine_va):
            code_check("CP5.6b", [int(mine["best_epoch"]), float(mine_va[mine["best_epoch"] - 1])],
                       "best_epoch puis la loss de validation de cette epoch.")'''

PAPER_CONTEXT = r'''# The paper questions only use the numbers of their statements (01_examen_sujet.md)
import math

import numpy as np

# CP5.2


def conv_out(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1


def receptive_field(layers, pool_adds=True):
    """Receptive field of the last layer of [(kernel, stride, is_pool)]: r += (k - 1) * jump, then jump *= stride."""
    r, jump = 1, 1
    for k, s, is_pool in layers:
        if pool_adds or not is_pool:
            r += (k - 1) * jump
        jump *= s
    return r


LAYERS_2 = [(3, 1, False), (2, 2, True), (3, 1, False), (2, 2, True), (3, 2, False)]
PARAMS_2 = [3 * 9 * 16 + 16, 16 * 9 * 32 + 32, 32 * 9 * 64 + 64, 64 * 10 + 10]
SIDE_C_2 = conv_out(8, 3, s=2, p=1)
MACS_2 = 64 * SIDE_C_2 ** 2 * 32 * 9

# CP5.3


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


X_3, C0_3 = 2.0, 0.5
PRE_3 = [0.5 * X_3 - 1, 1.0 * X_3 + 0, 0.25 * X_3 + 0, -0.5 * X_3 + 1]      # i, f, g, o (h0 = 0)
GATES_3 = [sigmoid(PRE_3[0]), sigmoid(PRE_3[1]), math.tanh(PRE_3[2]), sigmoid(PRE_3[3])]
I_3, F_3, G_3, O_3 = GATES_3
C1_3 = F_3 * C0_3 + I_3 * G_3
H1_3 = O_3 * math.tanh(C1_3)

# CP5.8
EMB_8 = 10_000 * 256
LSTM_8 = 4 * (512 * 256 + 512 * 512 + 2 * 512)
OUT_8 = 512 * 10_000 + 10_000
P_8 = EMB_8 + LSTM_8 + OUT_8
FLOP_8 = 6 * P_8 * 1e6

# CP5.10
X_10 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 9.0])
TRAIN_10 = X_10[2:]                                   # KFold(3) without shuffle: the first fold tests (1, 2)
Z_10 = float((2.0 - TRAIN_10.mean()) / TRAIN_10.std())
XD_10 = np.array([[1.0, 2.0], [0.0, -1.0]])
W_10 = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
DZ_10 = np.array([[0.5, -1.0, 0.0], [1.0, 0.0, 2.0]])

# CP5.12
X_12, W_12 = np.array([1.0, 2.0, -1.0, 3.0]), np.array([1.0, -1.0])
Y_12 = np.array([W_12 @ X_12[t:t + 2] for t in range(3)])          # cross-correlation, as nn.Conv1d
DW_12 = [float(Y_12 @ X_12[:3]), float(Y_12 @ X_12[1:]), float(Y_12.sum())]
DX_12 = np.zeros(4)
for t in range(3):
    DX_12[t:t + 2] += Y_12[t] * W_12'''


def _letters_mistakes(answer: str, messages: dict) -> str:
    """mistakes={message: letter} for every other choice of a lettered question."""
    pairs = ",\n                     ".join(f'"{message}": "{letter}"' for letter, message in messages.items()
                                            if letter != answer)
    return "mistakes={" + pairs + "}"


def _bug_letters_4() -> str:
    """mistakes for the sets of letters of CP5.4 a) that differ from BDFI by one letter (missing, extra, swapped)."""
    answer, lines = set("BDFI"), []
    lines.append('"il manque une ligne fautive : quelle forme Conv2d(1, …) attend-elle en entrée ?": "DFI"')
    lines.append('"il manque une ligne fautive : refais la taille de la carte après le MaxPool2d(2)": "BFI"')
    lines.append('"il manque une ligne fautive : dans quel ordre nn.LSTM lit-il les axes, par défaut ?": "BDI"')
    lines.append('"il manque une ligne fautive : quel pas de temps résume toute l\'image ?": "BDF"')
    others = "ACEGHJK"
    for extra in others:
        lines.append(f'"une ligne de trop : la ligne ({extra}) est correcte ; relis ce qu\'elle fait": '
                     f'"{"".join(sorted(answer | {extra}))}"')
    for removed in sorted(answer):
        for extra in others:
            letters = "".join(sorted((answer - {removed}) | {extra}))
            if (removed, extra) == ("F", "H"):     # the same bug, blamed on the call instead of the constructor
                lines.append('"l\'appel (H) passe bien un tenseur (batch, temps, features) : c\'est la couche qui doit '
                             'savoir lire cet ordre, et une ligne fautive manque": "BDHI"')
                continue
            if (removed, extra) == ("B", "E") or (removed, extra) == ("D", "E"):
                lines.append(f'"l\'erreur se déclare bien en (E), mais l\'énoncé demande la ligne à corriger : (E) est '
                             f'correcte, et une ligne fautive manque": "{letters}"')
                continue
            lines.append(f'"trois des quatre lignes sont justes : la ligne ({extra}) est correcte, et une ligne fautive '
                         f'manque": "{letters}"')
    return ",\n                     ".join(lines)


PAPER = [
    Paper("CP5.1", "Forme de sortie ou erreur ? Six extraits PyTorch", [
        ("a", 'a shape like "(1, 2)", or "erreur"', '"(8, 16, 28, 28)"',
         r'''mistakes={"sans padding, un noyau de taille k tient dans n − k + 1 positions, pas n − k": "(8, 16, 27, 27)",
                     "sans padding, la carte rétrécit : padding vaut 0 par défaut": "(8, 16, 32, 32)",
                     "les canaux de sortie sont les 16 filtres": "(8, 3, 28, 28)"}'''),
        ("b", 'a shape like "(1, 2)", or "erreur"', '"erreur"',
         r'''mistakes={"PyTorch lit les axes dans l'ordre (N, C, H, W) : quel axe prend-il ici pour les canaux ?": "(8, 16, 32, 32)",
                     "Conv2d(3, …) attend 3 canaux sur l'axe 1 ; combien en trouve-t-il ?": "(8, 16, 32, 3)"}'''),
        ("c", 'a shape like "(1, 2)", or "erreur"', '"(1, 4, 3, 3)"',
         r'''mistakes={"MaxPool2d arrondit vers le bas : la dernière ligne et la dernière colonne, sans voisine, sont ignorées": "(1, 4, 4, 4)",
                     "le stride de MaxPool2d(2) vaut sa taille, 2 : il divise la taille par 2": "(1, 4, 6, 6)",
                     "un pooling garde les canaux : il agit sur chaque carte séparément": "(1, 2, 3, 3)",
                     "MaxPool2d accepte une taille impaire : il arrondit": "erreur"}'''),
        ("d", 'a shape like "(1, 2)", or "erreur"', '"(2, 7, 20)"',
         r'''mistakes={"sans batch_first, nn.LSTM lit (temps, batch, features) : le batch, ici, c'est l'axe de taille 7": "(2, 4, 20)",
                     "[1][0] est h_n, l'état final de chaque couche (num_layers, batch, hidden), pas la sortie à chaque pas": "(4, 7, 20)",
                     "h_n a une ligne par couche : num_layers = 2": "(1, 7, 20)",
                     "h_n garde l'axe des couches en premier : (num_layers, batch, hidden)": "(7, 2, 20)"}'''),
        ("e", 'a shape like "(1, 2)", or "erreur"', '"(4, 7, 40)"',
         r'''mistakes={"bidirectionnel : la sortie de chaque pas concatène les deux sens, 2 × 20 features": "(4, 7, 20)",
                     "[0] est la sortie de chaque pas, pas h_n": "(2, 4, 20)",
                     "avec batch_first=True, la sortie garde l'ordre (batch, temps, features)": "(7, 4, 40)"}'''),
        ("f", 'a shape like "(1, 2)", or "erreur"', '"(2, 3, 5)"',
         r'''mistakes={"nn.Linear agit sur le dernier axe et garde tous les autres": "(6, 5)",
                     "nn.Linear accepte un nombre quelconque d'axes devant le dernier": "erreur",
                     "nn.Linear(16, 5) transforme les 16 features du dernier axe en 5": "(2, 3, 16)"}'''),
    ]),
    Paper("CP5.2", "Formes, paramètres et champ récepteur d'un petit CNN", [
        ("a", 'a shape like "(1, 2, 3, 4)"', '"(1, 16, 16, 16)"',
         r'''mistakes={"le bloc A se termine par un MaxPool2d(2)": "(1, 16, 32, 32)",
                     "padding=1 garde la taille avec un noyau 3 × 3 : la convolution sort 32 × 32": "(1, 16, 15, 15)"}'''),
        ("b", 'a shape like "(1, 2, 3, 4)"', '"(1, 32, 8, 8)"',
         r'''mistakes={"le bloc B se termine aussi par un MaxPool2d(2)": "(1, 32, 16, 16)",
                     "la sortie du bloc B a 32 canaux, un par filtre": "(1, 16, 8, 8)"}'''),
        ("c", 'a shape like "(1, 2, 3, 4)"', '"(1, 64, 4, 4)"',
         r'''mistakes={"cette convolution a un stride de 2 : elle saute une position sur deux": "(1, 64, 8, 8)",
                     "le padding de 1 s'ajoute des deux côtés de la carte": "(1, 64, 3, 3)",
                     "on demande la sortie de la convolution C, avant AdaptiveAvgPool2d(1)": "(1, 64, 1, 1)"}'''),
        ("d", "a list of five numbers", "[*PARAMS_2, sum(PARAMS_2)]",
         r'''mistakes={"compte aussi les biais, un par filtre ou par neurone": [*(p - b for p, b in zip(PARAMS_2, (16, 32, 64, 10))), sum(PARAMS_2) - 122],
                     "un filtre couvre TOUS les canaux d'entrée : k × k × C_in poids": [16 * 9 + 16, 32 * 9 + 32, 64 * 9 + 64, 650, 16 * 10 + 32 * 10 + 64 * 10 + 650],
                     "la couche dense reçoit les 64 valeurs d'AdaptiveAvgPool2d(1), pas 64 × 4 × 4": [*PARAMS_2[:3], 64 * 16 * 10 + 10, sum(PARAMS_2[:3]) + 64 * 16 * 10 + 10]}'''),
        ("e", "a number of multiply-adds", "MACS_2",
         r'''mistakes={"le stride de 2 réduit le nombre de positions de sortie : relis c)": 64 * 8 * 8 * 32 * 9,
                     "chaque position combine les 32 canaux d'entrée, chacun avec 3 × 3 poids": 64 * SIDE_C_2 ** 2 * 9,
                     "on compte une multiplication-addition par poids ET par position de sortie": 32 * 9 * 64}'''),
        ("f", "a size in pixels", "receptive_field(LAYERS_2)",
         r'''mistakes={"les poolings comptent : chacun agrandit le champ et multiplie l'écart entre positions voisines": receptive_field([(3, 1, False)] * 2 + [(3, 2, False)]),
                     "un pooling 2 × 2 agrandit aussi le champ, de (2 − 1) × l'écart courant": receptive_field(LAYERS_2, pool_adds=False),
                     "chaque couche ajoute (k − 1) × l'écart entre positions voisines, qui grandit après chaque stride": 3 + 1 + 2 + 1 + 2}'''),
    ]),
    Paper("CP5.3", "Un pas de LSTM à la main", [
        ("a", "[i, f, g, o], 4 decimals", "GATES_3",
         r'''decimals=4, mistakes={"la proposition g passe par tanh, pas par la sigmoïde": [I_3, F_3, sigmoid(PRE_3[2]), O_3],
                     "les portes i, f et o passent par la sigmoïde, pas par tanh": [math.tanh(PRE_3[0]), math.tanh(PRE_3[1]), G_3, math.tanh(PRE_3[3])],
                     "ce sont les sommes pondérées : applique ensuite la sigmoïde ou tanh": PRE_3}'''),
        ("b", "c1, 4 decimals", "C1_3",
         r'''decimals=4, mistakes={"la cellule garde f ⊙ c₀ de son ancien contenu": I_3 * G_3,
                     "la proposition g entre à travers la porte d'entrée : i ⊙ g": F_3 * C0_3 + G_3,
                     "c'est la porte d'oubli qui multiplie l'ancienne cellule, et la porte d'entrée la proposition": I_3 * C0_3 + F_3 * G_3}'''),
        ("c", "h1, 4 decimals", "H1_3",
         r'''decimals=4, mistakes={"h = o ⊙ tanh(c) : la cellule passe par tanh": O_3 * C1_3,
                     "la porte de sortie o filtre tanh(c)": math.tanh(C1_3)}'''),
        ("d", "number of parameters", "4 * (20 * 10 + 20 * 20 + 20 + 20)",
         r'''mistakes={"PyTorch a deux vecteurs de biais par porte, b_ih et b_hh": 4 * (20 * 10 + 20 * 20 + 20),
                     "un LSTM a quatre blocs de poids (i, f, g, o), pas un seul": 20 * 10 + 20 * 20 + 20 + 20,
                     "c'est le compte d'un GRU (trois blocs) : le LSTM en a quatre": 3 * (20 * 10 + 20 * 20 + 20 + 20),
                     "les poids récurrents vont de l'état (20) aux portes : 20 × 20 par bloc": 4 * (20 * 10 + 20 * 10 + 20 + 20)}'''),
    ]),
    Paper("CP5.4", "Les lettres des quatre lignes fautives (a)", [
        ("a", 'four letters, for example "XYZW"', '"BDFI"', "mistakes={" + _bug_letters_4() + "}"),
    ]),
    Paper("CP5.7", "Diagnostiquer quatre paires de courbes", [
        ("a", "a letter A to F", '"C"', _letters_mistakes("C", {
            "A": "un learning rate trop grand fait sauter ou stagner la loss : ici les deux courbes baissent régulièrement",
            "B": "l'overfitting met la validation AU-DESSUS de l'entraînement, et la fait remonter : c'est l'inverse ici",
            "D": "avec des labels au hasard, la loss de validation resterait vers ln 10 ≈ 2,3",
            "E": "un learning rate trop petit ferait baisser la loss très lentement : elle baisse vite les premières epochs",
            "F": "eval() oublié laisserait le dropout actif pendant la validation : sa loss monterait au niveau de "
                 "l'entraînement, pas en dessous"})),
        ("b", "a letter A to F", '"A"', _letters_mistakes("A", {
            "B": "l'overfitting suppose une loss d'entraînement qui baisse : celle-ci ne descend jamais sous ln 10 ≈ 2,3",
            "C": "ce MLP n'a pas de dropout",
            "D": "des labels au hasard laisseraient la loss d'entraînement baisser lentement, à force de mémoriser : ici "
                 "elle dépasse 11 à la première epoch, puis reste plate",
            "E": "un pas trop petit ne fait pas monter la loss au-dessus de 11 à la première epoch",
            "F": "ce MLP n'a ni dropout ni batchnorm : eval() ne change rien à sa validation"})),
        ("c", "a letter A to F", '"B"', _letters_mistakes("B", {
            "A": "la loss d'entraînement baisse régulièrement jusque vers 0,1 : le pas n'est pas trop grand",
            "C": "ce CNN n'a pas de dropout ; regarde où va la loss de validation après son minimum",
            "D": "des labels au hasard empêcheraient la validation de descendre sous ln 10 ≈ 2,3",
            "E": "la loss d'entraînement baisse vite : le pas n'est pas trop petit",
            "F": "eval() oublié ne fait pas remonter la validation pendant que l'entraînement descend vers 0,1 : cet "
                 "écart qui se creuse a une autre cause"})),
        ("d", "a letter A to F", '"E"', _letters_mistakes("E", {
            "A": "un pas trop grand fait sauter la loss : ici elle baisse, sans un saut",
            "B": "l'overfitting suppose une loss d'entraînement basse : où en est-elle après 30 epochs ?",
            "C": "ce MLP n'a pas de dropout",
            "D": "des labels au hasard ne feraient pas baisser la loss de validation : ici elle baisse avec celle "
                 "d'entraînement",
            "F": "ce MLP n'a ni dropout ni batchnorm : eval() ne change rien à sa validation"})),
    ]),
    Paper("CP5.8", "Fermi : un générateur de texte LSTM", [
        ("a", "parameters in millions, 1 decimal", "P_8 / 1e6",
         r'''decimals=1, mistakes={"un LSTM a quatre blocs de poids (i, f, g, o)": (EMB_8 + LSTM_8 / 4 + OUT_8) / 1e6,
                     "l'embedding est une matrice de paramètres, 10 000 × 256": (LSTM_8 + OUT_8) / 1e6,
                     "la couche de sortie projette 512 → 10 000, avec ses biais": (EMB_8 + LSTM_8) / 1e6}'''),
        ("b", "a percentage, rounded to the unit", "round(100 * OUT_8 / P_8)",
         r'''fractional="arrondis au pourcent près", mistakes={"c'est la part de l'embedding : on demande celle de la couche de sortie": round(100 * EMB_8 / P_8),
                     "c'est la part du LSTM : on demande celle de la couche de sortie": round(100 * LSTM_8 / P_8)}'''),
        ("c", "FLOP in units of 10**13, 1 decimal", "FLOP_8 / 1e13",
         r'''decimals=1, mistakes={"la règle compte 6 FLOP par paramètre et par token": P_8 * 1e6 / 1e13,
                     "2PN est le coût de la passe avant seule ; l'entraînement compte aussi la passe arrière, 6PN": 2 * P_8 * 1e6 / 1e13}'''),
        ("d", "seconds, rounded to the unit", "round(FLOP_8 / 1e13)",
         r'''fractional="arrondis à la seconde", mistakes={"attention aux puissances de 10 : la machine fait 10¹³ FLOP par seconde": round(FLOP_8 / 1e12)}'''),
    ]),
    Paper("CP5.10", "Retour sur les parties III et IV", [
        ("a", "[mean, std] of the scaler, 4 decimals", "[TRAIN_10.mean(), TRAIN_10.std()]",
         r'''decimals=4, mistakes={"dans un Pipeline, le scaler s'ajuste sur les exemples d'entraînement du fold seulement": [X_10.mean(), X_10.std()],
                     "StandardScaler divise par n, pas par n − 1": [TRAIN_10.mean(), TRAIN_10.std(ddof=1)],
                     "KFold sans shuffle découpe dans l'ordre : le premier fold TESTE les deux premières valeurs": [X_10[[0, 1, 4, 5]].mean(), X_10[[0, 1, 4, 5]].std()]}'''),
        ("b", "z-score, 3 decimals", "Z_10",
         r'''decimals=3, mistakes={"dans un Pipeline, le scaler s'ajuste sur les exemples d'entraînement du fold seulement": float((2.0 - X_10.mean()) / X_10.std()),
                     "StandardScaler divise par n, pas par n − 1": float((2.0 - TRAIN_10.mean()) / TRAIN_10.std(ddof=1))}'''),
        ("c", "number of fits of the scaler", "4 * 3 + 1",
         r'''mistakes={"refit=True (le défaut) réentraîne ensuite le meilleur pipeline sur tout le dataset, scaler compris": 12,
                     "le Pipeline réajuste le scaler dans chaque fold, pour chaque valeur de C": 1,
                     "chaque valeur de C est évaluée sur 3 folds, et chaque fold ajuste son propre scaler": 4,
                     "il y a 4 valeurs de C à évaluer, pas une seule": 3}'''),
        ("d", "dW as a list of two rows", "(XD_10.T @ DZ_10).tolist()",
         r'''decimals=1, mistakes={"∂L/∂W = Xᵀ ∂L/∂Z : X est transposée, et elle vient à gauche": (XD_10 @ DZ_10).tolist()}'''),
        ("e", "db as a list", "DZ_10.sum(axis=0).tolist()",
         r'''decimals=1, mistakes={"∂L/∂b SOMME les deltas sur les exemples : la moyenne est déjà dans ∂L/∂Z": DZ_10.mean(axis=0).tolist(),
                     "on somme sur les exemples (les lignes), pour garder un gradient par neurone": DZ_10.sum(axis=1).tolist()}'''),
        ("f", "dX as a list of two rows", "(DZ_10 @ W_10.T).tolist()",
         r'''decimals=1, mistakes={"∂L/∂X a la forme de X, (exemples, features) : une ligne par exemple": (DZ_10 @ W_10.T).T.tolist()}'''),
    ]),
    Paper("CP5.11", "Questions flash : paramètres et tailles", [
        ("a", "number of parameters", "8 * 4 * 5 + 8",
         r'''mistakes={"un filtre 1D couvre les 4 canaux d'entrée : 4 × 5 poids": 8 * 5 + 8,
                     "compte aussi les biais, un par filtre": 8 * 4 * 5}'''),
        ("b", "number of parameters", "3 * (20 * 10 + 20 * 20 + 20 + 20)",
         r'''mistakes={"PyTorch a deux vecteurs de biais par bloc, b_ih et b_hh": 3 * (20 * 10 + 20 * 20 + 20),
                     "c'est le compte d'un LSTM (quatre blocs) : le GRU en a trois": 4 * (20 * 10 + 20 * 20 + 20 + 20)}'''),
        ("c", "the output height", "(16 - 1) * 2 - 2 * 1 + 4",
         r'''mistakes={"c'est la formule d'une convolution ordinaire : la convolution transposée agrandit": conv_out(16, 4, s=2, p=1),
                     "le padding retire 2 × 1 lignes à la sortie d'une convolution transposée": (16 - 1) * 2 + 4}'''),
        ("d", "number of parameters", "5000 * 64",
         r'''mistakes={"un embedding n'a pas de biais : une ligne de 64 valeurs par token": 5000 * 64 + 64}'''),
    ]),
    Paper("CP5.12", "Rétropropager dans une convolution 1D", [
        ("a", "[y1, y2, y3]", "Y_12.tolist()",
         r'''mistakes={"nn.Conv1d (et le ch. 21) calcule une corrélation croisée : le noyau n'est pas retourné": [-v for v in Y_12.tolist()]}'''),
        ("c", "[dL/dw1, dL/dw2, dL/db]", "DW_12",
         r'''decimals=2, mistakes={"la loss de l'énoncé est une SOMME sur les trois positions, pas une moyenne": [v / 3 for v in DW_12],
                     "w₁ multiplie x_t et w₂ multiplie x_{t+1} : chaque poids voit sa propre fenêtre d'entrées": [DW_12[1], DW_12[0], DW_12[2]]}'''),
        ("d", "[dL/dx1, ..., dL/dx4]", "DX_12.tolist()",
         r'''mistakes={"x₂ et x₃ servent à deux positions : leurs gradients additionnent deux contributions": [Y_12[0] * W_12[0], Y_12[1] * W_12[0], Y_12[2] * W_12[0], Y_12[2] * W_12[1]]}'''),
    ]),
]

WRAP = {f"CP5.1{letter}": "as_outcome" for letter in "abcdef"}
WRAP.update({f"CP5.2{letter}": "shape_text" for letter in "abc"})
WRAP["CP5.4a"] = "sorted_letters"
WRAP.update({f"CP5.7{letter}": "letters_text" for letter in "abcd"})

SUMMARY = r'''# Summary of the automatic checks
import re


def natural(sub_id):
    """CP5.2a before CP5.10a: the question number as a number, then the letter."""
    number, letter = re.match(r"CP5\.(\d+)(.*)", sub_id).groups()
    return int(number), letter


if exam_over():
    done = {k: r for k, r in RESULTS.items() if r.status != "pending"}
    right = sorted((k for k, r in done.items() if r), key=natural)
    wrong = sorted((k for k, r in done.items() if not r), key=natural)
    pending = sorted((k for k, r in RESULTS.items() if r.status == "pending"), key=natural)
    print(f"Réponses vérifiées : {len(right)} juste(s) sur {len(done)} ; pas remplies : {len(pending)}.")
    if wrong:
        print("À revoir avec le corrigé :", ", ".join(wrong))
    for question in ("CP5.5", "CP5.6"):
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
        cells.append(code("\n".join(lines), tags=["answer"]))
    return cells


def header_cells(kind: str) -> list:
    rel = EXAM if kind == "exercise" else SOLUTIONS
    if kind == "exercise":
        title = "# Checkpoint V · Examen blanc — notebook de l'examen"
        how = ("Ce notebook accompagne le sujet `01_examen_sujet.md`.\n\n"
               "- **Partie A, pendant l'examen** : les deux questions de code, CP5.5 (🔨, 1,5 point, avec NumPy "
               "seulement) et CP5.6 (🔨, 2 points, avec PyTorch). Écris ton code dans les cellules à compléter, sans "
               "`mylearn`, sans les corrigés et sans assistant IA ; `help()` et les documentations officielles de NumPy "
               "et de PyTorch sont permis. Rien n'est vérifié pendant l'examen : les cellules « essai » montrent "
               "seulement ce que fait ton code. Tes réponses chiffrées et tes explications vont sur ta feuille "
               "(`04_mes_reponses.md`).\n"
               "- **Partie B, après l'examen** : la vérification automatique de ton code et des réponses de ta "
               "feuille. Ses cellules ne vérifient rien tant que `EXAM_OVER` vaut `False`.\n\n"
               "« Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche ⏳ "
               "(partie A) ou ⏸️ (partie B).\n\n"
               "> Travaille dans **ta copie** (`mon_travail/checkpoints/partie_5/02_examen_notebook.ipynb`, créée "
               "par `python tools/start_chapter.py CP5`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# Checkpoint V · Examen blanc — solutions (notebook exécuté)"
        how = ("Les solutions des deux questions de code (CP5.5, comparée à `nn.RNN` ; CP5.6, l'entraînement avec "
               "early stopping), exécutées, puis l'enregistrement des réponses que vérifie la partie B du notebook de "
               "l'examen (cellules marquées `answer`, récoltées par `tools/build_answers.py`). Le corrigé détaillé et le "
               "barème sont dans `03_examen_corrige.md`.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell("demo"))
    cells.append(md("## Partie A · Pendant l'examen : les questions de code" if kind == "exercise"
                    else "## Partie A · Les questions de code"))
    cells.append(code(GIVEN))
    cells.append(md(CP55_MD))
    if kind == "exercise":
        cells += [code(CP55_TODO), code(TRY_TOOL)]
        cells.append(md(CP56_MD))
        cells += [code(CP56_TODO), code(CP56_TRY)]
        cells.append(md("---\n\n### 🛑 Fin de la partie A\n\nL'examen s'arrête ici. Rends ta copie (enregistre ce "
                        "notebook et ta feuille), puis, seulement après, passe à la partie B."))
        cells.append(md(PART_B_INTRO))
        cells.append(code(PART_B_FLAG))
        cells.append(md("### Le code de la partie A"))
        cells += [code(CODE_CHECK_5), code(CODE_CHECK_6)]
        cells.append(md("### Les réponses de ta feuille"))
        for paper in PAPER:
            cells += paper_cell(kind, paper)
        cells.append(md("### Bilan"))
        cells.append(code(SUMMARY))
        cells.append(md("**Ce que cette vérification ne voit pas** : les démarches et les justifications ; les "
                        "explications (CP5.1 g, CP5.2 g, CP5.3 e, CP5.4 b, CP5.9, CP5.12 b, CP5.13, CP5.14) ; et, pour "
                        "CP5.5 et CP5.6, les nombres recopiés sur ta feuille (seules tes fonctions sont vérifiées). Tout "
                        "cela se corrige avec `03_examen_corrige.md`. Ensuite : la remédiation du corrigé (relis la "
                        "synthèse, `05_synthese.md`, pour les notions ratées), puis le mini-projet."))
    else:
        cells += [code(CP55_SOLUTION), code(CP55_RECORD, tags=["answer"])]
        cells.append(md(CP56_MD))
        cells += [code(CP56_SOLUTION), code(CP56_RECORD, tags=["answer"])]
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
