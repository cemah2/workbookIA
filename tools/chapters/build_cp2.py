#!/usr/bin/env python
"""Build the two notebooks of the mock exam of part II from a single source (used by Claude).

    python tools/chapters/figures_cp2.py
    python tools/chapters/build_cp2.py
    python tools/run_all_notebooks.py checkpoints/partie_2/03_examen_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py checkpoints/partie_2/02_examen_notebook.ipynb

The exam notebook has two parts. Part A is done DURING the exam: the two code
questions CP2.10 (🐛) and CP2.11 (🔨), without mylearn and without any check. Part B
is done AFTER the exam: it checks the code of part A, then the answers written on
paper (sub-IDs CP2.1a, CP2.2a, ...). Its cells only run once ``EXAM_OVER = True``.
The solutions notebook records every checked answer with ``wb.record``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import Paper, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "checkpoints/partie_2"
EXAM = f"{FOLDER}/02_examen_notebook.ipynb"
SOLUTIONS = f"{FOLDER}/03_examen_solutions.ipynb"

# ---------------------------------------------------------------------------
# Data and tools shared by both notebooks
# ---------------------------------------------------------------------------
GIVEN = r'''# Data and tools of the two code questions (run this cell first)
import numpy as np

california = wb.datasets.load_california()                    # 20 640 districts (1990 census)
FEATURES = california.columns[:8].tolist()                    # MedInc, HouseAge, ..., Latitude, Longitude
rows = np.sort(np.random.default_rng(2610).choice(len(california), 2000, replace=False))
X = california[FEATURES].to_numpy(dtype=float)[rows]          # shape (2000, 8)
y = california["MedHouseVal"].to_numpy(dtype=float)[rows]     # median house value (hundreds of thousands of $)


def make_cells(size):
    """Id of the square cell (side `size`, in degrees) of the latitude-longitude grid that holds each district."""
    row = np.floor(np.round(X[:, 6] / size, 8)).astype(int)    # latitude band
    col = np.floor(np.round(X[:, 7] / size, 8)).astype(int)    # longitude band
    return row * 100_000 + col


CELLS = {size: make_cells(size) for size in (0.5, 0.1, 0.05)}  # the three grids of CP2.10 d)


def cell_means(cells, y):
    """Dict: cell id -> mean of y over the districts of that cell (only the districts passed)."""
    return {cell: float(y[cells == cell].mean()) for cell in np.unique(cells)}


def lookup(cells, means, default):
    """For each district, the mean of its cell in `means`, or `default` if its cell is not in `means`."""
    return np.array([means.get(cell, default) for cell in cells], dtype=float)


def kfold(n, k=5, seed=0):
    """List of k pairs (train_indices, val_indices): the n indices, shuffled once, cut into k folds."""
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]


def ridge_fit(F, y, alpha):
    """Ridge regression with an intercept that is not penalized: returns (w, b)."""
    F_mean, y_mean = F.mean(axis=0), y.mean()
    Fc = F - F_mean
    w = np.linalg.solve(Fc.T @ Fc + alpha * np.eye(F.shape[1]), Fc.T @ (y - y_mean))
    return w, y_mean - F_mean @ w


def ridge_predict(model, F):
    w, b = model
    return F @ w + b


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


print("X:", X.shape, "· y:", y.shape, "· cells of 0.1°:", len(np.unique(CELLS[0.1])))'''

# ---------------------------------------------------------------------------
# Part A: the two code questions of the exam
# ---------------------------------------------------------------------------
CP210_MD = r'''### CP2.10 — La fuite cachée d'une validation croisée 🐛 ★ ⏱️ 9 min · 1,5 point
**Objectif :** trouver ce qu'une validation croisée a appris de ses folds de validation, et la corriger.

Un collègue estime la RMSE d'une régression Ridge par validation croisée à 5 folds, sur les 2 000 districts de `X` et `y`. En plus des huit features, il ajoute « le prix du quartier » : la grille latitude-longitude est découpée en cases de 0,1° de côté (`CELLS[0.1]` donne la case de chaque district), et chaque district reçoit la moyenne des prix (`y`) des districts de sa case. Il standardise ensuite les neuf colonnes, puis lance la validation croisée. Exécute son code (cellule suivante).

a) Deux étapes de son code apprennent quelque chose des données avant la boucle de validation croisée. Lesquelles ? Laquelle fausse vraiment son score, et pourquoi l'autre ne le change presque pas ? (sur ta feuille) **[0,4 pt]**
b) Écris `cv_rmse_fixed(X, y, cells, k=5, alpha=1.0)`, qui fait le même calcul **sans fuite**. Dans chaque tour (`kfold(len(y), k)`, comme chez le collègue) : le prix du quartier de chaque district, d'entraînement comme de validation, est la moyenne des prix des districts **d'entraînement du tour** qui sont dans sa case ; un district dont la case ne contient aucun district d'entraînement reçoit la moyenne des prix d'entraînement du tour ; la standardisation utilise la moyenne et l'écart-type des lignes d'entraînement du tour. La fonction renvoie la moyenne des $k$ RMSE de validation, sans modifier ses arguments. **[0,6 pt]**
c) La RMSE moyenne de ta fonction avec les cases de 0,1° (3 décimales, sur ta feuille). **[0,2 pt]**
d) Le collègue a aussi comparé des cases de 0,5°, 0,1° et 0,05° avec son code (sortie de sa cellule), et garde les plus petites. Compare avec ta fonction : quelle taille garderais-tu ? Pourquoi la fuite le poussait-elle vers les plus petites cases ? (sur ta feuille) **[0,3 pt]**'''

CP210_BUG = r'''# The colleague's code (do not change this cell: run it and read what it prints)
def cv_rmse_colleague(X, y, cells, k=5, alpha=1.0):
    neighbourhood = lookup(cells, cell_means(cells, y), y.mean())   # "price of the neighbourhood" of each district
    F = np.column_stack([X, neighbourhood])
    F = (F - F.mean(axis=0)) / F.std(axis=0)                         # standardize the 9 columns
    scores = []
    for train, val in kfold(len(y), k):
        model = ridge_fit(F[train], y[train], alpha)
        scores.append(rmse(y[val], ridge_predict(model, F[val])))
    return float(np.mean(scores))


for size, cells in CELLS.items():
    print(f"cells of {size}°: the colleague's cross-validated RMSE = {cv_rmse_colleague(X, y, cells):.4f}")'''

CP210_TODO = r'''def cv_rmse_fixed(X, y, cells, k=5, alpha=1.0):
    """Mean validation RMSE of the colleague's model over k folds, without leakage (see CP2.10 b)."""
    raise NotImplementedError("cv_rmse_fixed")  # TODO CP2.10 b)'''

CP210_TRY = r'''# Try your function (no check during the exam: compare with what you expect)
with wb.attempt("CP2.10"):
    for size, cells in CELLS.items():
        print(f"cells of {size}°: cross-validated RMSE without leakage = {cv_rmse_fixed(X, y, cells):.4f}")'''

CP210_SOLUTION = r'''# a) Two steps learn from ALL the districts before the folds exist: the cell means of the target (each
#    district's own price is in the mean of its cell: the validation targets leak into the features), and the
#    standardization (a mean and a std over 2 000 districts barely move when 400 of them are removed).
def cv_rmse_fixed(X, y, cells, k=5, alpha=1.0):
    """Mean validation RMSE of the colleague's model over k folds, without leakage (see CP2.10 b)."""
    scores = []
    for train, val in kfold(len(y), k):
        means = cell_means(cells[train], y[train])                    # learnt on the training districts only
        neighbourhood = lookup(cells, means, y[train].mean())          # for every district, train and validation
        F = np.column_stack([X, neighbourhood])
        F = (F - F[train].mean(axis=0)) / F[train].std(axis=0)        # standardization fitted on the training rows
        model = ridge_fit(F[train], y[train], alpha)
        scores.append(rmse(y[val], ridge_predict(model, F[val])))
    return float(np.mean(scores))


fixed_10 = {size: cv_rmse_fixed(X, y, cells) for size, cells in CELLS.items()}
leaky_10 = {size: cv_rmse_colleague(X, y, cells) for size, cells in CELLS.items()}
for size in CELLS:
    print(f"cells of {size}°: colleague {leaky_10[size]:.4f} · without leakage {fixed_10[size]:.4f}")'''

CP210_RECORD = r'''# Classic wrong procedures of b), each recorded with its message
def variant_10(cells, *, means="train", default="train"):
    """The cross-validated RMSE when the neighbourhood feature is computed another (wrong) way."""
    scores = []
    for train, val in kfold(len(y), 5):
        fallback = y[train].mean() if default == "train" else y.mean()
        if means == "all":                                     # the colleague's means, over all the districts
            neighbourhood = lookup(cells, cell_means(cells, y), fallback)
        else:
            neighbourhood = lookup(cells, cell_means(cells[train], y[train]), fallback)
        if means == "own_fold":                                # validation districts encoded with their own fold
            neighbourhood[val] = lookup(cells[val], cell_means(cells[val], y[val]), y[val].mean())
        if means == "leave_one_out":                           # training districts encoded without their own price
            for i in train:
                others = train[(cells[train] == cells[i]) & (train != i)]
                neighbourhood[i] = y[others].mean() if others.size else y[train].mean()
        F = np.column_stack([X, neighbourhood])
        F = (F - F[train].mean(axis=0)) / F[train].std(axis=0)
        scores.append(rmse(y[val], ridge_predict(ridge_fit(F[train], y[train], 1.0), F[val])))
    return float(np.mean(scores))


def no_neighbourhood():
    """The cross-validated RMSE of the 8 features alone (the new feature forgotten)."""
    scores = []
    for train, val in kfold(len(y), 5):
        F = (X - X[train].mean(axis=0)) / X[train].std(axis=0)
        scores.append(rmse(y[val], ridge_predict(ridge_fit(F[train], y[train], 1.0), F[val])))
    return float(np.mean(scores))


cells_10 = CELLS[0.1]
wb.record("CP2.10c", fixed_10[0.1], decimals=4, mistakes={
    "c'est encore la fuite du collègue : les moyennes par case utilisent les prix de TOUS les districts, ceux du fold de validation compris":
    variant_10(cells_10, means="all"),
    "les districts de validation reçoivent la moyenne de leur propre fold : leur prix entre encore dans leur feature ; prends les moyennes des seuls districts d'entraînement du tour":
    variant_10(cells_10, means="own_fold"),
    "ta version retire aussi son propre prix à chaque district d'entraînement : c'est une bonne idée, mais l'énoncé demande, pour tous les districts, la moyenne des districts d'entraînement de la case":
    variant_10(cells_10, means="leave_one_out"),
    "ta fonction oublie le prix du quartier : le modèle doit garder les neuf colonnes du collègue":
    no_neighbourhood()})
wb.record("CP2.10d", [fixed_10[size] for size in CELLS], decimals=4, mistakes={
    "c'est encore la fuite du collègue : les moyennes par case utilisent les prix de TOUS les districts, ceux du fold de validation compris":
    [variant_10(cells, means="all") for cells in CELLS.values()],
    "les districts de validation reçoivent la moyenne de leur propre fold : leur prix entre encore dans leur feature ; prends les moyennes des seuls districts d'entraînement du tour":
    [variant_10(cells, means="own_fold") for cells in CELLS.values()],
    "pour une case sans district d'entraînement, le défaut est la moyenne des prix d'entraînement du tour, pas celle de tous les districts":
    [variant_10(cells, default="all") for cells in CELLS.values()]})'''

CP211_MD = r'''### CP2.11 — Coder `epsilon_greedy_action` et une moyenne incrémentale 🔨 ★ ⏱️ 9 min · 1 point
**Objectif :** programmer, sans `mylearn`, les deux briques d'un agent de bandit.

a) Écris `epsilon_greedy_action(q_values, epsilon, rng)`. Avec la probabilité $\varepsilon$ (un tirage `rng.random() < epsilon`), elle explore : un bras au hasard parmi **tous** les bras, `rng.integers(K)` ($K$ = nombre de bras). Sinon, elle exploite : un bras d'estimation maximale, tiré au hasard parmi les ex aequo avec `rng.choice`. Elle renvoie un `int` et ne modifie pas `q_values`. **[0,5 pt]**
b) Écris `incremental_estimates(rewards, step=None)`, qui renvoie la liste des estimations successives d'un bras, une après chaque récompense, en partant de $Q = 0$ : à la $n$-ième récompense $R$, $Q \leftarrow Q + a\,(R - Q)$, avec $a = 1/n$ si `step` vaut `None` (la moyenne exacte), et $a$ = `step` sinon (un pas constant). La fonction ne recalcule jamais la somme de toutes les récompenses. **[0,3 pt]**
c) Avec ta fonction, la dernière estimation pour les récompenses `[1, 0, 0, 1, 1, 1, 0, 1]` et un pas constant de 0,25 (4 décimales, sur ta feuille). **[0,2 pt]**'''

CP211_TODO = r'''def epsilon_greedy_action(q_values, epsilon, rng):
    """The arm chosen by the epsilon-greedy rule (an int), see CP2.11 a)."""
    raise NotImplementedError("epsilon_greedy_action")  # TODO CP2.11 a)


def incremental_estimates(rewards, step=None):
    """The list of the successive estimates Q, one after each reward, starting from Q = 0 (CP2.11 b)."""
    raise NotImplementedError("incremental_estimates")  # TODO CP2.11 b)'''

CP211_TRY = r'''# Try your functions (no check during the exam)
with wb.attempt("CP2.11"):
    rng_try = np.random.default_rng(0)
    print("ten choices with epsilon = 0.2:", [epsilon_greedy_action([0.2, 0.9, 0.5, 0.9], 0.2, rng_try) for _ in range(10)])
with wb.attempt("CP2.11"):
    print("estimates, exact mean   :", incremental_estimates([1, 0, 0, 1, 1, 1, 0, 1]))
    print("estimates, step of 0.25 :", incremental_estimates([1, 0, 0, 1, 1, 1, 0, 1], step=0.25))'''

CP211_SOLUTION = r'''def epsilon_greedy_action(q_values, epsilon, rng):
    """The arm chosen by the epsilon-greedy rule (an int), see CP2.11 a)."""
    q_values = np.asarray(q_values, dtype=float)
    if rng.random() < epsilon:                                   # explore: any arm, the best one included
        return int(rng.integers(len(q_values)))
    best = np.flatnonzero(q_values == q_values.max())            # all the arms of maximal estimate
    return int(rng.choice(best))                                 # ties broken at random


def incremental_estimates(rewards, step=None):
    """The list of the successive estimates Q, one after each reward, starting from Q = 0 (CP2.11 b)."""
    q, estimates = 0.0, []
    for n, reward in enumerate(rewards, start=1):
        a = 1 / n if step is None else step
        q = q + a * (reward - q)
        estimates.append(q)
    return estimates


rewards_11 = [1, 0, 0, 1, 1, 1, 0, 1]
print("exact means   :", np.round(incremental_estimates(rewards_11), 4).tolist())
print("step of 0.25  :", np.round(incremental_estimates(rewards_11, step=0.25), 4).tolist())'''

CP211_RECORD = r'''def constant_step(rewards, step, q):
    for reward in rewards:
        q = q + step * (reward - q)
    return q


wb.record("CP2.11c", incremental_estimates(rewards_11, step=0.25)[-1], decimals=4,
          mistakes={"c'est la moyenne exacte des récompenses : avec un pas constant, a = step à chaque récompense, pas 1/n":
                    float(np.mean(rewards_11)),
                    "l'énoncé part de Q = 0 : la première récompense ne remplace pas l'estimation, elle la déplace de a (R − Q)":
                    constant_step(rewards_11[1:], 0.25, float(rewards_11[0])),
                    "relis la mise à jour : Q ← Q + a (R − Q), où (R − Q) est l'écart entre la récompense et l'estimation":
                    constant_step(rewards_11, 0.75, 0.0)})'''

# ---------------------------------------------------------------------------
# Part B: checks after the exam
# ---------------------------------------------------------------------------
PART_B_INTRO = r'''## Partie B · Après l'examen : vérification automatique

**Ne commence cette partie qu'une fois ta copie terminée**, examen rendu. Elle vérifie le code de la partie A, puis les réponses que tu as écrites sur ta feuille, question par question : ✅ juste, ❌ faux (avec une piste), ⏳ pas rempli. Elle ne note pas : les démarches, les justifications et les questions rédigées se corrigent avec `03_examen_corrige.md`, qui donne aussi le barème.

1. Passe `EXAM_OVER` à `True` dans la cellule ci-dessous, puis exécute-la.
2. Exécute la vérification du code (CP2.10 et CP2.11).
3. Reporte ensuite chaque réponse de ta feuille : **la valeur** que tu as écrite, pas un nouveau calcul (sinon tu ne vérifies plus ta copie). En Python, le séparateur décimal est un **point** (`0.125`), et les lettres s'écrivent entre guillemets (`"NSN"`).

Si tu as fermé le notebook depuis l'examen, exécute d'abord la cellule de configuration (tout en haut) et les cellules de la partie A, sauf les essais.'''

PART_B_FLAG = r'''EXAM_OVER = False   # set it to True once your exam is finished, then run the cells of part B

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


def as_float(value):
    """A number returned by a function, as a float (None when it is not one)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None'''

CODE_CHECK_10 = r'''# Checks of cv_rmse_fixed (CP2.10)
if exam_over() and ready("CP2.10", "np", "X", "y", "CELLS", "cell_means", "lookup", "kfold", "ridge_fit", "ridge_predict",
                         "rmse", "cv_rmse_colleague", "cv_rmse_fixed"):
    PROPERTIES["CP2.10"] = []
    for sub in ("CP2.10c", "CP2.10d"):
        RESULTS.pop(sub, None)
    X_check, y_check = X.copy(), y.copy()
    cells_check = {size: cells.copy() for size, cells in CELLS.items()}
    fitted_on = []                             # the matrices that cv_rmse_fixed gives to ridge_fit
    given_ridge_fit = ridge_fit

    def ridge_fit(F, y, alpha):                # the same ridge_fit (same parameter names), which keeps a copy of F
        fitted_on.append(np.array(F, dtype=float, copy=True))
        return given_ridge_fit(F, y, alpha)

    values = None
    try:
        values = [cv_rmse_fixed(X_check, y_check, cells_check[size]) for size in (0.5, 0.1, 0.05)]
    except NotImplementedError:
        print("⏳ CP2.10 : cv_rmse_fixed n'est pas encore écrite.")
    except Exception as exc:                   # a bug of the exam code must not stop part B
        verdict("CP2.10", False, "", f"cv_rmse_fixed(X, y, cells) lève une erreur : {type(exc).__name__}: {exc}")
    finally:
        ridge_fit = given_ridge_fit
    if values is not None:
        unchanged = (np.array_equal(X_check, X) and np.array_equal(y_check, y)
                     and all(np.array_equal(cells_check[size], CELLS[size]) for size in CELLS))
        verdict("CP2.10", unchanged, "X, y et cells ne sont pas modifiés.",
                "ta fonction modifie X, y ou cells : calcule de nouveaux tableaux (np.column_stack, puis une "
                "standardisation qui crée un nouveau tableau) au lieu d'écrire dans ses arguments.")
        numbers = []
        for value in values:
            try:
                numbers.append(float(value))
            except (TypeError, ValueError):
                numbers.append(None)
        if any(v is None for v in values):
            verdict("CP2.10", False, "", "cv_rmse_fixed ne renvoie rien : as-tu oublié le return ?")
        elif any(n is None for n in numbers):
            verdict("CP2.10", False, "", f"cv_rmse_fixed doit renvoyer un nombre (la moyenne des k RMSE), pas "
                    f"un objet de type {type(values[0]).__name__}.")
        elif not all(np.isfinite(numbers)):
            verdict("CP2.10", False, "", "cv_rmse_fixed renvoie NaN ou l'infini : un district dont la case ne "
                    "contient aucun district d'entraînement du tour reçoit-il bien une valeur par défaut ?")
        else:
            n_train = len(kfold(len(y), 5)[0][0])
            if fitted_on:
                standardized = all(F.shape == (n_train, X.shape[1] + 1) and np.allclose(F.mean(axis=0), 0, atol=1e-6)
                                   and np.allclose(F.std(axis=0), 1, atol=1e-2) for F in fitted_on)
                verdict("CP2.10", standardized,
                        "ridge_fit reçoit les lignes d'entraînement du tour, standardisées avec leurs propres "
                        "moyennes et écarts-types.",
                        f"ridge_fit doit recevoir les {n_train} lignes d'entraînement du tour et leurs neuf colonnes, "
                        f"standardisées avec la moyenne et l'écart-type de ces lignes-là (chaque colonne de moyenne "
                        f"0 et d'écart-type 1).")
            else:
                print("ℹ️ CP2.10 : ta fonction n'appelle pas ridge_fit ; la standardisation n'est pas vérifiée.")
            leaky = cv_rmse_colleague(X, y, CELLS[0.1])
            if numbers[1] < leaky - 0.02:
                verdict("CP2.10", False, "", "ta valeur est nettement plus basse que la RMSE du collègue, alors que "
                        "supprimer sa fuite fait remonter l'erreur : une fuite plus forte reste-t-elle (le prix d'un "
                        "district de validation dans sa propre feature) ? Et renvoies-tu bien la moyenne des k RMSE "
                        "(des racines, pas des carrés) ?")
            else:
                verdict("CP2.10", numbers[1] > leaky + 0.05,
                        "sans la fuite du prix du quartier, la RMSE est nettement plus haute que celle du collègue.",
                        "ta RMSE reste trop près de celle du collègue : le prix du quartier des districts de "
                        "validation doit venir des seuls districts d'entraînement du tour.")
            reminder = ("relis l'énoncé de b) : dans chaque tour, d'où viennent les moyennes par case, la valeur par "
                        "défaut et la standardisation ?")
            code_check("CP2.10c", numbers[1], reminder)
            code_check("CP2.10d", numbers, reminder)'''

CODE_CHECK_11 = r'''# Checks of epsilon_greedy_action and incremental_estimates (CP2.11)
if exam_over() and ready("CP2.11", "np", "epsilon_greedy_action", "incremental_estimates"):
    PROPERTIES["CP2.11"] = []
    RESULTS.pop("CP2.11c", None)

    def choices(q, epsilon, n, seed):
        """n choices with a fresh copy of q at each call (a function that changes q must not skew the counts)."""
        rng_check = np.random.default_rng(seed)
        return [epsilon_greedy_action(list(q), epsilon, rng_check) for _ in range(n)]

    try:
        first = epsilon_greedy_action([0.2, 0.9, 0.5], 0.0, np.random.default_rng(0))
    except NotImplementedError:
        print("⏳ CP2.11 a) : epsilon_greedy_action n'est pas encore écrite.")
    except Exception as exc:                   # a bug of the exam code must not stop part B
        verdict("CP2.11", False, "", f"epsilon_greedy_action lève une erreur : {type(exc).__name__}: {exc}")
    else:
        if first is None:
            verdict("CP2.11", False, "", "epsilon_greedy_action ne renvoie rien : as-tu oublié le return ?")
        else:
            try:
                explored = epsilon_greedy_action([0.2, 0.9, 0.5], 1.0, np.random.default_rng(0))
                kinds = [type(v).__name__ for v in (first, explored)]
                verdict("CP2.11", all(isinstance(v, int) and not isinstance(v, bool) for v in (first, explored)),
                        "elle renvoie un int, en exploitant comme en explorant.",
                        f"elle doit renvoyer un int Python, en exploitant comme en explorant (ici : {kinds[0]} et "
                        f"{kinds[1]}) : return int(...) (le reste est vérifié quand même).")
                q_list, q_array = [0.2, 0.9, 0.5], np.array([0.2, 0.9, 0.5])
                for seed in range(20):
                    epsilon_greedy_action(q_list, 0.5, np.random.default_rng(seed))
                    epsilon_greedy_action(q_array, 0.5, np.random.default_rng(seed))
                verdict("CP2.11", q_list == [0.2, 0.9, 0.5] and q_array.tolist() == [0.2, 0.9, 0.5],
                        "q_values n'est pas modifié.",
                        "ta fonction modifie q_values : elle doit seulement le lire.")
                greedy = choices([0.2, 0.9, 0.5], 0.0, 300, 1)
                verdict("CP2.11", set(map(int, greedy)) == {1}, "avec ε = 0, toujours le meilleur bras.",
                        "avec ε = 0, elle doit toujours renvoyer le bras d'estimation maximale (ici 1).")
                tied = np.array(choices([0.5, 0.9, 0.9, 0.1], 0.0, 4000, 2), dtype=int)
                share = np.mean(tied == 1)
                verdict("CP2.11", set(tied.tolist()) <= {1, 2} and 0.45 < share < 0.55,
                        "les ex aequo sont tirés au hasard, à parts égales.",
                        "avec ε = 0 et deux maxima (les bras 1 et 2), chacun doit sortir environ une fois sur deux "
                        "(rng.choice parmi les bras d'estimation maximale ; np.argmax prend toujours le premier).")
                explore = np.array(choices([0.0, 1.0, 0.0, 0.0], 1.0, 8000, 3), dtype=int)
                counts = np.bincount(explore, minlength=4)
                verdict("CP2.11", len(counts) == 4 and np.all(np.abs(counts / 8000 - 0.25) < 0.02),
                        "avec ε = 1, chacun des quatre bras sort environ une fois sur quatre.",
                        f"avec ε = 1, l'exploration tire un bras parmi TOUS (rng.integers(K)) ; fréquences obtenues : "
                        f"{np.round(counts / 8000, 3).tolist()}.")
                mixed = np.array(choices([0.1, 0.3, 0.8, 0.2, 0.4], 0.2, 20000, 4), dtype=int)
                share_best = np.mean(mixed == 2)
                verdict("CP2.11", abs(share_best - 0.84) < 0.015,
                        "avec ε = 0,2 et cinq bras, le meilleur sort environ 84 % du temps (1 − ε + ε/K).",
                        f"avec ε = 0,2 et cinq bras, le meilleur devrait sortir 1 − ε + ε/K = 84 % du temps, pas "
                        f"{share_best * 100:.1f} %".replace(".", ",") + " : l'exploration se décide avec "
                        "rng.random() < epsilon, et inclut le meilleur bras.")
                verdict("CP2.11", choices([0.3, 0.3, 0.1], 0.5, 50, 5) == choices([0.3, 0.3, 0.1], 0.5, 50, 5),
                        "même générateur, mêmes choix.",
                        "deux générateurs de même graine doivent donner les mêmes choix : n'utilise que le rng reçu.")
            except Exception as exc:
                verdict("CP2.11", False, "", f"epsilon_greedy_action lève une erreur : {type(exc).__name__}: {exc}")
    rewards_check = [1, 0, 0, 1, 1, 1, 0, 1]
    try:
        means_out = incremental_estimates(list(rewards_check))
        steps_out = incremental_estimates(list(rewards_check), step=0.25)
    except NotImplementedError:
        print("⏳ CP2.11 b) : incremental_estimates n'est pas encore écrite.")
    except Exception as exc:
        verdict("CP2.11", False, "", f"incremental_estimates lève une erreur : {type(exc).__name__}: {exc}")
    else:
        means_arr = steps_arr = None
        if means_out is None or steps_out is None:
            verdict("CP2.11", False, "", "incremental_estimates ne renvoie rien : as-tu oublié le return ?")
        else:
            try:
                means_arr = np.asarray(list(means_out), dtype=float).ravel()     # list(): a generator works too
                steps_arr = np.asarray(list(steps_out), dtype=float).ravel()
            except (TypeError, ValueError):
                means_arr = steps_arr = None
                verdict("CP2.11", False, "", f"incremental_estimates doit renvoyer la liste des estimations, une par "
                        f"récompense, pas un objet de type {type(steps_out).__name__}.")
        if means_arr is not None:
            exact = np.cumsum(rewards_check) / np.arange(1, 9)
            verdict("CP2.11", means_arr.shape == (8,) and np.allclose(means_arr, exact),
                    "sans pas constant, les estimations sont les moyennes successives.",
                    "sans pas constant (step=None), la n-ième estimation doit être la moyenne des n premières "
                    "récompenses : Q ← Q + (R − Q)/n, en partant de Q = 0, une estimation par récompense.")
            if steps_arr.shape == (8,):
                code_check("CP2.11c", float(steps_arr[-1]),
                           "Q part de 0, puis Q ← Q + 0,25 (R − Q) à chaque récompense, dans l'ordre.")
            else:
                verdict("CP2.11", False, "", f"incremental_estimates doit renvoyer une estimation par récompense "
                        f"(8 ici), pas {steps_arr.size}.")'''

PAPER_CONTEXT = r'''# The paper questions only use the numbers of their statements (01_examen_sujet.md)
import math
from fractions import Fraction

import numpy as np


def ovo_votes(winners, classes="ABCDE"):
    """Votes of a one-versus-one from the winners of the duels, listed in the order of the statement."""
    return [winners.count(c) for c in classes]


WINNERS_2A = "ACAECBEDCD"                   # A-B, A-C, A-D, A-E, B-C, B-D, B-E, C-D, C-E, D-E
WINNERS_2B = "BCDABDBCED"
POINTS_3 = np.array([[2, 6], [3, 6], [4, 6], [6, 4], [9, 5], [9, 6]], dtype=float)


def assign_3(centres):
    d2 = ((POINTS_3[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1), d2.min(axis=1)


CENTRES0_3 = POINTS_3[[1, 2]]                                   # c1 = P2, c2 = P3
LABELS1_3, D2_0_3 = assign_3(CENTRES0_3)
CENTRES1_3 = np.array([POINTS_3[LABELS1_3 == k].mean(axis=0) for k in range(2)])
LABELS2_3, _ = assign_3(CENTRES1_3)
CENTRES2_3 = np.array([POINTS_3[LABELS2_3 == k].mean(axis=0) for k in range(2)])
LABELS3_3, D2_2_3 = assign_3(CENTRES2_3)
D2_PP_3 = ((POINTS_3 - POINTS_3[0]) ** 2).sum(axis=1)             # k-means++ from P1


def r_orange(d):
    return 2 * math.sqrt(d) - 1                                   # balloons centred at (±2, ..., ±2), radius 1


N_5 = 333
N_TEST_5 = math.ceil(0.2 * N_5)
REST_5 = N_5 - N_TEST_5
SE_5 = math.sqrt(0.94 * 0.06 / N_TEST_5)
x_6 = np.array([1, 2, 3, 4, 5], dtype=float)
y_6 = np.array([2, 3, 5, 4, 6], dtype=float)


def ridge_1d(x, y, lam):
    sxx = ((x - x.mean()) ** 2).sum()
    sxy = ((x - x.mean()) * (y - y.mean())).sum()
    w = sxy / (sxx + lam)
    return w, y.mean() - w * x.mean()


def sse_6(x, y, lam):
    w, b = ridge_1d(x, y, lam)
    return float(((y - w * x - b) ** 2).sum())


X_8 = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y_8 = [1, 1, 1, -1]                                               # NAND, 0 coded -1


def perceptron_epoch(w, b, labels=Y_8, strict=False):
    """One epoch of the perceptron rule of the fiche; returns (w, b, the values of z, the number of corrections)."""
    w, zs, corrections = list(w), [], 0
    for (x1, x2), label in zip(X_8, labels):
        z = w[0] * x1 + w[1] * x2 + b
        zs.append(z)
        if (label * z < 0) if strict else (label * z <= 0):
            w, b, corrections = [w[0] + label * x1, w[1] + label * x2], b + label, corrections + 1
    return w, b, zs, corrections


W1_8, B1_8, Z1_8, C1_8 = perceptron_epoch([0, 0], 0)
W2_8, B2_8, _, _ = perceptron_epoch(W1_8, B1_8)
TP_13, FP_13 = 120, 40
FN_13, TN_13 = 150 - TP_13, 1000 - 150 - FP_13
P_13, R_13 = TP_13 / (TP_13 + FP_13), TP_13 / (TP_13 + FN_13)
FPR_13 = FP_13 / (FP_13 + TN_13)


def entropy_bits(counts):
    p = np.asarray(counts, dtype=float) / sum(counts)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())'''

PAPER = [
    Paper("CP2.1", "Vrai ou faux justifiés (le verdict seulement ; les justifications se notent avec le corrigé)", [
        ("a", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("b", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("c", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("d", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("e", 'True / False (ou "vrai" / "faux")', "True", ""),
        ("f", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("g", 'True / False (ou "vrai" / "faux")', "True", ""),
        ("h", 'True / False (ou "vrai" / "faux")', "True", ""),
    ]),
    Paper("CP2.2", "Un-contre-tous, un-contre-un", [
        ("a", "[N_OvR, N_OvO] for K = 12", "[12, 12 * 11 // 2]",
         r'''mistakes={"chaque paire de classes ne fait qu'un duel : divise le nombre de couples ordonnés par 2": [12, 132],
                     "K² compte aussi des « duels » d'une classe contre elle-même : un duel oppose deux classes différentes": [12, 144],
                     "l'ordre demandé est [N_OvR, N_OvO]": [66, 12]}'''),
        ("b", "the smallest number of classes with at least 100 duels", "min(k for k in range(2, 100) if k * (k - 1) // 2 >= 100)",
         r'''mistakes={"le nombre de duels est K(K − 1)/2, pas K²": 10,
                     "chaque paire de classes ne fait qu'un duel : K(K − 1)/2 duels, pas K(K − 1)": 11,
                     "calcule K(K − 1)/2 pour ta réponse : atteint-il 100 ?": 14}'''),
        ("c", "votes of the first point [A, B, C, D, E]", "ovo_votes(WINNERS_2A)", ""),
        ("d", "the class predicted for the first point (a letter)", '"C"', ""),
        ("e", "votes of the second point [A, B, C, D, E]", "ovo_votes(WINNERS_2B)", ""),
        ("f", "the class predicted with mylearn's rule (a letter)", '"B"',
         r'''mistakes={"relis la règle d'égalité de mylearn : elle départage par l'indice des classes, sans regarder les duels": "D"}'''),
        ("g", "the class predicted with the duel rule (a letter)", '"D"',
         r'''mistakes={"ici, la règle départage les deux classes à égalité par le duel qui les a opposées, pas par leur indice": "B"}'''),
    ]),
    Paper("CP2.3", "Une itération de k-means et l'inertie obtenue", [
        ("a", "the cluster (1 or 2) of P1 to P6 after the first assignment", "(LABELS1_3 + 1).tolist()",
         r'''mistakes={"compare les distances au carré de P3 aux deux centres de DÉPART (c1 = P2, c2 = P3)": [1, 1, 1, 2, 2, 2],
                     "les numéros des clusters suivent ceux des centres : le cluster 1 est celui de c1 = P2": [2, 2, 1, 1, 1, 1],
                     "les clusters sont numérotés 1 et 2, comme leurs centres c1 et c2 (pas 0 et 1)": [0, 0, 1, 1, 1, 1]}'''),
        ("b", "the inertia with the initial centres", "float(D2_0_3.sum())",
         r'''decimals=2, mistakes={"l'inertie additionne des distances AU CARRÉ": float(np.sqrt(D2_0_3).sum())}'''),
        ("c", "the new centres [[x1, y1], [x2, y2]]", "CENTRES1_3.tolist()",
         r'''decimals=2, mistakes={"l'ordre demandé est [[x1, y1], [x2, y2]] : le centre du cluster 1 d'abord": CENTRES1_3[::-1].tolist()}'''),
        ("d", "the inertia with these centres, same assignment, 2 decimals",
         "float(((POINTS_3 - CENTRES1_3[LABELS1_3]) ** 2).sum())",
         r'''decimals=2, mistakes={"l'inertie additionne des distances AU CARRÉ": float(np.sqrt(((POINTS_3 - CENTRES1_3[LABELS1_3]) ** 2).sum(axis=1)).sum()),
                     "d) garde l'affectation de a), mais mesure les distances aux NOUVEAUX centres de c), pas aux centres de départ": float(D2_0_3.sum())}'''),
        ("e", "the number (1 to 6) of the point that changes cluster", "int(np.flatnonzero(LABELS1_3 != LABELS2_3)[0] + 1)",
         r'''mistakes={f"compare les distances au carré de chaque point aux deux NOUVEAUX centres de c) : P{point} reste du même côté": point
                     for point in range(1, 7) if point != int(np.flatnonzero(LABELS1_3 != LABELS2_3)[0] + 1)}'''),
        ("f", "the centres after the second update", "CENTRES2_3.tolist()",
         r'''decimals=2, mistakes={"l'ordre demandé est [[x1, y1], [x2, y2]] : le centre du cluster 1 d'abord": CENTRES2_3[::-1].tolist()}'''),
        ("g", "the final inertia", "float(D2_2_3.sum())", "decimals=2"),
        ("h", "True or False", "bool((LABELS3_3 != LABELS2_3).any())", ""),
        ("i", "probability that the second centre is P4, P5 or P6 (3 decimals)", "float(D2_PP_3[3:].sum() / D2_PP_3.sum())",
         r'''decimals=3, mistakes={"k-means++ tire chaque point avec une probabilité proportionnelle au CARRÉ de sa distance D(x)": float(np.sqrt(D2_PP_3[3:]).sum() / np.sqrt(D2_PP_3).sum()),
                     "les points ne sont pas tirés uniformément : chacun a une probabilité proportionnelle à D(x)²": 0.6,
                     "le premier centre, P1, a D(P1) = 0 : il ne peut pas être tiré une seconde fois": 0.5,
                     "la question porte sur l'un des TROIS points P4, P5, P6": float(D2_PP_3[4] / D2_PP_3.sum())}'''),
    ]),
    Paper("CP2.4", "Densité d'échantillons et hyper-orange", [
        ("a", "[density in dimension 3, in dimension 6], 3 decimals", "[2000 / 4 ** 3, 2000 / 4 ** 6]",
         r'''decimals=3, mistakes={"en dimension d, il y a b^d cases (b par axe), pas b × d": [2000 / 12, 2000 / 24]}'''),
        ("b", "number of samples for a density of 2 in dimension 8", "2 * 4 ** 8",
         r'''mistakes={"c'est le nombre de cases : multiplie-le par la densité voulue": 4 ** 8,
                     "le nombre de cases est b^d : la base est le nombre de cases par axe, l'exposant la dimension": 2 * 8 ** 4}'''),
        ("c", "the smallest dimension where the density falls below 0.01", "min(d for d in range(1, 30) if 2000 / 4 ** d < 0.01)",
         r'''mistakes={"calcule la densité dans cette dimension : elle n'est pas encore sous 0,01": 8}'''),
        ("d", "radius of the orange in dimension 9", "r_orange(9)",
         r'''decimals=3, mistakes={"le centre d'un ballon a toutes ses coordonnées égales à ±2 : sa distance à l'origine vaut 2√d, pas √d": math.sqrt(9) - 1,
                     "l'orange touche les ballons : son rayon est la distance au centre d'un ballon MOINS le rayon du ballon": 2 * math.sqrt(9)}'''),
        ("e", "the smallest dimension where the orange leaves the box", "min(d for d in range(1, 30) if r_orange(d) > 3)",
         r'''mistakes={"dans cette dimension, l'orange touche les faces de la boîte sans en sortir : elle sort quand son rayon DÉPASSE la demi-largeur 3": 4,
                     "c'est la dimension où sortait l'orange de la fiche (boîte de côté 4, ballons centrés en ±1) : ici, la boîte a pour côté 6 et les ballons sont centrés en ±2": 10}'''),
        ("f", "share of the volume in the 5 % skin in dimension 30, 3 decimals", "1 - 0.95 ** 30",
         r'''decimals=3, mistakes={"c'est la part du volume à l'INTÉRIEUR de la boule de rayon 0,95 : la peau, c'est le reste": 0.95 ** 30,
                     "une peau de 5 % du rayon laisse une boule intérieure de rayon 0,95": 1 - 0.9 ** 30}'''),
    ]),
    Paper("CP2.5", "Plan d'évaluation", [
        ("a", "size of the test set", "N_TEST_5",
         r'''mistakes={"la taille du test s'arrondit vers le HAUT : ⌈t·n⌉": 66}'''),
        ("b", "sizes of the 5 folds, in the order of the folds", "[REST_5 // 5 + (i < REST_5 % 5) for i in range(5)]",
         r'''mistakes={"les n mod k PREMIERS folds reçoivent l'exemple de plus": [53, 53, 53, 53, 54],
                     "la validation croisée porte sur les manchots qui RESTENT une fois le test mis de côté": [67, 67, 67, 66, 66]}'''),
        ("c", "total number of trainings", "15 * 5 + 1",
         r'''mistakes={"n'oublie pas le réentraînement final du réglage retenu": 75,
                     "chaque réglage demande un entraînement PAR FOLD": 16}'''),
        ("d", "total number of trainings with the nested cross-validation", "5 * (15 * 5 + 1)",
         r'''mistakes={"dans chaque tour extérieur, le réglage retenu est réentraîné une fois sur la partie qui reste": 375,
                     "l'énoncé ne prévoit pas de réentraînement après les cinq tours extérieurs": 381,
                     "chaque tour extérieur fait UNE recherche de c), pas une par fold intérieur": 5 * 5 * (15 * 5 + 1),
                     "d) ne compte que les cinq tours extérieurs : la recherche finale qui produirait le modèle livré n'en fait pas partie": 5 * (15 * 5 + 1) + 15 * 5 + 1}'''),
        ("e", "[standard error, half-width 1.96 SE], 3 decimals", "[SE_5, 1.96 * SE_5]",
         r'''decimals=3, mistakes={"n est la taille du jeu de TEST, sur lequel l'accuracy a été mesurée": [math.sqrt(0.94 * 0.06 / N_5), 1.96 * math.sqrt(0.94 * 0.06 / N_5)],
                     "la demi-largeur de l'intervalle à 95 % utilise 1,96, pas 2": [SE_5, 2 * SE_5]}'''),
        ("f", "the numbers of the protocols with a leak (a list)", "[1, 2, 5]",
         r'''ordered=False, mistakes={"il manque une fuite : relis les protocoles qui calculent quelque chose sur des données avant de les découper": [1, 5],
                     "il manque une fuite : un prétraitement fait avant le tirage même du test compte aussi": [2, 5],
                     "il manque une fuite : un choix fait en regardant le test est aussi une fuite": [1, 2],
                     "un protocole de trop : un prétraitement ajusté sur l'entraînement du tour seulement n'est pas une fuite": [1, 2, 3, 5],
                     "un protocole de trop : évaluer une seule fois sur le test, après le choix, est la bonne pratique": [1, 2, 4, 5]}'''),
    ]),
    Paper("CP2.6", "Ridge en dimension 1", [
        ("a", "[w, b] for lambda = 0", "list(ridge_1d(x_6, y_6, 0))",
         r'''decimals=3, mistakes={"le modèle a une ordonnée à l'origine : b = ȳ − w x̄": [ridge_1d(x_6, y_6, 0)[0], 0.0],
                     "l'ordre demandé est [w, b]": list(ridge_1d(x_6, y_6, 0))[::-1],
                     "w* utilise les écarts aux moyennes : S_xy et S_xx": [float(x_6 @ y_6 / (x_6 @ x_6)), float(y_6.mean() - x_6 @ y_6 / (x_6 @ x_6) * x_6.mean())]}'''),
        ("b", "[w, b] for lambda = 5", "list(ridge_1d(x_6, y_6, 5))",
         r'''decimals=3, mistakes={"b dépend de w : recalcule b = ȳ − w x̄ avec le nouveau w": [ridge_1d(x_6, y_6, 5)[0], ridge_1d(x_6, y_6, 0)[1]],
                     "w* utilise S_xy et S_xx, calculés sur les écarts aux moyennes": [float(x_6 @ y_6 / (x_6 @ x_6 + 5)), float(y_6.mean() - x_6 @ y_6 / (x_6 @ x_6 + 5) * x_6.mean())]}'''),
        ("c", "[sum of squared residuals for lambda = 0, for lambda = 5]", "[sse_6(x_6, y_6, 0), sse_6(x_6, y_6, 5)]",
         r'''decimals=3, mistakes={"l'ordre demandé est λ = 0, puis λ = 5": [sse_6(x_6, y_6, 5), sse_6(x_6, y_6, 0)],
                     "c) demande les résidus seuls, sans la pénalité λ w²": [sse_6(x_6, y_6, 0), sse_6(x_6, y_6, 5) + 5 * ridge_1d(x_6, y_6, 5)[0] ** 2],
                     "c) demande la SOMME des carrés des résidus, pas leur moyenne": [sse_6(x_6, y_6, 0) / 5, sse_6(x_6, y_6, 5) / 5]}'''),
        ("d", "prediction of the new model for x' = 50, 3 decimals", "sum(np.array(ridge_1d(10 * x_6, y_6, 5)) * [50, 1])",
         r'''decimals=3, mistakes={"les données ont changé d'unité : recalcule S_x'x', S_x'y, w et b avec x' = 10 x": 5.2,
                     "λ = 5 compte encore : w = S_x'y / (S_x'x' + λ), ce n'est pas tout à fait la pente des moindres carrés": 5.8,
                     "l'ordonnée à l'origine change aussi : b' = ȳ − w' x̄'": 2.2 + ridge_1d(10 * x_6, y_6, 5)[0] * 50,
                     "le point x = 5 s'écrit x' = 50 dans la nouvelle unité": sum(np.array(ridge_1d(10 * x_6, y_6, 5)) * [5, 1])}'''),
    ]),
    Paper("CP2.7", "Trois paires de courbes d'apprentissage (la lecture de c se note avec le corrigé)", [
        ("a", "the panel (1, 2 or 3) of the underfitting", "2",
         r'''mistakes={"l'underfitting se lit sur le NIVEAU des deux erreurs, comparé au plancher du bruit": 1,
                     "l'underfitting se lit sur le NIVEAU des deux erreurs, comparé au plancher du bruit (relis la fiche du ch. 9, §9.2)": 3}'''),
        ("b", "the panel of the overfitting with few examples", "3",
         r'''mistakes={"l'overfitting se lit sur l'ÉCART entre les deux courbes : une erreur d'entraînement nettement plus basse que celle de validation": 1,
                     "l'overfitting se lit sur l'ÉCART entre les deux courbes : une erreur d'entraînement nettement plus basse que celle de validation (relis la fiche du ch. 9, §9.2)": 2}'''),
        ("d", "the letter of the action for panel 2", '"B"',
         r'''mistakes={"plus de données réduit la variance ; relis ce que dit le niveau des deux courbes du panneau 2": "A",
                     "une régularisation plus forte réduit encore la capacité du modèle": "C",
                     "l'early stopping limite lui aussi la capacité du modèle": "D"}'''),
        ("e", 'True / False (ou "vrai" / "faux")', "False", ""),
    ]),
    Paper("CP2.8", "Perceptron : NAND, puis XOR", [
        ("a", "the four values of z during the first epoch", "Z1_8",
         r'''mistakes={"z se calcule avec les poids et le biais tels qu'ils sont APRÈS la correction de l'exemple précédent": [0, 0, 0, 0]}'''),
        ("b", "[w1, w2, b] after the first epoch", "W1_8 + [B1_8]",
         r'''mistakes={"la sortie 0 se code y = −1 : avec y = 0, l'exemple (1, 1) ne corrige rien": [0, 0, 1],
                     "une somme nulle compte comme une erreur (y z ≤ 0) : avec y z < 0, partis de zéro, les poids ne bougeraient jamais": [0, 0, 0],
                     "l'exemple (1, 1) corrige aussi le biais : b ← b + η y": [-1, -1, 1]}'''),
        ("c", "the number of corrections during the first epoch", "C1_8",
         r'''mistakes={"une somme nulle compte comme une erreur : y z ≤ 0": 1,
                     "un exemple bien classé (y z > 0) ne déclenche aucune correction": 4}'''),
        ("d", "the number of inputs well classified with the weights of b)",
         "sum((1 if W1_8[0] * x1 + W1_8[1] * x2 + B1_8 > 0 else -1) == label for (x1, x2), label in zip(X_8, Y_8))",
         r'''mistakes={"la prédiction vaut +1 seulement si z est STRICTEMENT positif : applique-la aux quatre entrées": 2,
                     "relis la règle de prédiction de d) : +1 quand z > 0, −1 sinon": 3}'''),
        ("e", "[w1, w2, b] after the second epoch", "W2_8 + [B2_8]",
         r'''mistakes={"une somme nulle compte comme une erreur (y z ≤ 0) : avec y z < 0, partis de zéro, les poids ne bougeraient jamais": [0, 0, 0],
                     "la deuxième époque repart des poids de b) et passe sur les QUATRE exemples": [-1, 0, 2]}'''),
    ]),
    Paper("CP2.9", "Syllogismes et sophismes", [
        ("a", "six letters S, V or N, in order (for example \"SSSSSS\")", '"NSNVNN"',
         r'''mistakes={"relis le raisonnement 2 : ses deux prémisses sont-elles vraies ?": "NVNVNN",
                     "relis le raisonnement 4 : une forme valide ne suffit pas, vérifie ses prémisses": "NSNSNN",
                     "relis le raisonnement 3 : écris sa forme avec des lettres (si … alors …)": "NSVVNN",
                     "relis le raisonnement 6 : écris sa forme avec des lettres (si … alors …)": "NSNVNV",
                     "relis le raisonnement 1 : la conclusion découle-t-elle vraiment des deux prémisses ?": "VSNVNN"}'''),
        ("b", "the letters of the fallacies of the non-valid arguments, in order", '"EBCA"',
         r'''mistakes={"relis le raisonnement 1 : il n'a pas la forme « si … alors … »": "ABCA",
                     "relis le raisonnement 3 : sa deuxième prémisse nie-t-elle la condition, ou affirme-t-elle la conséquence ?": "EACA",
                     "relis le raisonnement 5 : quel terme est distribué dans la conclusion sans l'être dans sa prémisse ?": "EBDA",
                     "relis le raisonnement 6 : sa deuxième prémisse nie-t-elle la condition, ou affirme-t-elle la conséquence ?": "EBCB",
                     "relis les raisonnements 3 et 6 : lequel nie la condition, lequel affirme la conséquence ?": "EACB"}'''),
        ("c", 'True / False (ou "vrai" / "faux")', "False", ""),
    ]),
    Paper("CP2.13", "Parties antérieures : matrice de confusion, Bayes et entropie", [
        ("a", "[TP, FP, FN, TN]", "[TP_13, FP_13, FN_13, TN_13]",
         r'''mistakes={"FP : les e-mails signalés qui ne sont pas des spams ; FN : les spams non signalés": [TP_13, FN_13, FP_13, TN_13],
                     "TN : les e-mails qui ne sont ni des spams ni signalés (les 1 000 e-mails se répartissent dans les quatre cases)": [TP_13, FP_13, FN_13, TN_13 + FP_13]}'''),
        ("b", "[precision, recall, F1], 3 decimals", "[P_13, R_13, 2 * P_13 * R_13 / (P_13 + R_13)]",
         r'''decimals=3, mistakes={"precision = TP / (TP + FP) et recall = TP / (TP + FN)": [R_13, P_13, 2 * P_13 * R_13 / (P_13 + R_13)],
                     "le F1 est la moyenne HARMONIQUE de la precision et du recall": [P_13, R_13, (P_13 + R_13) / 2]}'''),
        ("c", "accuracy, 2 decimals", "(TP_13 + TN_13) / 1000",
         r'''decimals=2, mistakes={"c'est la moyenne du recall et de la spécificité : l'accuracy compte les bonnes réponses parmi les 1 000 e-mails": (R_13 + TN_13 / (TN_13 + FP_13)) / 2}'''),
        ("d", "P(spam | flagged) with 40 % of spams, 3 decimals", "R_13 * 0.4 / (R_13 * 0.4 + FPR_13 * 0.6)",
         r'''decimals=3, mistakes={"la proportion de spams a changé : recalcule P(signalé) avec 40 % de spams": P_13,
                     "c'est P(signalé | spam) : on demande P(spam | signalé)": R_13,
                     "le taux de faux positifs est FP / (FP + TN) : la part des e-mails normaux signalés à tort": R_13 * 0.4 / (R_13 * 0.4 + (1 - FPR_13) * 0.6),
                     "c'est P(spam et signalé) : divise par P(signalé)": R_13 * 0.4}'''),
        ("e", "entropy of the first cluster in bits, 3 decimals", "entropy_bits([30, 10])",
         r'''decimals=3, mistakes={"c'est en nats : on demande des bits (log₂)": entropy_bits([30, 10]) * math.log(2),
                     "additionne les termes −p log₂ p des DEUX espèces": -0.75 * math.log2(0.75),
                     "le cluster ne contient que deux espèces, et pas à parts égales": math.log2(3)}'''),
        ("f", "entropy of the second cluster in bits", "entropy_bits([20, 10, 10])",
         r'''decimals=3, mistakes={"les trois espèces ne sont pas équiprobables dans ce cluster": math.log2(3),
                     "c'est en nats : on demande des bits (log₂)": entropy_bits([20, 10, 10]) * math.log(2)}'''),
    ]),
]

SUMMARY = r'''# Summary of the automatic checks
import re


def natural(sub_id):
    """CP2.2a before CP2.13a: the question number as a number, then the letter."""
    number, letter = re.match(r"CP2\.(\d+)(.*)", sub_id).groups()
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
    print(f"Propriétés du code (CP2.10, CP2.11) : {sum(checked)} sur {len(checked)}.")
    print("Note ensuite ta copie avec le barème de 03_examen_corrige.md : il donne aussi les points des démarches, "
          "des justifications et des questions rédigées, que cette vérification ne voit pas.")'''


def paper_cell(kind: str, paper: Paper) -> list:
    """One paper question of part B: answers + guarded wb.check (exam), or wb.record (solutions)."""
    cells = [md(f"**{paper.id} — {paper.title}**")]
    stem = f"answer_{paper.id.replace('.', '_')}"
    if kind == "exercise":
        lines = [f"# {paper.id}: the values written on your answer sheet"]
        lines += [f"{stem}{letter} = ...  # {letter}) {hint}" for letter, hint, _, _ in paper.subs]
        names = ", ".join(f"{stem}{letter}" for letter, *_ in paper.subs)
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
        title = "# Checkpoint II · Examen blanc — notebook de l'examen"
        how = ("Ce notebook accompagne le sujet `01_examen_sujet.md`.\n\n"
               "- **Partie A, pendant l'examen** : les deux questions de code, CP2.10 (🐛, 1,5 point) et CP2.11 (🔨, "
               "1 point). Écris ton code dans les cellules à compléter, sans `mylearn`, sans scikit-learn, sans les "
               "corrigés et sans assistant IA ; `help()` et la documentation officielle de NumPy sont permis. Rien "
               "n'est vérifié pendant l'examen : les cellules « essai » montrent seulement ce que fait ton code. Tes "
               "réponses chiffrées et tes explications vont sur ta feuille (`04_mes_reponses.md`).\n"
               "- **Partie B, après l'examen** : la vérification automatique de ton code et des réponses de ta "
               "feuille. Ses cellules ne vérifient rien tant que `EXAM_OVER` vaut `False`.\n\n"
               "« Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche ⏳ "
               "(partie A) ou ⏸️ (partie B).\n\n"
               "> Travaille dans **ta copie** (`mon_travail/checkpoints/partie_2/02_examen_notebook.ipynb`, créée "
               "par `python tools/start_chapter.py CP2`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# Checkpoint II · Examen blanc — solutions (notebook exécuté)"
        how = ("Les solutions des deux questions de code (CP2.10 et CP2.11), exécutées, puis l'enregistrement des "
               "réponses que vérifie la partie B du notebook de l'examen (cellules marquées `answer`, récoltées "
               "par `tools/build_answers.py`). Le corrigé détaillé et le barème sont dans `03_examen_corrige.md`.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell("demo"))
    cells.append(md("## Partie A · Pendant l'examen : les deux questions de code" if kind == "exercise"
                    else "## Partie A · Les deux questions de code"))
    cells.append(code(GIVEN))
    cells.append(md(CP210_MD))
    cells.append(code(CP210_BUG))
    if kind == "exercise":
        cells += [code(CP210_TODO), code(CP210_TRY)]
    else:
        cells += [code(CP210_SOLUTION), code(CP210_RECORD, tags=["answer"])]
    cells.append(md(CP211_MD))
    if kind == "exercise":
        cells += [code(CP211_TODO), code(CP211_TRY)]
        cells.append(md("---\n\n### 🛑 Fin de la partie A\n\nL'examen s'arrête ici. Rends ta copie (enregistre ce "
                        "notebook et ta feuille), puis, seulement après, passe à la partie B."))
        cells.append(md(PART_B_INTRO))
        cells.append(code(PART_B_FLAG))
        cells.append(md("### Le code de la partie A"))
        cells.append(code(CODE_CHECK_10))
        cells.append(code(CODE_CHECK_11))
        cells.append(md("### Les réponses de ta feuille"))
        for paper in PAPER:
            cells += paper_cell(kind, paper)
        cells.append(md("### Bilan"))
        cells.append(code(SUMMARY))
        cells.append(md("**Ce que cette vérification ne voit pas** : les démarches et les justifications ; les "
                        "explications et les démonstrations (CP2.1, CP2.3 j, CP2.6 1 à 3, d et e, CP2.7 f, CP2.8 f "
                        "et g, CP2.10 a et d, CP2.12) ; les lectures de la figure de CP2.7 (c) ; et, pour CP2.10 et "
                        "CP2.11, les nombres recopiés sur ta feuille (seules tes fonctions sont vérifiées). Tout cela se "
                        "corrige avec `03_examen_corrige.md`. Ensuite : la remédiation du corrigé (relis la synthèse, "
                        "`05_synthese.md`, pour les notions ratées), puis le mini-projet."))
    else:
        cells += [code(CP211_SOLUTION), code(CP211_RECORD, tags=["answer"])]
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
