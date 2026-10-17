#!/usr/bin/env python
"""Build the two notebooks of the mock exam of part III from a single source (used by Claude).

    python tools/chapters/figures_cp3.py
    python tools/chapters/build_cp3.py
    python tools/run_all_notebooks.py checkpoints/partie_3/03_examen_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py checkpoints/partie_3/02_examen_notebook.ipynb

The exam notebook has two parts. Part A is done DURING the exam: the two code
questions CP3.6 (🐛, with scikit-learn) and CP3.7 (🔨, NumPy only), without mylearn and
without any check. Part B is done AFTER the exam: it checks the code of part A, then the
answers written on paper (sub-IDs CP3.1a, CP3.2a, ...). Its cells only run once
``EXAM_OVER = True``. The solutions notebook records every checked answer with ``wb.record``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import Paper, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "checkpoints/partie_3"
EXAM = f"{FOLDER}/02_examen_notebook.ipynb"
SOLUTIONS = f"{FOLDER}/03_examen_solutions.ipynb"

# ---------------------------------------------------------------------------
# Data and tools shared by both notebooks
# ---------------------------------------------------------------------------
GIVEN = r'''# Data and tools of the two code questions (run this cell first)
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# CP3.6: 4 000 districts of California (1990 census)
california = wb.datasets.load_california()
FEATURES = california.columns[:8].tolist()                    # MedInc, HouseAge, AveRooms, AveBedrms, Population, ...
rows = np.sort(np.random.default_rng(2606).choice(len(california), 4000, replace=False))
X = california[FEATURES].to_numpy(dtype=float)[rows]          # shape (4000, 8)
y = california["MedHouseVal"].to_numpy(dtype=float)[rows]     # median house value (hundreds of thousands of $)
K_GRID = [1, 3, 5, 10, 20, 40]                                # numbers of neighbours tried


# CP3.7: the probabilities of a binary classifier, for 600 validation and 600 test examples
def make_scores(n, seed):
    """n labels (about 25 % of positives) and their probability of being positive (calibrated, 3 decimals)."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.25).astype(int)
    score = rng.normal(np.where(labels == 1, 1.0, -1.0), 1.0)
    return np.round(1 / (1 + 3 * np.exp(-2 * score)), 3), labels


PROBA_VAL, Y_VAL = make_scores(600, 3707)
PROBA_TEST, Y_TEST = make_scores(600, 3708)
THRESHOLDS = np.round(np.arange(1, 100) / 100, 2)             # 0.01, 0.02, ..., 0.99
COST_FN, COST_FP = 5, 1                                       # a false negative costs 5, a false positive 1
_ORIGINALS = {"X": X.copy(), "y": y.copy(), "PROBA_VAL": PROBA_VAL.copy(), "Y_VAL": Y_VAL.copy(),
              "PROBA_TEST": PROBA_TEST.copy(), "Y_TEST": Y_TEST.copy()}   # part B restores them if a try changed them
print("X:", X.shape, "· y:", y.shape, "· validation:", PROBA_VAL.shape, "· positives:", int(Y_VAL.sum()))'''

# ---------------------------------------------------------------------------
# Part A: the two code questions of the exam
# ---------------------------------------------------------------------------
CP36_MD = r'''### CP3.6 — Trouver trois fuites dans un script d'entraînement 🐛 ★★ ⏱️ 12 min · 2 points
**Objectif :** repérer ce qu'une évaluation a appris de son jeu de test, puis refaire l'évaluation sans fuite.

Un collègue prédit la valeur médiane des logements des 4 000 districts de `X` et `y` avec un kNN de régression, dont il choisit le nombre de voisins dans `K_GRID`. Sa fonction `colleague_report(X, y)`, dont les lignes sont marquées (A) à (F), annonce au client un $R^2$ de test de 0,93. Exécute-la (cellule suivante).

a) Trois de ses lignes contiennent une fuite de données : leurs lettres (sur ta feuille). **[0,6]**
b) Laquelle de ces fuites fausse vraiment son score, et pourquoi ? Pourquoi les deux autres ne le changent-elles presque pas ici, et pourquoi faut-il quand même les corriger ? (sur ta feuille) **[0,4]**
c) Écris `honest_report(X, y)`, qui fait le même travail **sans fuite** et renvoie `(best_k, test_r2)` : garde la transformation (B), qui n'est pas une fuite, et retire ce qui en est une ; découpe d'abord, avec le même `train_test_split(..., test_size=0.25, random_state=0)` ; choisis le nombre de voisins par une `GridSearchCV` à 5 folds (`cv=5`, score $R^2$ par défaut) sur l'entraînement seul, avec un `Pipeline` qui standardise puis applique le kNN ; mesure enfin le $R^2$ de test, une seule fois. La fonction ne modifie pas `X`. **[0,6]**
d) Les valeurs que renvoie ta fonction : $[\text{best\_k}, R^2_{\text{test}}]$ (3 décimales, sur ta feuille). **[0,2]**
e) Que dis-tu au client ? (sur ta feuille) **[0,2]**'''

CP36_BUG = r'''# The colleague's code (do not change this cell: run it and read what it prints)
def colleague_report(X, y):
    X = np.column_stack([X, y / X[:, 2]])                                        # (A) a 9th feature: the value per room
    X[:, 4] = np.log1p(X[:, 4])                                                  # (B) the log of the population, row by row
    X = StandardScaler().fit_transform(X)                                        # (C) standardize the 9 columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)   # (D) split
    scores = {k: KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)
              for k in K_GRID}                                                   # (E) the R² of each k on the test
    best = max(scores, key=scores.get)
    return best, scores[best]                                                    # (F) the k kept and its test R²


best_colleague, r2_colleague = colleague_report(X, y)
print(f"the colleague keeps k = {best_colleague}, with a test R² of {r2_colleague:.3f}")'''

CP36_TODO = r'''def honest_report(X, y):
    """The same evaluation without leakage: returns (best_k, test_r2), see CP3.6 c)."""
    raise NotImplementedError("honest_report")  # TODO CP3.6 c)'''

CP36_TRY = r'''# Try your function (no check during the exam: compare with what you expect)
with wb.attempt("CP3.6"):
    print("honest report:", honest_report(X, y))'''

CP36_SOLUTION = r'''# a) (A) builds a feature from the target y itself (y = value per room x AveRooms: the kNN reads the answer);
#    (C) learns the means and stds on all the districts, test included; (E) chooses k by its score ON THE TEST
#    and reports that same score. (B) is computed row by row: nothing learnt, no leak. (D) and (F) are fine.
def honest_report(X, y):
    """The same evaluation without leakage: returns (best_k, test_r2), see CP3.6 c)."""
    X = X.copy()                                                    # do not modify the caller's array
    X[:, 4] = np.log1p(X[:, 4])                                     # (B) kept: a row-by-row transformation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    search = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                          {"kneighborsregressor__n_neighbors": K_GRID}, cv=5).fit(X_train, y_train)
    return search.best_params_["kneighborsregressor__n_neighbors"], search.score(X_test, y_test)


best_6, r2_6 = honest_report(X, y)
print(f"honest report: k = {best_6}, test R² = {r2_6:.4f} (the colleague: k = {best_colleague}, {r2_colleague:.4f})")'''

CP36_RECORD = r'''# Classic wrong versions of c), each recorded with its message
def variant_6(*, target_feature=False, keep_log=True, scaler_all=False, cv_score=False):
    """[best_k, R²] of honest_report written another (wrong) way."""
    Xv = X.copy()
    if target_feature:
        Xv = np.column_stack([Xv, y / Xv[:, 2]])
    if keep_log:
        Xv[:, 4] = np.log1p(Xv[:, 4])
    if scaler_all:
        Xv = StandardScaler().fit_transform(Xv)
    X_train, X_test, y_train, y_test = train_test_split(Xv, y, test_size=0.25, random_state=0)
    search = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                          {"kneighborsregressor__n_neighbors": K_GRID}, cv=5).fit(X_train, y_train)
    best = search.best_params_["kneighborsregressor__n_neighbors"]
    return [best, float(search.best_score_ if cv_score else search.score(X_test, y_test))]


wb.record("CP3.6d", [best_6, r2_6], decimals=4, mistakes={
    "ta fonction garde la feature (A), calculée à partir de y : c'est la fuite qui fait le score du collègue":
    variant_6(target_feature=True),
    "ta fonction a retiré la transformation (B) : l'énoncé demande de la garder, puisqu'elle se calcule ligne par ligne":
    variant_6(keep_log=False),
    "c'est le score moyen de validation croisée de la recherche (best_score_) : on demande le R² sur le jeu de test":
    variant_6(cv_score=True)})'''

CP37_MD = r'''### CP3.7 — Coder le seuil de décision de coût minimal et le comparer au seuil théorique 🔨 ★★ ⏱️ 14 min · 2 points
**Objectif :** choisir un seuil de décision d'après le coût des erreurs, sur la validation, et le confronter au calcul théorique.

Avec NumPy seulement. `PROBA_VAL` donne, pour 600 exemples de validation, la probabilité d'être positif qu'annonce un classifieur, et `Y_VAL` leurs labels (0 ou 1) ; `PROBA_TEST` et `Y_TEST`, la même chose pour 600 exemples de test. Un faux négatif coûte `COST_FN = 5`, un faux positif `COST_FP = 1`. La règle de décision : classe 1 quand la probabilité est **supérieure ou égale** au seuil $t$.

a) Écris `total_cost(y_true, y_pred, cost_fn, cost_fp)`, qui renvoie le coût total (un `int`) de décisions 0/1. **[0,3]**
b) Écris `min_cost_threshold(proba, y_true, cost_fn, cost_fp, thresholds)`, qui renvoie le couple `(t, coût)` du seuil de plus petit coût parmi `thresholds` (le premier dans l'ordre de `thresholds` en cas d'égalité). Les deux fonctions ne modifient pas leurs arguments. **[0,5]**
c) Avec `THRESHOLDS` (0,01 ; 0,02 ; … ; 0,99), le seuil et le coût que donne ta fonction sur la validation : $[t, \text{coût}]$ (sur ta feuille). **[0,3]**
d) Le seuil théorique de coût moyen minimal, si les probabilités sont calibrées (4 décimales, sur ta feuille). Montre-le en comparant le coût moyen des deux décisions pour un exemple de probabilité $p$. **[0,3]**
e) Les coûts sur le test, avec le seuil de c), puis avec le seuil 0,5 : $[\text{coût}(t), \text{coût}(0{,}5)]$ (sur ta feuille). **[0,3]**
f) Pourquoi c) et d) ne donnent-ils pas le même seuil ? Lequel déploierais-tu, et pourquoi ? (sur ta feuille) **[0,3]**'''

CP37_TODO = r'''def total_cost(y_true, y_pred, cost_fn, cost_fp):
    """Total cost (an int) of the 0/1 decisions y_pred, see CP3.7 a)."""
    raise NotImplementedError("total_cost")  # TODO CP3.7 a)


def min_cost_threshold(proba, y_true, cost_fn, cost_fp, thresholds):
    """(t, cost) of the threshold of smallest total cost among thresholds (the first one if tied), CP3.7 b)."""
    raise NotImplementedError("min_cost_threshold")  # TODO CP3.7 b)'''

CP37_TRY = r'''# Try your functions (no check during the exam)
with wb.attempt("CP3.7"):
    print("cost of [1, 0, 1] for the labels [1, 1, 0]:", total_cost(np.array([1, 1, 0]), np.array([1, 0, 1]), 5, 1))
with wb.attempt("CP3.7"):
    chosen = min_cost_threshold(PROBA_VAL, Y_VAL, COST_FN, COST_FP, THRESHOLDS)
    print("validation: (t, cost) =", chosen)'''

CP37_SOLUTION = r'''def total_cost(y_true, y_pred, cost_fn, cost_fp):
    """Total cost (an int) of the 0/1 decisions y_pred, see CP3.7 a)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return int(cost_fn * false_negatives + cost_fp * false_positives)


def min_cost_threshold(proba, y_true, cost_fn, cost_fp, thresholds):
    """(t, cost) of the threshold of smallest total cost among thresholds (the first one if tied), CP3.7 b)."""
    proba = np.asarray(proba)
    costs = [total_cost(y_true, (proba >= t).astype(int), cost_fn, cost_fp) for t in thresholds]
    best = int(np.argmin(costs))                                   # argmin keeps the first minimum
    return float(thresholds[best]), costs[best]


t_7, cost_7 = min_cost_threshold(PROBA_VAL, Y_VAL, COST_FN, COST_FP, THRESHOLDS)
theory_7 = COST_FP / (COST_FP + COST_FN)                           # decide 1 when 5 p >= 1 - p
test_7 = [total_cost(Y_TEST, (PROBA_TEST >= t).astype(int), COST_FN, COST_FP) for t in (t_7, 0.5)]
print(f"validation: t = {t_7}, cost {cost_7} · theory {theory_7:.4f} · test: cost {test_7[0]} at t, {test_7[1]} at 0.5")'''

CP37_RECORD = r'''def best_with(rule, cost_fn=COST_FN, cost_fp=COST_FP):
    """[t, cost] on the validation with another (wrong) decision rule or other costs."""
    costs = [total_cost(Y_VAL, rule(PROBA_VAL, t).astype(int), cost_fn, cost_fp) for t in THRESHOLDS]
    best = int(np.argmin(costs))
    return [float(THRESHOLDS[best]), costs[best]]


wb.record("CP3.7c", [t_7, cost_7], decimals=4, mistakes={
    "la règle de l'énoncé décide « classe 1 » quand la probabilité est SUPÉRIEURE OU ÉGALE au seuil":
    best_with(lambda p, t: p > t),
    "les deux coûts sont échangés : un faux NÉGATIF coûte 5, un faux positif 1": best_with(lambda p, t: p >= t, 1, 5)})
wb.record("CP3.7e", test_7, decimals=4, mistakes={
    "l'ordre demandé est [coût au seuil de c), coût au seuil 0,5]": test_7[::-1],
    "ces coûts sont mesurés sur la validation : e) les demande sur le test":
    [total_cost(Y_VAL, (PROBA_VAL >= t).astype(int), COST_FN, COST_FP) for t in (t_7, 0.5)]})'''

# ---------------------------------------------------------------------------
# Part B: checks after the exam
# ---------------------------------------------------------------------------
PART_B_INTRO = r'''## Partie B · Après l'examen : vérification automatique

**Ne commence cette partie qu'une fois ta copie terminée**, examen rendu. Elle vérifie le code de la partie A, puis les réponses que tu as écrites sur ta feuille, question par question : ✅ juste, ❌ faux (avec une piste), ⏳ pas rempli. Elle ne note pas : les démarches, les justifications et les questions rédigées se corrigent avec `03_examen_corrige.md`, qui donne aussi le barème.

1. Passe `EXAM_OVER` à `True` dans la cellule ci-dessous, puis exécute-la.
2. Exécute la vérification du code (CP3.6 et CP3.7).
3. Reporte ensuite chaque réponse de ta feuille : **la valeur** que tu as écrite, pas un nouveau calcul (sinon tu ne vérifies plus ta copie). En Python, le séparateur décimal est un **point** (`0.125`), et les lettres s'écrivent entre guillemets (`"XYZ"`).

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


def as_curve(value):
    """A curve number typed as 2, "2", "(2)" or "②" -> 2 (anything else is left as typed)."""
    if isinstance(value, str):
        text = value.strip().replace("①", "1").replace("②", "2").strip("()[] .")
        if text.isdigit():
            return int(text)
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

CODE_CHECK_6 = r'''# Checks of honest_report (CP3.6)
if exam_over() and ready("CP3.6", "np", "X", "y", "K_GRID", "colleague_report", "honest_report", "_ORIGINALS",
                         "california"):
    PROPERTIES["CP3.6"] = []
    other_rows = np.sort(np.random.default_rng(2710).choice(len(california), 4000, replace=False))
    X_OTHER_6 = california[FEATURES].to_numpy(dtype=float)[other_rows]       # 4 000 other districts
    y_OTHER_6 = california["MedHouseVal"].to_numpy(dtype=float)[other_rows]
    OTHER_6, OTHER_6_ON_TEST = (5, 0.6544), 10      # the honest (best_k, R²) there, and the k that the test prefers
    RESULTS.pop("CP3.6d", None)
    restore_data("CP3.6", ["X", "y"])
    X_check, y_check = X.copy(), y.copy()
    report, ran = None, False
    try:
        report = honest_report(X_check, y_check)
        ran = True
    except NotImplementedError:
        print("⏳ CP3.6 : honest_report n'est pas encore écrite.")
    except Exception as exc:                   # a bug of the exam code must not stop part B
        verdict("CP3.6", False, "", f"honest_report(X, y) lève une erreur : {type(exc).__name__}: {exc}")
    if ran and report is None:
        verdict("CP3.6", False, "", "honest_report ne renvoie rien : as-tu oublié le return ?")
    if report is not None:
        verdict("CP3.6", np.array_equal(X_check, X) and np.array_equal(y_check, y), "X et y ne sont pas modifiés.",
                "ta fonction modifie X (ou y) : la ligne (B) écrit dans le tableau qu'elle reçoit ; travaille sur une "
                "copie (X = X.copy()) avant de transformer la colonne de la population.")
        pair = None
        try:
            pair = [float(v) for v in report]
        except (TypeError, ValueError):
            pass
        if pair is None or len(pair) != 2:
            verdict("CP3.6", False, "", f"honest_report doit renvoyer un couple (best_k, test_r2), pas "
                    f"{type(report).__name__} {report!r}"[:300])
        else:
            verdict("CP3.6", pair[0] in K_GRID, "le nombre de voisins retenu fait partie de K_GRID.",
                    f"best_k vaut {pair[0]:g} : ce doit être l'une des valeurs de K_GRID, lue dans best_params_.")
            leaky = colleague_report(X, y)[1]
            verdict("CP3.6", pair[1] < leaky - 0.05,
                    "sans la fuite de la ligne (A), le R² de test est nettement plus bas que celui du collègue.",
                    "ton R² reste près de celui du collègue : une feature calculée à partir de y reste-t-elle dans X ?")
            try:                               # other districts, where the k of the test is not the k of the folds
                other = [float(v) for v in honest_report(X_OTHER_6.copy(), y_OTHER_6.copy())]
            except Exception as exc:
                other = f"{type(exc).__name__}: {exc}"
            verdict("CP3.6", isinstance(other, list) and len(other) == 2 and other[0] == OTHER_6[0]
                    and abs(other[1] - OTHER_6[1]) < 5e-4,
                    "sur d'autres districts, k est choisi par la validation croisée, et le test ne sert qu'à la fin.",
                    f"sur 4 000 autres districts, la validation croisée choisit k = {OTHER_6[0]} (R² du test "
                    f"{OTHER_6[1]:.4f}), et ta fonction renvoie {other!r} : "
                    + ("k est choisi d'après le score du TEST (la fuite (E)), ou bien le StandardScaler est appris une "
                       "fois sur tout l'entraînement, hors du pipeline, et la validation croisée ne le réapprend pas "
                       "dans chaque fold, ou bien tes folds ne sont pas ceux de cv=5 (sans mélange) que demande "
                       "l'énoncé." if isinstance(other, list) and len(other) == 2 and other[0] == OTHER_6_ON_TEST else
                       "relis l'ordre : découper, chercher k par validation croisée sur l'entraînement, puis évaluer."))
            code_check("CP3.6d", pair, "découpe d'abord ; un Pipeline (StandardScaler, kNN) dans une GridSearchCV à "
                       "5 folds sur l'entraînement ; puis le R² du test, une seule fois.")'''

CODE_CHECK_7 = r'''# Checks of total_cost and min_cost_threshold (CP3.7)
if exam_over() and ready("CP3.7", "np", "PROBA_VAL", "Y_VAL", "PROBA_TEST", "Y_TEST", "THRESHOLDS", "total_cost",
                         "min_cost_threshold"):
    PROPERTIES["CP3.7"] = []
    for sub in ("CP3.7c", "CP3.7e"):
        RESULTS.pop(sub, None)
    restore_data("CP3.7", ["PROBA_VAL", "Y_VAL", "PROBA_TEST", "Y_TEST"])

    def safe(function, *args):
        """function(*args), or the text of the error it raises (a bug of the exam code must not stop part B)."""
        try:
            return function(*args)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def as_pair(value):
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            return value

    OTHER_COSTS_7 = [0.75, 120.0]                   # (t, cost) on the validation with cost_fn = 1, cost_fp = 5
    labels_small, decisions_small = np.array([1, 1, 0, 0, 1]), np.array([1, 0, 1, 0, 0])
    costs_ok = False
    try:
        labels_arg, decisions_arg = labels_small.copy(), decisions_small.copy()
        small = total_cost(labels_arg, decisions_arg, 5, 1)
    except NotImplementedError:
        print("⏳ CP3.7 a) : total_cost n'est pas encore écrite.")
    except Exception as exc:
        verdict("CP3.7", False, "", f"total_cost lève une erreur : {type(exc).__name__}: {exc}")
    else:
        if small is None:
            verdict("CP3.7", False, "", "total_cost ne renvoie rien : as-tu oublié le return ?")
        else:
            verdict("CP3.7", isinstance(small, int) and not isinstance(small, bool),
                    "total_cost renvoie un int.",
                    f"total_cost doit renvoyer un int Python (ici : {type(small).__name__}) : return int(...).")
            verdict("CP3.7", np.array_equal(labels_arg, labels_small) and np.array_equal(decisions_arg, decisions_small),
                    "total_cost ne modifie pas ses arguments.",
                    "total_cost modifie y_true ou y_pred (par exemple avec -=) : elle doit seulement les lire.")
            swapped = safe(total_cost, labels_small.copy(), decisions_small.copy(), 1, 5)
            costs_ok = verdict("CP3.7", small == 11 and swapped == 7,
                               "total_cost compte cost_fn par faux négatif et cost_fp par faux positif.",
                               "sur labels [1, 1, 0, 0, 1] et décisions [1, 0, 1, 0, 0] (deux faux négatifs, un faux "
                               f"positif), le coût vaut 2 × 5 + 1 × 1 = 11 avec les coûts (5, 1), et 2 × 1 + 1 × 5 = 7 "
                               f"avec (1, 5) ; ta fonction renvoie {small!r} et {swapped!r}.")
    try:
        proba_copy, labels_copy = PROBA_VAL.copy(), Y_VAL.copy()
        chosen = min_cost_threshold(proba_copy, labels_copy, 5, 1, THRESHOLDS)
    except NotImplementedError:
        print("⏳ CP3.7 b) : min_cost_threshold n'est pas encore écrite.")
    except Exception as exc:
        verdict("CP3.7", False, "", f"min_cost_threshold lève une erreur : {type(exc).__name__}: {exc}")
    else:
        pair = None
        try:
            pair = [float(v) for v in chosen]
        except (TypeError, ValueError):
            pass
        if chosen is None:
            verdict("CP3.7", False, "", "min_cost_threshold ne renvoie rien : as-tu oublié le return ?")
        elif pair is None or len(pair) != 2:
            verdict("CP3.7", False, "", f"min_cost_threshold doit renvoyer un couple (t, coût), pas {chosen!r}"[:300])
        else:
            verdict("CP3.7", np.array_equal(proba_copy, PROBA_VAL) and np.array_equal(labels_copy, Y_VAL),
                    "min_cost_threshold ne modifie pas ses arguments.",
                    "min_cost_threshold modifie proba ou y_true : elle doit seulement les lire.")
            tie = as_pair(safe(min_cost_threshold, np.array([0.2, 0.4, 0.6, 0.8]), np.array([0, 1, 0, 1]), 1, 1,
                               np.array([0.7, 0.5, 0.3])))
            verdict("CP3.7", tie == [0.7, 1.0],
                    "à égalité de coût, le premier seuil de thresholds l'emporte, dans l'ordre reçu.",
                    f"sur proba [0,2 ; 0,4 ; 0,6 ; 0,8], labels [0, 1, 0, 1] et seuils [0,7 ; 0,5 ; 0,3] (dans cet "
                    f"ordre), les trois seuils coûtent 1, 2 et 1 : on attend (0.7, 1), le premier minimum dans l'ordre "
                    f"de thresholds ; ta fonction renvoie {tie!r}.")
            edge = as_pair(safe(min_cost_threshold, np.array([0.5, 0.5]), np.array([1, 1]), 5, 1, np.array([0.5, 0.6])))
            verdict("CP3.7", edge == [0.5, 0.0],
                    "une probabilité égale au seuil donne la classe 1 (≥).",
                    f"sur proba [0,5 ; 0,5], labels [1, 1] et seuils [0,5 ; 0,6], on attend (0.5, 0) : une probabilité "
                    f"égale au seuil donne la classe 1 (proba >= t) ; ta fonction renvoie {edge!r}.")
            other_costs = as_pair(safe(min_cost_threshold, PROBA_VAL.copy(), Y_VAL.copy(), 1, 5, THRESHOLDS))
            verdict("CP3.7", other_costs == OTHER_COSTS_7,
                    "avec d'autres coûts (1, 5), le seuil change comme il doit.",
                    f"avec les coûts cost_fn = 1 et cost_fp = 5 sur la validation, on attend {tuple(OTHER_COSTS_7)} ; "
                    f"ta fonction renvoie {other_costs!r} : utilise-t-elle bien cost_fn et cost_fp ?")
            code_check("CP3.7c", pair, "classe 1 quand proba >= t ; le coût compte 5 par faux négatif et 1 par faux "
                       "positif ; le premier seuil de plus petit coût.")
            if costs_ok:
                test_costs = [safe(total_cost, Y_TEST.copy(), (PROBA_TEST >= t).astype(int), 5, 1) for t in (pair[0], 0.5)]
                code_check("CP3.7e", test_costs, "les coûts du test, avec le seuil de c), puis avec 0,5.")'''

PAPER_CONTEXT = r'''# The paper questions only use the numbers of their statements (01_examen_sujet.md)
import math

import numpy as np

# CP3.2
BILLS_2 = np.array([38, 46, 42, 50], dtype=float)
MU_2, SIGMA_2 = BILLS_2.mean(), BILLS_2.std()                      # ddof = 0, like StandardScaler
SIGMA1_2 = BILLS_2.std(ddof=1)
LOW_2, HIGH_2 = BILLS_2.min(), BILLS_2.max()

# CP3.3
X_3 = np.array([1, 2, 3, 5, 6, 8], dtype=float)
CLASSES_3 = np.array(list("AABBAB"))


def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(1 - (p ** 2).sum())


def entropy_bits(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


THRESHOLDS_3 = (X_3[:-1] + X_3[1:]) / 2


def weighted_gini(s, weighted=True):
    left, right = CLASSES_3[X_3 <= s], CLASSES_3[X_3 > s]
    if not weighted:
        return (gini(left) + gini(right)) / 2
    return (left.size * gini(left) + right.size * gini(right)) / X_3.size


BEST_3 = float(THRESHOLDS_3[np.argmin([weighted_gini(s) for s in THRESHOLDS_3])])

# CP3.4
TRAIN_4 = np.array([[4000, 42], [3900, 48], [4300, 39], [4400, 46], [4450, 50]], dtype=float)
SPECIES_4 = np.array(["Adélie", "Chinstrap", "Adélie", "Chinstrap", "Chinstrap"])
Q_4 = np.array([3800, 46], dtype=float)
MEAN_4, STD_4 = np.array([4210, 45], dtype=float), np.array([220, 4], dtype=float)


def neighbours_4(standardize):
    A, q = (TRAIN_4 - MEAN_4) / STD_4 if standardize else TRAIN_4, (Q_4 - MEAN_4) / STD_4 if standardize else Q_4
    order = np.argsort(np.sqrt(((A - q) ** 2).sum(axis=1)))[:3]
    return (order + 1).tolist(), max(set(SPECIES_4[order]), key=list(SPECIES_4[order]).count)


# CP3.5
EPS_5 = 3 / 10
ALPHA_5 = math.log((1 - EPS_5) / EPS_5)
WRONG_5, RIGHT_5 = 0.1 * math.exp(ALPHA_5), 0.1
TOTAL_5 = 3 * WRONG_5 + 7 * RIGHT_5
X_5, Y_5 = np.array([1, 2, 3, 4], dtype=float), np.array([2, 4, 9, 11], dtype=float)
F0_5 = float(Y_5.mean())
R_5 = Y_5 - F0_5


def stump_5(r):
    """Regression stump on x_5: [threshold, left value, right value] of smallest sum of squares."""
    best = None
    for s in (X_5[:-1] + X_5[1:]) / 2:
        left, right = r[X_5 <= s], r[X_5 > s]
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, [float(s), float(left.mean()), float(right.mean())])
    return best[1]


STUMP_5 = stump_5(R_5)


def f1_5(eta):
    return F0_5 + eta * np.where(X_5 <= STUMP_5[0], STUMP_5[1], STUMP_5[2])


# CP3.11
TP_11, FP_11 = 45, 60 - 45
FN_11, TN_11 = 75 - TP_11, 500 - 75 - FP_11
P_11, R_11 = TP_11 / (TP_11 + FP_11), TP_11 / (TP_11 + FN_11)

# CP3.13


def var_mean(sigma2, rho, m):
    return rho * sigma2 + (1 - rho) * sigma2 / m


# CP3.14
CANDIDATES_14 = 4 * 3 * 5
FITS_14 = CANDIDATES_14 * 5 + 1'''


def _letters_mistakes(answer: str, messages: dict) -> str:
    """mistakes={message: letter} for every other choice of a lettered question."""
    pairs = ",\n                     ".join(f'"{message}": "{letter}"' for letter, message in messages.items()
                                            if letter != answer)
    return "mistakes={" + pairs + "}"


PAPER = [
    Paper("CP3.1", "Prédire la sortie de six extraits scikit-learn (une lettre par extrait)", [
        ("a", "a letter A, B, C or D", '"A"', _letters_mistakes("A", {
            "B": "un classifieur binaire n'a qu'UN score par exemple : decision_function renvoie un tableau à une dimension",
            "C": "relis la forme : à deux classes, decision_function renvoie un tableau à UNE dimension, pas une colonne",
            "D": "la forme d'une sortie a d'abord le nombre d'exemples ; relis ce que rend decision_function à deux classes"})),
        ("b", "a letter A, B, C or D", '"C"', _letters_mistakes("C", {
            "A": "predict_proba donne une distribution de probabilité par exemple : que vaut la somme de chaque ligne ?",
            "B": "chaque ligne de predict_proba contient les probabilités des trois classes : quelle est leur somme ?",
            "D": "une forêt fait la moyenne des probabilités de ses arbres : elle a bien un predict_proba"})),
        ("c", "a letter A, B, C or D", '"B"', _letters_mistakes("B", {
            "A": "une ligne de zéros, c'est le réglage handle_unknown « ignore » : quel est le réglage PAR DÉFAUT de OneHotEncoder ?",
            "C": "un encodeur appris ne crée jamais de colonne après fit : relis le réglage par défaut de handle_unknown",
            "D": "relis le réglage par défaut de handle_unknown de OneHotEncoder"})),
        ("d", "a letter A, B, C or D", '"D"', _letters_mistakes("D", {
            "A": "la moyenne et l'écart-type viennent de X_train : sur des données nouvelles, la moyenne n'est pas exactement 0",
            "B": "relis ce que transform fait des moyennes : chaque colonne est centrée sur la moyenne d'ENTRAÎNEMENT",
            "C": "transform s'applique ligne par ligne : il n'a pas besoin du même nombre de lignes qu'à l'entraînement"})),
        ("e", "a letter A, B, C or D", '"A"', _letters_mistakes("A", {
            "B": "le pipeline a deux étapes : la grille doit dire à quelle étape appartient le paramètre",
            "C": "compte les tirets bas entre le nom de l'étape et le paramètre",
            "D": "le nom d'une étape est celui que tu lui as donné dans le Pipeline"})),
        ("f", "a letter A, B, C or D", '"D"', _letters_mistakes("D", {
            "A": "quand toutes les entrées sont des listes, la recherche au hasard tire-t-elle avec ou sans remise ?",
            "B": "scikit-learn ne s'arrête pas : relis ce qu'il fait quand n_iter dépasse la taille de la grille",
            "C": "une liste ne se tire pas comme une loi continue : RandomizedSearchCV choisit parmi les valeurs données"})),
    ]),
    Paper("CP3.2", "Standardiser et encoder un mini-tableau", [
        ("a", "[mean, std], 4 decimals", "[MU_2, SIGMA_2]",
         r'''decimals=4, mistakes={"StandardScaler divise par n, pas par n − 1 : relis l'énoncé": [MU_2, SIGMA1_2],
                     "c'est la variance : l'écart-type en est la racine": [MU_2, SIGMA_2 ** 2]}'''),
        ("b", "z-score of penguin 2, 3 decimals", "(46 - MU_2) / SIGMA_2",
         r'''decimals=3, mistakes={"StandardScaler divise par l'écart-type calculé avec n, pas n − 1": (46 - MU_2) / SIGMA1_2}'''),
        ("c", "z-score of T1, 3 decimals", "(54 - MU_2) / SIGMA_2",
         r'''decimals=3, mistakes={"StandardScaler divise par l'écart-type calculé avec n, pas n − 1": (54 - MU_2) / SIGMA1_2,
                     "le scaler a appris sur les quatre manchots d'entraînement seulement : T1 n'entre pas dans la moyenne": (54 - np.append(BILLS_2, 54).mean()) / np.append(BILLS_2, 54).std()}'''),
        ("d", "min-max of T1, 3 decimals", "(54 - LOW_2) / (HIGH_2 - LOW_2)",
         r'''decimals=3, mistakes={"MinMaxScaler ne borne pas les données nouvelles par défaut (clip=False)": 1.0,
                     "le min-max divise par l'étendue (max − min) de l'entraînement": (54 - LOW_2) / HIGH_2}'''),
        ("e", "one-hot of penguin 3 (a list)", "[0, 1, 0]",
         r'''mistakes={"les catégories sont rangées dans l'ordre alphabétique, pas dans l'ordre d'apparition": [1, 0, 0]}'''),
        ("f", "one-hot of T2 (a list)", "[0, 0, 0]", ""),
        ("g", "row of T2 from the ColumnTransformer (a list, 3 decimals)", "[(41 - MU_2) / SIGMA_2, 0, 0, 0]",
         r'''decimals=3, mistakes={"StandardScaler divise par l'écart-type calculé avec n, pas n − 1": [(41 - MU_2) / SIGMA1_2, 0, 0, 0],
                     "relis l'ordre des colonnes : d'abord le z-score du bec, puis le one-hot": [0, 0, 0, (41 - MU_2) / SIGMA_2]}'''),
    ]),
    Paper("CP3.3", "Gini et meilleur seuil", [
        ("a", "Gini of the root", "gini(CLASSES_3)",
         r'''decimals=3, mistakes={"c'est l'entropie de la racine, en bits : on demande son Gini, 1 − Σ p²": 1.0,
                     "le Gini est 1 − Σ p², la somme portant sur les deux classes": 0.25}'''),
        ("b", "number of candidate thresholds", "THRESHOLDS_3.size",
         r'''fractional="on demande un nombre de seuils", mistakes={"un seuil se place ENTRE deux abscisses consécutives": 6}'''),
        ("c", "weighted Gini of s = 4, 3 decimals", "weighted_gini(4.0)", "decimals=3"),
        ("d", "the best threshold", "BEST_3",
         r'''decimals=1, choices=THRESHOLDS_3.tolist(), mistakes={f"s = {s:g} : calcule son Gini pondéré et compare-le à celui des autres seuils".replace(".", ","): float(s)
                     for s in THRESHOLDS_3 if s != BEST_3}'''),
        ("e", "[weighted Gini, gain], 3 decimals", "[weighted_gini(BEST_3), gini(CLASSES_3) - weighted_gini(BEST_3)]",
         r'''decimals=3, mistakes={"le Gini pondéré pèse chaque enfant par son NOMBRE d'exemples, pas une moyenne simple": [weighted_gini(BEST_3, weighted=False), gini(CLASSES_3) - weighted_gini(BEST_3, weighted=False)]}'''),
        ("f", "entropy of the right child in bits, 3 decimals", "entropy_bits(CLASSES_3[X_3 > BEST_3])",
         r'''decimals=3, mistakes={"l'entropie demandée est en bits : log₂, pas ln": entropy_bits(CLASSES_3[X_3 > BEST_3]) * math.log(2),
                     "c'est le Gini de cet enfant : on demande son entropie, −Σ p log₂ p": gini(CLASSES_3[X_3 > BEST_3])}'''),
    ]),
    Paper("CP3.4", "kNN avant et après standardisation", [
        ("a", "the 3 nearest neighbours on the raw measures (a list)", "neighbours_4(False)[0]",
         r'''mistakes={"la liste va du plus proche au moins proche": neighbours_4(False)[0][::-1]}'''),
        ("b", "the species predicted on the raw measures", "neighbours_4(False)[1]",
         r'''mistakes={"relis le vote : quelle espèce a la majorité parmi les trois voisins de a) ?": "Chinstrap"}'''),
        ("c", "[z_mass, z_bill] of Q, 3 decimals", "((Q_4 - MEAN_4) / STD_4).tolist()",
         r'''decimals=3, mistakes={"un z-score retire la moyenne PUIS divise par l'écart-type : (x − μ)/σ": ((MEAN_4 - Q_4) / STD_4).tolist()}'''),
        ("d", "the 3 nearest neighbours after standardization (a list)", "neighbours_4(True)[0]",
         r'''mistakes={"la liste va du plus proche au moins proche": neighbours_4(True)[0][::-1],
                     "ce sont encore les voisins des mesures brutes : standardise aussi les cinq manchots d'entraînement, avec les mêmes statistiques": neighbours_4(False)[0]}'''),
        ("e", "the species predicted after standardization", "neighbours_4(True)[1]",
         r'''mistakes={"relis le vote : quelle espèce a la majorité parmi les trois voisins de d) ?": "Adélie"}'''),
    ]),
    Paper("CP3.5", "AdaBoost et gradient boosting", [
        ("a", "weighted error", "EPS_5", "decimals=2"),
        ("b", "alpha, 3 decimals", "ALPHA_5",
         r'''decimals=3, mistakes={"la fiche du ch. 14 (et scikit-learn) n'a pas le facteur ½ des textes à labels ±1 : α = ln((1 − ε)/ε)": ALPHA_5 / 2,
                     "ln est le logarithme népérien, pas log₁₀": math.log10((1 - EPS_5) / EPS_5),
                     "le rapport est (1 − ε)/ε, pas ε/(1 − ε)": -ALPHA_5}'''),
        ("c", "normalized weight of a misclassified example, 4 decimals", "WRONG_5 / TOTAL_5",
         r'''decimals=4, mistakes={"n'oublie pas la normalisation : divise par la somme des nouveaux poids": WRONG_5,
                     "seuls les exemples MAL classés sont multipliés par e^α": 0.1 * math.exp(-ALPHA_5) / (3 * 0.1 * math.exp(-ALPHA_5) + 0.7)}'''),
        ("d", "normalized weight of a well-classified example, 4 decimals", "RIGHT_5 / TOTAL_5",
         r'''decimals=4, mistakes={"n'oublie pas la normalisation : divise par la somme des nouveaux poids": RIGHT_5}'''),
        ("e", "alpha with K = 4 classes, 3 decimals", "ALPHA_5 + math.log(4 - 1)",
         r'''decimals=3, mistakes={"le terme de SAMME est ln(K − 1), pas ln(K)": ALPHA_5 + math.log(4),
                     "avec K = 4, ln(K − 1) = ln 3": ALPHA_5 + math.log(2)}'''),
        ("f", "F0", "F0_5", "decimals=2"),
        ("g", "residuals (a list)", "R_5.tolist()",
         r'''decimals=2, mistakes={"un résidu est y − F₀, pas F₀ − y": (-R_5).tolist()}'''),
        ("h", "[s, left value, right value]", "STUMP_5",
         r'''decimals=2, mistakes={"les feuilles prédisent la moyenne des RÉSIDUS, pas celle des y": [STUMP_5[0], float(Y_5[X_5 <= STUMP_5[0]].mean()), float(Y_5[X_5 > STUMP_5[0]].mean())]}'''),
        ("i", "F1(1), 2 decimals", "float(f1_5(0.1)[0])",
         r'''decimals=2, mistakes={"le learning rate multiplie la valeur de la feuille : F₁ = F₀ + η × feuille": float(f1_5(1.0)[0]),
                     "F₁ part de F₀, pas de 0 : F₁ = F₀ + η × feuille": float(0.1 * STUMP_5[1])}'''),
        ("j", "training MSE after the step, 4 decimals", "float(np.mean((Y_5 - f1_5(0.1)) ** 2))",
         r'''decimals=4, mistakes={"c'est la MSE de départ, avec F₀ : on la demande après l'étape": float(np.mean(R_5 ** 2)),
                     "c'est la MSE avec η = 1 : l'énoncé prend η = 0,1": float(np.mean((Y_5 - f1_5(1.0)) ** 2)),
                     "c'est une RMSE : on demande la MSE, sans racine": float(np.sqrt(np.mean((Y_5 - f1_5(0.1)) ** 2)))}'''),
    ]),
    Paper("CP3.8", "Lire les courbes de la figure", [
        ("a", "the number of the training curve (1 or 2)", "2",
         r'''choices=[1, 2], mistakes={"la courbe d'entraînement est celle qu'un arbre plus profond fait toujours baisser, jusqu'à apprendre par cœur": 1}'''),
        ("b", "a letter A, B, C or D", '"B"', _letters_mistakes("B", {
            "A": "cherche le point le plus bas de la courbe de VALIDATION, pas celui où les deux courbes sont proches",
            "C": "relis la courbe de validation : où atteint-elle son minimum ?",
            "D": "l'erreur d'entraînement baisse toujours : la profondeur se choisit sur la courbe de validation"})),
        ("c", "a letter A, B or C", '"A"', _letters_mistakes("A", {
            "B": "à max_depth=2, les deux erreurs sont hautes et proches : est-ce de l'overfitting ?",
            "C": "à max_depth=2, l'erreur de validation est encore loin de son minimum"})),
        ("d", "a letter A, B or C", '"B"', _letters_mistakes("B", {
            "A": "à max_depth=18, l'erreur d'entraînement est presque nulle : est-ce de l'underfitting ?",
            "C": "à max_depth=18, l'écart entre les deux courbes est grand et la validation remonte"})),
        ("e", "the learning rate (1 or 0.1)", "1.0",
         r'''decimals=1, choices=[1.0, 0.1], mistakes={"avec le learning rate 0,1, l'erreur de validation ne remonte pas (elle baisse de plus en plus lentement jusqu'à 400 arbres) : laquelle remonte ?": 0.1}'''),
        ("f", "a letter A, B, C or D", '"B"', _letters_mistakes("B", {
            "A": "regarde où la courbe de validation de ce learning rate atteint son point le plus bas",
            "C": "après son minimum, la courbe de validation remonte : où est ce minimum ?",
            "D": "la courbe de validation de ce learning rate remonte bien avant la fin"})),
        ("g", 'True / False (ou "vrai" / "faux")', "True", ""),
    ]),
    Paper("CP3.11", "Parties I et II", [
        ("a", "[TP, FP, FN, TN]", "[TP_11, FP_11, FN_11, TN_11]",
         r'''mistakes={"FP : les transactions signalées qui ne sont pas des fraudes ; FN : les fraudes non signalées": [TP_11, FN_11, FP_11, TN_11],
                     "les 500 transactions se répartissent dans les quatre cases : TN = 500 − TP − FP − FN": [TP_11, FP_11, FN_11, 500 - 75]}'''),
        ("b", "precision, 3 decimals", "P_11",
         r'''decimals=3, mistakes={"precision = TP / (TP + FP) : parmi les transactions SIGNALÉES": R_11}'''),
        ("c", "recall, 3 decimals", "R_11",
         r'''decimals=3, mistakes={"recall = TP / (TP + FN) : parmi les VRAIES fraudes": P_11}'''),
        ("d", "F1, 3 decimals", "2 * P_11 * R_11 / (P_11 + R_11)",
         r'''decimals=3, mistakes={"le F1 est la moyenne HARMONIQUE de la precision et du recall": (P_11 + R_11) / 2}'''),
        ("e", "accuracy, 3 decimals", "(TP_11 + TN_11) / 500",
         r'''decimals=3, mistakes={"l'accuracy compte les bonnes réponses, positives ET négatives, parmi les 500": TP_11 / 500}'''),
        ("f", "[validation fold size, class 1 in it]", "[48, 12]",
         r'''mistakes={"un fold de validation contient 1/5 des exemples, et la stratification y met 1/5 de chaque classe": [48, 60],
                     "relis l'ordre demandé : [taille, classe 1]": [12, 48]}'''),
        ("g", "[training size, class 1 in it]", "[192, 48]",
         r'''mistakes={"l'entraînement d'un tour, ce sont les QUATRE autres folds": [48, 12],
                     "relis l'ordre demandé : [taille, classe 1]": [48, 192]}'''),
        ("h", "number of models", "5 * 3",
         r'''mistakes={"chaque répétition refait les 5 folds : multiplie": 5 + 3,
                     "cross_val_score ne réentraîne pas de modèle final : il ne compte que les tours": 16}'''),
    ]),
    Paper("CP3.12", "Vrai ou faux justifiés (le verdict seulement ; les justifications se notent avec le corrigé)", [
        ("a", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("b", 'True / False (ou "vrai" / "faux")', "True", ""),
        ("c", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("d", 'True / False (ou "vrai" / "faux")', "True", ""),
        ("e", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("f", 'True / False (ou "vrai" / "faux")', "True", ""),
    ]),
    Paper("CP3.13", "Variance d'une moyenne de modèles corrélés", [
        ("b", "variance for M = 10, 3 decimals", "var_mean(4, 0.25, 10)",
         r'''decimals=3, mistakes={"c'est la variance de modèles indépendants, σ²/M : la corrélation ρ ajoute ρ σ²": 4 / 10,
                     "relis la formule : le second terme est (1 − ρ) σ² / M": 0.25 * 4 + 4 / 10}'''),
        ("c", "the limit for M -> infinity", "0.25 * 4",
         r'''decimals=3, mistakes={"c'est le terme qui disparaît : quand M grandit, (1 − ρ) σ²/M tend vers 0, et c'est ρ σ² qui reste": 0.75 * 4,
                     "la variance ne tend pas vers 0 : seul le second terme disparaît, le premier, ρ σ², reste": 0.0}'''),
        ("d", "the smallest M", "min(m for m in range(1, 1000) if var_mean(4, 0.25, m) <= 1.08)",
         r'''fractional="on demande un nombre d'arbres", mistakes={"calcule la variance pour ce M : elle dépasse encore 1,08 (M doit être un entier, arrondi vers le haut)": 37,
                     "le second terme est (1 − ρ) σ² / M, pas σ² / M": min(m for m in range(1, 1000) if 1 + 4 / m <= 1.08)}'''),
        ("e", "[limit, variance for M = 30], 3 decimals", "[0.1 * 5, var_mean(5, 0.1, 30)]",
         r'''decimals=3, mistakes={"relis l'ordre demandé : [limite, variance pour M = 30]": [var_mean(5, 0.1, 30), 0.1 * 5]}'''),
    ]),
    Paper("CP3.14", "Coût d'une recherche en grille", [
        ("a", "number of trainings", "FITS_14",
         r'''mistakes={"n'oublie pas le réentraînement final du meilleur candidat (refit=True)": FITS_14 - 1,
                     "relis le décompte : une grille essaie le PRODUIT des nombres de valeurs, et chaque candidat est entraîné une fois PAR FOLD": CANDIDATES_14 + 1,
                     "c'est le nombre de candidats : chacun est entraîné une fois par fold, plus le réentraînement final": CANDIDATES_14}'''),
        ("b", "duration in minutes", "round(FITS_14 * 2 / 60)",
         r'''fractional="arrondis à l'entier le plus proche", mistakes={"c'est une durée en secondes : convertis-la en minutes": FITS_14 * 2}'''),
        ("c", "number of trainings of the nested cross-validation", "5 * FITS_14",
         r'''mistakes={"chaque tour extérieur refait toute la recherche, réentraînement final compris": 5 * (FITS_14 - 1),
                     "l'énoncé ne compte pas la recherche finale qui produirait le modèle livré": 6 * FITS_14}'''),
        ("d", "number of trainings of the random search", "12 * 5 + 1",
         r'''mistakes={"n'oublie pas le réentraînement final": 12 * 5,
                     "chaque candidat est entraîné une fois PAR FOLD": 12 + 1}'''),
    ]),
]

SUMMARY = r'''# Summary of the automatic checks
import re


def natural(sub_id):
    """CP3.2a before CP3.13a: the question number as a number, then the letter."""
    number, letter = re.match(r"CP3\.(\d+)(.*)", sub_id).groups()
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
    print(f"Propriétés du code (CP3.6, CP3.7) : {sum(checked)} sur {len(checked)}.")
    print("Note ensuite ta copie avec le barème de 03_examen_corrige.md : il donne aussi les points des démarches, "
          "des justifications et des questions rédigées, que cette vérification ne voit pas.")'''


WRAP = {f"CP3.1{letter}": "as_letters" for letter in "abcdef"}
WRAP.update({f"CP3.8{letter}": "as_letters" for letter in "bcdf"})
WRAP["CP3.8a"] = "as_curve"


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


def _three_letters_6() -> str:
    """mistakes entries for every wrong set of three letters of CP3.6 a): how many of them are leaks."""
    from itertools import combinations
    words = {0: "aucune n'est une fuite", 1: "une seule est une fuite", 2: "deux seulement sont des fuites"}
    lines = []
    for combo in combinations("ABCDEF", 3):
        letters = "".join(combo)
        if letters == "ACE":
            continue
        n = len(set(letters) & set("ACE"))
        lines.append(f'"parmi les lignes {combo[0]}, {combo[1]} et {combo[2]}, {words[n]} : demande-toi pour chaque '
                     f'ligne si elle apprend quelque chose de la cible ou des districts de test": "{letters}"')
    return ",\n                     ".join(lines)


CODE_PAPER = [
    Paper("CP3.6", "Les lettres des trois fuites (a)", [
        ("a", 'three letters, for example "XYZ"', '"ACE"',
         r'''mistakes={"il manque une fuite : la ligne (E) choisit k d'après son score sur le TEST": "AC",
                     "il manque une fuite : la ligne (C) apprend ses moyennes et ses écarts-types sur TOUS les districts, test compris": "AE",
                     "il manque une fuite : la ligne (A) construit une feature à partir de y, la cible": "CE",
                     "une ligne de trop : (B) se calcule ligne par ligne, sans rien apprendre des autres districts": "ABCE",
                     "une ligne de trop : (D) découpe ; c'est ce qu'elle reçoit qui pose problème, pas le découpage": "ACDE",
                     "une ligne de trop : (F) rend ce que (E) a calculé ; la fuite est dans le choix de k": "ACEF",
                     ''' + _three_letters_6() + '''}'''),
    ]),
    Paper("CP3.7", "Le seuil théorique (d)", [
        ("d", "the theoretical threshold, 4 decimals", "COST_FP / (COST_FP + COST_FN)",
         r'''decimals=4, mistakes={"les deux coûts sont échangés : décider « 1 » ne coûte que si l'exemple est négatif, décider « 0 » que s'il est positif": COST_FN / (COST_FP + COST_FN),
                     "le rapport des coûts n'est pas le seuil : écris le coût moyen de chaque décision pour une probabilité p, puis cherche où ils s'égalent": COST_FP / COST_FN}'''),
    ]),
]
WRAP["CP3.6a"] = "as_letters"


def header_cells(kind: str) -> list:
    rel = EXAM if kind == "exercise" else SOLUTIONS
    if kind == "exercise":
        title = "# Checkpoint III · Examen blanc — notebook de l'examen"
        how = ("Ce notebook accompagne le sujet `01_examen_sujet.md`.\n\n"
               "- **Partie A, pendant l'examen** : les deux questions de code, CP3.6 (🐛, 2 points, avec "
               "scikit-learn) et CP3.7 (🔨, 2 points, avec NumPy seulement). Écris ton code dans les cellules à "
               "compléter, sans `mylearn`, sans les corrigés et sans assistant IA ; `help()` et la documentation "
               "officielle sont permis. Rien n'est vérifié pendant l'examen : les cellules « essai » montrent seulement "
               "ce que fait ton code. Tes réponses chiffrées et tes explications vont sur ta feuille "
               "(`04_mes_reponses.md`).\n"
               "- **Partie B, après l'examen** : la vérification automatique de ton code et des réponses de ta "
               "feuille. Ses cellules ne vérifient rien tant que `EXAM_OVER` vaut `False`.\n\n"
               "« Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche ⏳ "
               "(partie A) ou ⏸️ (partie B).\n\n"
               "> Travaille dans **ta copie** (`mon_travail/checkpoints/partie_3/02_examen_notebook.ipynb`, créée "
               "par `python tools/start_chapter.py CP3`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# Checkpoint III · Examen blanc — solutions (notebook exécuté)"
        how = ("Les solutions des deux questions de code (CP3.6 et CP3.7), exécutées, puis l'enregistrement des "
               "réponses que vérifie la partie B du notebook de l'examen (cellules marquées `answer`, récoltées "
               "par `tools/build_answers.py`). Le corrigé détaillé et le barème sont dans `03_examen_corrige.md`.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell("demo"))
    cells.append(md("## Partie A · Pendant l'examen : les deux questions de code" if kind == "exercise"
                    else "## Partie A · Les deux questions de code"))
    cells.append(code(GIVEN))
    cells.append(md(CP36_MD))
    cells.append(code(CP36_BUG))
    if kind == "exercise":
        cells += [code(CP36_TODO), code(CP36_TRY)]
    else:
        cells += [code(CP36_SOLUTION), code(CP36_RECORD, tags=["answer"])]
    cells.append(md(CP37_MD))
    if kind == "exercise":
        cells += [code(CP37_TODO), code(CP37_TRY)]
        cells.append(md("---\n\n### 🛑 Fin de la partie A\n\nL'examen s'arrête ici. Rends ta copie (enregistre ce "
                        "notebook et ta feuille), puis, seulement après, passe à la partie B."))
        cells.append(md(PART_B_INTRO))
        cells.append(code(PART_B_FLAG))
        cells.append(md("### Le code de la partie A"))
        cells.append(code(CODE_CHECK_6))
        cells.append(code(CODE_CHECK_7))
        cells.append(md("### Les réponses de ta feuille"))
        for paper in PAPER[:5]:
            cells += paper_cell(kind, paper)
        for paper in CODE_PAPER:
            cells += paper_cell(kind, paper)
        for paper in PAPER[5:]:
            cells += paper_cell(kind, paper)
        cells.append(md("### Bilan"))
        cells.append(code(SUMMARY))
        cells.append(md("**Ce que cette vérification ne voit pas** : les démarches et les justifications ; les "
                        "explications et les démonstrations (CP3.2 h, CP3.4 f, CP3.5 k, CP3.6 b et e, CP3.7 d (la "
                        "démonstration) et f, CP3.9, CP3.10, CP3.12 (les justifications), CP3.13 a et f) ; et, pour "
                        "CP3.6 et CP3.7, les nombres recopiés sur ta feuille (seules tes fonctions sont vérifiées). "
                        "Tout cela se corrige avec `03_examen_corrige.md`. Ensuite : la remédiation du corrigé (relis la "
                        "synthèse, `05_synthese.md`, pour les notions ratées), puis le mini-projet."))
    else:
        cells += [code(CP37_SOLUTION), code(CP37_RECORD, tags=["answer"])]
        cells.append(md("## Partie B · Les réponses vérifiées après l'examen\n\n"
                        "Chaque réponse est calculée à partir des nombres du sujet ; les erreurs classiques reçoivent "
                        "un message qui dit où chercher."))
        cells.append(code(PAPER_CONTEXT))
        for paper in PAPER[:5]:
            cells += paper_cell(kind, paper)
        for paper in CODE_PAPER:
            cells += paper_cell(kind, paper)
        for paper in PAPER[5:]:
            cells += paper_cell(kind, paper)
    return cells


def main() -> int:
    write_notebook(EXAM, build("exercise"))
    write_notebook(SOLUTIONS, build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER + CODE_PAPER)
    print(f"✅ {EXAM} and {SOLUTIONS} written ({len(PAPER) + len(CODE_PAPER)} paper questions, {n_checks} paper "
          "checks, 2 code questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
