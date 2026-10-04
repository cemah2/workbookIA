#!/usr/bin/env python
"""Build the two notebooks of the mock exam of part I from a single source (used by Claude).

    python tools/chapters/build_cp1.py
    python tools/run_all_notebooks.py checkpoints/partie_1/03_examen_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py checkpoints/partie_1/02_examen_notebook.ipynb

The exam notebook has two parts. Part A is done DURING the exam: the two code
questions CP1.3 (🐛) and CP1.5 (🔨), without mylearn and without any check. Part B
is done AFTER the exam: it checks the code of part A, then the answers written on
paper (sub-IDs CP1.1a, CP1.2a, ...). Its cells only run once ``EXAM_OVER = True``.
The solutions notebook records every checked answer with ``wb.record``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import Paper, badge, code, md, setup_cell, write_notebook  # noqa: E402

FOLDER = "checkpoints/partie_1"
EXAM = f"{FOLDER}/02_examen_notebook.ipynb"
SOLUTIONS = f"{FOLDER}/03_examen_solutions.ipynb"

# ---------------------------------------------------------------------------
# Data shared by both notebooks
# ---------------------------------------------------------------------------
GIVEN = r'''# Data of the two code questions (run this cell first)
import numpy as np

penguins = wb.datasets.load_penguins(dropna=True)          # 333 penguins without missing values
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X = penguins[MEASURES].to_numpy(dtype=float)               # shape (333, 4): one row per penguin
gentoo_mass = penguins.loc[penguins["species"] == "Gentoo", "body_mass_g"].to_numpy(dtype=float)
print("X:", X.shape, "· gentoo_mass:", gentoo_mass.shape)'''

# ---------------------------------------------------------------------------
# Part A: the two code questions of the exam
# ---------------------------------------------------------------------------
CP13_MD = r'''### CP1.3 — Partie 0 : la moyenne qui oublie `axis` 🐛 ★ ⏱️ 5 min · 1 point
**Objectif :** diagnostiquer et corriger une standardisation qui mélange les colonnes.

Un collègue veut standardiser les quatre mesures des manchots **colonne par colonne** : dans chaque colonne, retirer la moyenne de la colonne puis diviser par son écart-type (ddof = 0, comme `np.std`). Exécute son code (cellule suivante) : les moyennes des colonnes standardisées devraient valoir 0, et leurs écarts-types 1.

a) Que calcule `X.mean()` ? Quelle est la forme (*shape*) de son résultat ? Pourquoi la standardisation du collègue est-elle fausse ? (réponds sur ta feuille) **[0,3 pt]**
b) Écris `standardize(X)`, qui renvoie la version standardisée de `X` colonne par colonne (ddof = 0), sans modifier `X`. **[0,4 pt]**
c) Avec ta fonction, donne le z-score de la masse du premier manchot (ligne 0, colonne `body_mass_g`), à 2 décimales (sur ta feuille). **[0,3 pt]**'''

CP13_BUG = r'''# The colleague's code (do not change this cell: run it and read what it prints)
X_std_bug = (X - X.mean()) / X.std()
print("means of the columns:", X_std_bug.mean(axis=0).round(2))
print("stds of the columns :", X_std_bug.std(axis=0).round(3))'''

CP13_TODO = r'''def standardize(X):
    """Return the z-scores of X, column by column: (X - column mean) / column std, with ddof = 0.

    X is a 2-D array of shape (n_samples, n_features); it must not be modified.
    """
    raise NotImplementedError  # TODO CP1.3 b)'''

CP13_TRY = r'''# Try your function (no check during the exam: compare with what you expect)
with wb.attempt("CP1.3"):
    result = standardize(X)
    if result is None:
        print("⚠️ standardize renvoie None : as-tu oublié le return ?")
    else:
        Z_try = np.asarray(result, dtype=float)          # a DataFrame or a list is looked at as an array
        print("means of the columns:", Z_try.mean(axis=0).round(2) + 0.0)   # + 0.0 turns -0.0 into 0.0
        print("stds of the columns (np.std, ddof = 0):", Z_try.std(axis=0).round(3))
        print("z-score of the mass of the first penguin:", Z_try[0, 3])'''

CP13_SOLUTION = r'''# a) X.mean() averages ALL the values of the table (333 x 4 = 1 332 numbers, millimetres and grams mixed):
#    one number, of shape (). The colleague removes the same number from every column and divides by one global std.
print("X.mean():", X.mean(), "· shape:", np.shape(X.mean()), "· X.std():", X.std())
print("X.mean(axis=0):", X.mean(axis=0).round(1), "· shape:", X.mean(axis=0).shape)


def standardize(X):
    """Return the z-scores of X, column by column: (X - column mean) / column std, with ddof = 0.

    X is a 2-D array of shape (n_samples, n_features); it must not be modified.
    """
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0)) / X.std(axis=0)       # shapes (333, 4) - (4,): broadcasting along the rows


Z_3 = standardize(X)
print("means of the columns:", Z_3.mean(axis=0).round(2) + 0.0)   # + 0.0 turns -0.0 into 0.0
print("stds of the columns :", Z_3.std(axis=0).round(3))
print("z-score of the mass of the first penguin:", round(float(Z_3[0, 3]), 4))'''

CP13_RECORD = r'''wb.record("CP1.3c", float(Z_3[0, 3]), decimals=4,
          mistakes={"c'est avec ddof = 1, la valeur par défaut de pandas `.std()` : l'énoncé demande ddof = 0, comme `np.std`":
                    float((X[0, 3] - X[:, 3].mean()) / X[:, 3].std(ddof=1)),
                    "c'est la standardisation fautive du collègue : la moyenne et l'écart-type de TOUT le tableau, au lieu de ceux de la colonne":
                    float(X_std_bug[0, 3]),
                    "tu as standardisé chaque LIGNE (un manchot, avec ses quatre unités) : la moyenne et l'écart-type se calculent par colonne (axis=0)":
                    float((X[0, 3] - X[0].mean()) / X[0].std()),
                    "c'est une normalisation min-max : la standardisation retire la moyenne de la colonne et divise par son écart-type":
                    float((X[0, 3] - X[:, 3].min()) / (X[:, 3].max() - X[:, 3].min()))})'''

CP15_MD = r'''### CP1.5 — Coder un intervalle de confiance bootstrap 🔨 ★★ ⏱️ 10 min · 1,5 point
**Objectif :** programmer le bootstrap percentile de la moyenne, sans librairie.

a) Écris `bootstrap_ci_mean(x, confidence=0.95, n_boot=1000, seed=0)`, qui renvoie l'intervalle de confiance bootstrap **percentile** de la moyenne de `x`, sous la forme d'un tuple de deux `float` `(bas, haut)`, bornes non arrondies : **[0,8 pt]**
- un seul générateur, créé dans la fonction : `rng = np.random.default_rng(seed)` ;
- pour chacun des `n_boot` rééchantillons, **dans l'ordre**, tire `n` indices avec remise, `idx = rng.integers(0, n, size=n)` ($n$ = taille de `x`), et calcule la moyenne de `x[idx]` (tirer toutes les lignes d'un coup, `size=(n_boot, n)`, ou utiliser `rng.choice(x, size=n)` donne exactement les mêmes nombres) ;
- les bornes sont les percentiles $50\,(1 - c)$ et $50\,(1 + c)$ de ces moyennes, avec $c$ = `confidence` (`np.percentile`).

Puis, sur la masse des manchots Gentoo (`gentoo_mass`, en grammes), et sur ta feuille :
b) l'intervalle à 95 % (graine 0), à 1 décimale ; **[0,2 pt]**
c) l'intervalle à 90 % (graine 0) : est-il plus large ou plus étroit ? Pourquoi ? **[0,2 pt]**
d) Laquelle de ces deux phrases est juste, et qu'est-ce qui ne va pas dans l'autre ? (1) « 95 % des manchots Gentoo pèsent entre … et … g. » (2) « Cette méthode donne un intervalle qui contient la vraie masse moyenne des Gentoo environ 95 fois sur 100. » **[0,3 pt]**'''

CP15_TODO = r'''def bootstrap_ci_mean(x, confidence=0.95, n_boot=1000, seed=0):
    """Percentile bootstrap confidence interval of the mean of x, as a tuple (low, high) of floats."""
    raise NotImplementedError  # TODO CP1.5 a)'''

CP15_TRY = r'''# Try your function (no check during the exam)
with wb.attempt("CP1.5"):
    print("95 % interval:", bootstrap_ci_mean(gentoo_mass))
    print("90 % interval:", bootstrap_ci_mean(gentoo_mass, confidence=0.90))'''

CP15_SOLUTION = r'''def bootstrap_ci_mean(x, confidence=0.95, n_boot=1000, seed=0):
    """Percentile bootstrap confidence interval of the mean of x, as a tuple (low, high) of floats."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):                            # in order: resample b uses the b-th draws of rng
        idx = rng.integers(0, n, size=n)               # n indices drawn WITH replacement
        means[b] = x[idx].mean()
    low, high = np.percentile(means, [50 * (1 - confidence), 50 * (1 + confidence)])
    return float(low), float(high)


ci_95 = bootstrap_ci_mean(gentoo_mass)
ci_90 = bootstrap_ci_mean(gentoo_mass, confidence=0.90)
print(f"sample mean: {gentoo_mass.mean():.1f} g · 95 %: {ci_95[0]:.1f} to {ci_95[1]:.1f} g · "
      f"90 %: {ci_90[0]:.1f} to {ci_90[1]:.1f} g")'''

CP15_RECORD = r'''# Classic wrong procedures of a), each recorded with its message
n_g = len(gentoo_mass)
rng = np.random.default_rng(0)
means_ok = np.array([gentoo_mass[rng.integers(0, n_g, size=n_g)].mean() for _ in range(1000)])     # the right one
assert np.allclose(np.percentile(means_ok, [2.5, 97.5]), ci_95)
rng = np.random.default_rng(0)
means_nboot = np.array([gentoo_mass[rng.integers(0, n_g, size=1000)].mean() for _ in range(1000)])  # size n_boot
rng = np.random.default_rng(0)
means_cols = gentoo_mass[rng.integers(0, n_g, size=(n_g, 1000))].mean(axis=0)                       # shape (n, n_boot)
np.random.seed(0)
means_legacy = np.array([gentoo_mass[np.random.randint(0, n_g, size=n_g)].mean() for _ in range(1000)])


def wrong_procedures(c, right):
    """The interval at confidence c given by each classic wrong procedure (message -> [low, high])."""
    q = [50 * (1 - c), 50 * (1 + c)]
    return {
        "chaque rééchantillon a la taille de x (n tirages avec remise), pas n_boot": np.percentile(means_nboot, q).tolist(),
        "ce sont les percentiles des masses elles-mêmes : l'intervalle porte sur la MOYENNE, prends les percentiles des n_boot moyennes bootstrap":
            np.percentile(gentoo_mass, q).tolist(),
        "tes tirages ont la forme (n, n_boot) : chaque rééchantillon doit utiliser, dans l'ordre, n tirages consécutifs (size=n dans la boucle, ou size=(n_boot, n))":
            np.percentile(means_cols, q).tolist(),
        "np.random.seed et np.random.randint utilisent l'ancien générateur : l'énoncé impose rng = np.random.default_rng(seed) et rng.integers":
            np.percentile(means_legacy, q).tolist(),
        "ta fonction arrondit les bornes : renvoie-les sans arrondir (on n'arrondit qu'en écrivant la réponse)":
            [round(v, 1) for v in right],
    }


wb.record("CP1.5b", list(ci_95), decimals=4,
          mistakes={"pour 95 %, coupe 2,5 % de chaque côté : les percentiles 2,5 et 97,5 de la distribution bootstrap":
                    list(ci_90), **wrong_procedures(0.95, ci_95)})
wb.record("CP1.5c", list(ci_90), decimals=4,
          mistakes={"pour 90 %, coupe 5 % de chaque côté : les percentiles 5 et 95 (utilise bien l'argument confidence)":
                    list(ci_95),
                    "pour 90 %, les percentiles sont 50 (1 − c) = 5 et 50 (1 + c) = 95, pas 100 (1 − c) = 10 et 100 c = 90":
                    np.percentile(means_ok, [10, 90]).tolist(),
                    **wrong_procedures(0.90, ci_90)})'''

# ---------------------------------------------------------------------------
# Part B: checks after the exam
# ---------------------------------------------------------------------------
PART_B_INTRO = r'''## Partie B · Après l'examen : vérification automatique

**Ne commence cette partie qu'une fois ta copie terminée**, examen rendu. Elle vérifie le code de la partie A, puis les réponses que tu as écrites sur ta feuille, question par question : ✅ juste, ❌ faux (avec une piste), ⏳ pas rempli. Elle ne note pas : les démarches, les justifications et les questions rédigées se corrigent avec `03_examen_corrige.md`, qui donne aussi le barème.

1. Passe `EXAM_OVER` à `True` dans la cellule ci-dessous, puis exécute-la.
2. Exécute la vérification du code (CP1.3 et CP1.5).
3. Reporte ensuite chaque réponse de ta feuille : **la valeur** que tu as écrite, pas un nouveau calcul (sinon tu ne vérifies plus ta copie). En Python, le séparateur décimal est un **point** (`0.125`).

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


def as_letters(value):
    """Letters typed as "ABCD", "A, B, C, D" or ["A", "B", "C", "D"] -> "ABCD" (`...` stays `...`)."""
    if value is ... or value is None:
        return value
    if isinstance(value, (list, tuple)):
        value = "".join(str(v) for v in value)
    return "".join(ch for ch in str(value) if ch.isalpha()).upper()


def verdict(question, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message, and remember the result."""
    PROPERTIES.setdefault(question, []).append((success or failure, bool(ok)))
    print(f"✅ {question} : {success}" if ok else f"❌ {question} : {failure}")


_LIBRARY_ADVICE = " ; si elle fait partie de `mylearn`, ses tests (`pytest`) disent quel cas pose problème."


def code_check(ex_id, value, reminder=""):
    """wb.check of a value computed by your exam function (no advice about mylearn: it is not used here)."""
    result = wb.check(ex_id, value, computed=True, quiet=True)
    message = result.message.replace(_LIBRARY_ADVICE, ".")
    if not result and reminder and not message.startswith("Erreur classique"):
        message += f" Rappel : {reminder}"
    print(f"{'✅' if result else '❌'} Ex {ex_id} : {message}")
    RESULTS[ex_id] = result
    return result'''

CODE_CHECK_3 = r'''# Checks of standardize (CP1.3)
if exam_over() and ready("CP1.3", "penguins", "MEASURES", "standardize"):
    PROPERTIES["CP1.3"] = []
    RESULTS.pop("CP1.3c", None)
    X_fresh = penguins[MEASURES].to_numpy(dtype=float)     # a fresh copy: the trial of part A may have changed X
    X_before = X_fresh.copy()
    try:
        result = standardize(X_fresh)
    except NotImplementedError:
        print("⏳ CP1.3 : standardize n'est pas encore écrite.")
    except Exception as exc:                   # a bug of the exam code must not stop part B
        verdict("CP1.3", False, "", f"standardize(X) lève une erreur : {type(exc).__name__}: {exc}")
    else:
        verdict("CP1.3", np.array_equal(X_fresh, X_before), "X n'est pas modifié.",
                "ta fonction modifie X : calcule un nouveau tableau au lieu d'écrire dans X (pas de -= ni de /=).")
        Z_check = np.asarray(result if result is not None else np.nan, dtype=float)
        verdict("CP1.3", Z_check.shape == X_fresh.shape, "la forme est (333, 4).",
                "standardize ne renvoie rien : as-tu oublié le return ?" if result is None else
                f"le résultat doit avoir la forme de X, {X_fresh.shape}, pas {Z_check.shape}.")
        if Z_check.shape == X_fresh.shape:
            verdict("CP1.3", np.allclose(Z_check.mean(axis=0), 0, atol=1e-6), "chaque colonne a une moyenne nulle.",
                    "les moyennes des colonnes ne sont pas nulles : retire la moyenne de CHAQUE colonne (axis=0).")
            verdict("CP1.3", np.allclose(Z_check.std(axis=0), 1, atol=1e-6), "chaque colonne a un écart-type de 1 (ddof = 0).",
                    "les écarts-types des colonnes ne valent pas 1 avec ddof = 0 : divise par l'écart-type de chaque "
                    "colonne, calculé avec ddof = 0.")
            code_check("CP1.3c", float(Z_check[0, 3]),
                       "z = (x − moyenne de la colonne) / écart-type de la colonne (ddof = 0), ici pour la ligne 0 "
                       "et la colonne 3 (body_mass_g).")'''

CODE_CHECK_5 = r'''# Checks of bootstrap_ci_mean (CP1.5)
if exam_over() and ready("CP1.5", "gentoo_mass", "bootstrap_ci_mean"):
    PROPERTIES["CP1.5"] = []
    RESULTS.pop("CP1.5b", None)
    RESULTS.pop("CP1.5c", None)
    try:
        raw = [bootstrap_ci_mean(gentoo_mass), bootstrap_ci_mean(gentoo_mass),
               bootstrap_ci_mean(gentoo_mass, confidence=0.90)]
    except NotImplementedError:
        print("⏳ CP1.5 : bootstrap_ci_mean n'est pas encore écrite.")
    except Exception as exc:                   # a bug of the exam code must not stop part B
        verdict("CP1.5", False, "", f"bootstrap_ci_mean(gentoo_mass) lève une erreur : {type(exc).__name__}: {exc}")
    else:
        if any(ci is None for ci in raw):
            verdict("CP1.5", False, "", "bootstrap_ci_mean ne renvoie rien : as-tu oublié le return ?")
        else:
            verdict("CP1.5", all(isinstance(ci, tuple) for ci in raw), "la fonction renvoie un tuple (bas, haut).",
                    f"la fonction doit renvoyer un tuple, pas un objet de type {type(raw[0]).__name__} : "
                    "return float(low), float(high) (le reste est vérifié quand même).")
        try:
            ci_a, ci_b, ci_c = (tuple(float(v) for v in np.ravel(ci)) for ci in raw)
        except (TypeError, ValueError):
            ci_a = ci_b = ci_c = ()
        if not all(len(ci) == 2 for ci in (ci_a, ci_b, ci_c)):
            if all(ci is not None for ci in raw):
                verdict("CP1.5", False, "", "la fonction doit renvoyer deux nombres, la borne basse puis la borne haute.")
        else:
            verdict("CP1.5", ci_a == ci_b, "même graine, même intervalle.",
                    "deux appels avec la même graine donnent deux intervalles différents : crée le générateur "
                    "DANS la fonction, avec np.random.default_rng(seed).")
            if ci_a[0] == ci_a[1]:
                verdict("CP1.5", False, "", "l'intervalle a une largeur nulle : tous tes rééchantillons ont la même "
                        "moyenne. Tire les indices AVEC remise (rng.integers), avec un générateur créé une seule "
                        "fois, avant la boucle.")
            else:
                verdict("CP1.5", ci_a[0] < gentoo_mass.mean() < ci_a[1],
                        "l'intervalle contient la moyenne de l'échantillon.",
                        "l'intervalle devrait entourer la moyenne de l'échantillon : vérifie les percentiles.")
                verdict("CP1.5", ci_a[0] < ci_c[0] < ci_c[1] < ci_a[1],
                        "l'intervalle à 90 % est inclus dans celui à 95 %.",
                        "l'intervalle à 90 % devrait être inclus dans celui à 95 % : vérifie les percentiles "
                        "50 (1 − c) et 50 (1 + c).")
            checks = [code_check("CP1.5b", list(ci_a)), code_check("CP1.5c", list(ci_c))]
            if not all(checks) and not any(r.message.startswith("Erreur classique") for r in checks):
                print("   Rappel de la procédure imposée : un générateur np.random.default_rng(seed) créé dans la "
                      "fonction ; pour chacun des n_boot rééchantillons, dans l'ordre, n indices "
                      "rng.integers(0, n, size=n) et la moyenne de x[idx] ; puis les percentiles 50 (1 − c) et "
                      "50 (1 + c) de ces moyennes, sans arrondir.")'''

PAPER_CONTEXT = r'''# The paper questions only use the numbers of their statements (01_examen_sujet.md)
import math
from fractions import Fraction

import numpy as np

u_2, v_2, w_2 = np.array([1, -2, 2]), np.array([2, 3, 4]), np.array([2, 1, 0])          # CP1.2
x_4 = np.array([2, 4, 6, 8, 10], dtype=float)                                           # CP1.4
y_4 = np.array([3, 7, 5, 11, 9], dtype=float)
N_6, PREVALENCE_6, SENSITIVITY_6, SPECIFICITY_6 = 25_000, 0.04, 0.90, 0.92              # CP1.6
SICK_6 = N_6 * PREVALENCE_6
TP_6, FP_6 = SENSITIVITY_6 * SICK_6, (1 - SPECIFICITY_6) * (N_6 - SICK_6)
FN_6, TN_6 = SICK_6 - TP_6, (N_6 - SICK_6) - FP_6
THETA_8, PRIOR_8 = [Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)], [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)]


def posterior(prior, likelihood):
    """Bayes' rule on a list of hypotheses (exact with fractions)."""
    joint = [p * l for p, l in zip(prior, likelihood)]
    return [j / sum(joint) for j in joint]


POST1_8 = posterior(PRIOR_8, THETA_8)                    # after one heads
POST2_8 = posterior(POST1_8, THETA_8)                    # after heads, heads


def f_9(point):
    x, y = point
    return (x - 1) ** 2 - 0.5 * (y + 2) ** 2


def grad_9(point):
    x, y = point
    return np.array([2 * (x - 1), -(y + 2)])


STEPS_9 = [np.array([2.0, -1.0])]
for _ in range(2):
    STEPS_9.append(STEPS_9[-1] - 0.25 * grad_9(STEPS_9[-1]))
P_11 = np.array([0.4, 0.3, 0.2, 0.1])                    # vowels, frequent, medium, rare consonants
Q_11 = np.full(4, 0.25)
H_11 = float(-(P_11 * np.log2(P_11)).sum())
LENGTHS_11 = np.array([1, 2, 3, 3])                       # Huffman lengths [V, F, M, R]'''

PAPER = [
    Paper("CP1.1", "Questions flash (le verdict seulement ; les justifications se notent avec le corrigé)", [
        ("a", 'True / False (ou "vrai" / "faux")', "True", ""),
        ("b", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("c", 'True / False (ou "vrai" / "faux")', "True", ""),
        ("d", 'True / False (ou "vrai" / "faux")', "True", ""),
        ("e", 'True / False (ou "vrai" / "faux")', "False", ""),
        ("f", 'True / False (ou "vrai" / "faux")', "False", ""),
    ]),
    Paper("CP1.2", "Norme, produit scalaire et log₂", [
        ("a", "the norm of u", "int(round(float(np.linalg.norm(u_2))))",
         r'''mistakes={"c'est ‖u‖² : il manque la racine carrée": 9,
                     "la norme n'est pas la somme des valeurs absolues : c'est la racine de la somme des carrés": 5}'''),
        ("b", "u · v", "int(u_2 @ v_2)",
         r'''mistakes={"attention au signe : la deuxième composante de u est négative": 16}'''),
        ("c", "u and w orthogonal? True / False (ou \"vrai\" / \"faux\")", "bool(u_2 @ w_2 == 0)", ""),
        ("d", "log2(1/32)", "int(round(math.log2(1 / 32)))", ""),
        ("e", "log2(24), 3 decimals", "math.log2(24)",
         r'''decimals=3, mistakes={"24 = 8 × 3 : le log d'un produit est la SOMME des logs, pas un produit": 3 * math.log2(3)}'''),
    ]),
    Paper("CP1.4", "Statistiques de cinq points à la main", [
        ("a", "[mean of x, mean of y]", "[x_4.mean(), y_4.mean()]",
         r'''decimals=1, mistakes={"tu as inversé x̄ et ȳ : l'ordre demandé est [x̄, ȳ]": [y_4.mean(), x_4.mean()]}'''),
        ("b", "[variance of x, std of x], 2 decimals", "[x_4.var(), x_4.std()]",
         r'''decimals=2, mistakes={"tu as divisé par n − 1 : ici ddof = 0, on divise par n": [x_4.var(ddof=1), x_4.std(ddof=1)],
                     "l'ordre demandé est [variance, écart-type]": [x_4.std(), x_4.var()],
                     "tu n'as pas divisé la somme des carrés des écarts par n": [((x_4 - x_4.mean()) ** 2).sum(), math.sqrt(((x_4 - x_4.mean()) ** 2).sum())]}'''),
        ("c", "z-score of the value x = 10, 2 decimals", "(10 - x_4.mean()) / x_4.std()",
         r'''decimals=2, mistakes={"tu as pris l'écart-type avec ddof = 1 : ici ddof = 0, on divise par n": (10 - x_4.mean()) / x_4.std(ddof=1),
                     "on divise l'écart à la moyenne par l'écart-type, pas par la variance": (10 - x_4.mean()) / x_4.var()}'''),
        ("d", "[covariance with ddof = 0, with ddof = 1]", "[np.cov(x_4, y_4, ddof=0)[0, 1], np.cov(x_4, y_4, ddof=1)[0, 1]]",
         r'''decimals=1, mistakes={"tu as inversé les deux : avec ddof = 0, on divise par n = 5 ; avec ddof = 1, par n − 1 = 4": [np.cov(x_4, y_4, ddof=1)[0, 1], np.cov(x_4, y_4, ddof=0)[0, 1]]}'''),
        ("e", "correlation r, 2 decimals", "np.corrcoef(x_4, y_4)[0, 1]",
         r'''decimals=2, mistakes={"c'est r² : la corrélation est la covariance divisée par le produit des écarts-types": np.corrcoef(x_4, y_4)[0, 1] ** 2,
                     "divise la covariance par le PRODUIT des écarts-types, pas par la somme des carrés des écarts": np.cov(x_4, y_4, ddof=0)[0, 1] / ((x_4 - x_4.mean()) ** 2).sum()}'''),
        ("f", "[new mean of y, new median of y]", "[np.append(y_4, 40).mean(), np.median(np.append(y_4, 40))]",
         r'''decimals=1, mistakes={"avec six valeurs, la médiane est la moyenne des deux valeurs du milieu, une fois triées": [np.append(y_4, 40).mean(), 7.0]}'''),
    ]),
    Paper("CP1.6", "Dépistage : matrice de confusion, precision, NPV et règle de Bayes", [
        ("a", "[TP, FN, FP, TN]", "[int(round(TP_6)), int(round(FN_6)), int(round(FP_6)), int(round(TN_6))]",
         r'''mistakes={"les faux positifs se comptent parmi les SAINS seulement, pas parmi toutes les personnes": [900, 100, 2000, 22000],
                     "l'ordre demandé est TP, FN, FP, TN": [900, 1920, 100, 22080]}'''),
        ("b", "precision, 3 decimals", "TP_6 / (TP_6 + FP_6)",
         r'''decimals=3, mistakes={"c'est la sensibilité P(positif | malade) : on demande P(malade | positif)": 0.9,
                     "c'est P(malade et positif) : divise par P(positif)": SENSITIVITY_6 * PREVALENCE_6}'''),
        ("c", "NPV, 4 decimals", "TN_6 / (TN_6 + FN_6)",
         r'''decimals=4, mistakes={"c'est la spécificité P(négatif | sain) : on demande P(sain | négatif)": 0.92}'''),
        ("d", "accuracy, 3 decimals", "(TP_6 + TN_6) / N_6",
         r'''decimals=3, mistakes={"c'est la moyenne de la sensibilité et de la spécificité (la balanced accuracy) : l'accuracy compte les bonnes réponses parmi les 25 000": 0.91}'''),
        ("f", "P(sick | two positive tests), 3 decimals",
         "SENSITIVITY_6 * (TP_6 / (TP_6 + FP_6)) / (SENSITIVITY_6 * (TP_6 / (TP_6 + FP_6)) + (1 - SPECIFICITY_6) * (FP_6 / (TP_6 + FP_6)))",
         r'''decimals=3, mistakes={"un second résultat positif doit faire monter la probabilité : applique la règle de Bayes une seconde fois": TP_6 / (TP_6 + FP_6),
                     "c'est P(deux positifs | malade) : on demande P(malade | deux positifs)": SENSITIVITY_6 ** 2}'''),
    ]),
    Paper("CP1.7", "Lire une courbe ROC et une courbe PR (seule la ligne de base se vérifie ici : les lectures se notent avec le corrigé)", [
        ("e", "height of the random 'curve' with 1 % of frauds, 2 decimals", "0.01",
         r'''decimals=2, mistakes={"la diagonale vaut pour la courbe ROC ; un classifieur au hasard a une precision égale à la proportion de positifs": 0.5,
                     "0 serait la precision d'un modèle dont aucune alerte n'est juste : au hasard, la part des fraudes parmi les alertes est la même que partout": 0.0}'''),
    ]),
    Paper("CP1.8", "Trois hypothèses, deux lancers", [
        ("a", "posterior after the first heads, 2 decimals", "[float(p) for p in POST1_8]",
         r'''decimals=2, mistakes={"tu n'as pas divisé par l'évidence : un posterior somme à 1": [float(p * t) for p, t in zip(PRIOR_8, THETA_8)],
                     "tu as oublié le prior : multiplie chaque vraisemblance par son prior avant de normaliser": [float(t / sum(THETA_8)) for t in THETA_8]}'''),
        ("b", "posterior after heads, heads, 3 decimals", "[float(p) for p in POST2_8]",
         r'''decimals=3, mistakes={"tu n'as pas divisé par l'évidence : un posterior somme à 1": [float(p * t) for p, t in zip(POST1_8, THETA_8)],
                     "tu as oublié le prior : multiplie chaque vraisemblance par son prior avant de normaliser": [float(t * t / sum(s * s for s in THETA_8)) for t in THETA_8],
                     "prior × θ² est juste, mais il faut encore diviser par leur somme : un posterior somme à 1": [float(p * t * t) for p, t in zip(PRIOR_8, THETA_8)]}'''),
        ("c", "P(heads, heads), 3 decimals", "float(sum(p * t * t for p, t in zip(PRIOR_8, THETA_8)))",
         r'''decimals=3, mistakes={"c'est l'évidence du second lancer seulement : celle de la suite est le produit des deux évidences (ou Σ prior × θ²)": float(sum(p * t for p, t in zip(POST1_8, THETA_8))),
                     "c'est l'évidence du premier lancer seulement : il faut celle de la suite des deux": float(sum(p * t for p, t in zip(PRIOR_8, THETA_8)))}'''),
        ("d", "the MAP theta (0.2, 0.5 or 0.8)", "0.8",
         r'''decimals=2, choices=[0.2, 0.5, 0.8], mistakes={"0,5 avait le plus grand PRIOR : le MAP se lit sur le posterior après les deux lancers": 0.5,
                     "relis le posterior de b) : le MAP est l'hypothèse dont le posterior est le plus grand": 0.2}'''),
        ("e", "P(third throw = heads), 3 decimals", "float(sum(p * t for p, t in zip(POST2_8, THETA_8)))",
         r'''decimals=3, mistakes={"ta valeur est l'un des θ : la probabilité de face au troisième lancer pondère chaque θ par son posterior": 0.8,
                     "c'est la prédiction faite après UNE face : pour le troisième lancer, le prior est le posterior après les deux faces": float(sum(p * t for p, t in zip(POST1_8, THETA_8)))}'''),
    ]),
    Paper("CP1.9", "Gradient, point selle et deux pas de descente", [
        ("b", "[x, y] of the critical point", "[1, -2]",
         r'''mistakes={"attention au signe : vérifie ta valeur de y dans l'équation ∂f/∂y = 0": [1, 2]}'''),
        ("d", "[[x1, y1], [x2, y2]], exact values", "[STEPS_9[1].tolist(), STEPS_9[2].tolist()]",
         r'''decimals=4, mistakes={"tu es monté : la descente RETIRE η × gradient": [[2.5, -1.25], [3.25, -1.4375]],
                     "tes pas déplacent y dans le mauvais sens : vérifie le signe de ∂f/∂y": [[1.5, -1.25], [1.25, -1.4375]]}'''),
        ("e", "[f at the start, after one step, after two steps], 3 decimals", "[f_9(p) for p in STEPS_9]", "decimals=3"),
    ]),
    Paper("CP1.10", "Prédire l'effet du learning rate", [
        ("a", "your four letters, in the order of the learning rates (for example \"AAAA\")", "\"CADB\"",
         r'''mistakes={"revois le cas η = 0,3 : calcule x₁ et x₂ à partir de x₀ = 1": "DADB",
                     "revois le cas η = 0,4 : calcule x₁ et x₂ à partir de x₀ = 1": "CACB"}'''),
        ("b", "upper bound of the learning rates that converge, 3 decimals", "1 / 3",
         r'''decimals=3, mistakes={"1/6 amène x à 0 en un seul pas ; la convergence demande seulement |1 − 6η| < 1": 1 / 6,
                     "c'est la borne pour f′(x) = 2x (le facteur 3 oublié) ; ici f′(x) = 6x": 1.0,
                     "f′(x) = 6x : dérive 3x² avant d'écrire x_{t+1}": 2 / 3}'''),
    ]),
    Paper("CP1.11", "Entropie, cross-entropy, KL et Huffman", [
        ("a", "H(p) in bits, 3 decimals", "H_11",
         r'''decimals=3, mistakes={"c'est en nats : on demande des bits (log₂)": float(-(P_11 * np.log(P_11)).sum()),
                     "pondère chaque surprise −log₂ p_i par sa probabilité p_i": float(-np.log2(P_11).sum()),
                     "p n'est pas uniforme : calcule −Σ p_i log₂ p_i avec ses quatre probabilités": 2.0}'''),
        ("b", "Huffman lengths [V, F, M, R]", "LENGTHS_11.tolist()",
         r'''mistakes={"l'ordre demandé est [V, F, M, R], du plus probable au moins probable": [3, 3, 2, 1],
                     "c'est un code de longueur fixe : Huffman fusionne à chaque étape les deux groupes les moins probables": [2, 2, 2, 2]}'''),
        ("c", "mean length, 2 decimals", "float((P_11 * LENGTHS_11).sum())",
         r'''decimals=2, mistakes={"pondère chaque longueur par la probabilité de son symbole": float(LENGTHS_11.mean())}'''),
        ("e", "H(p, q) in bits, 3 decimals", "float(-(P_11 * np.log2(Q_11)).sum())",
         r'''decimals=3, mistakes={"dans H(p, q), les poids viennent de p et les logs de q": H_11,
                     "attention à l'ordre : H(p, q) pondère par p, le premier argument": float(-(Q_11 * np.log2(P_11)).sum())}'''),
        ("f", "KL(p || q), 3 decimals", "float((P_11 * np.log2(P_11 / Q_11)).sum())",
         r'''decimals=3, mistakes={"attention au sens : dans KL(p ‖ q), les poids viennent de p": float((Q_11 * np.log2(Q_11 / P_11)).sum()),
                     "c'est en nats : on demande des bits (log₂)": float((P_11 * np.log(P_11 / Q_11)).sum())}'''),
        ("g", "KL(q || p), 3 decimals", "float((Q_11 * np.log2(Q_11 / P_11)).sum())",
         r'''decimals=3, mistakes={"attention au sens : dans KL(q ‖ p), les poids viennent de q": float((P_11 * np.log2(P_11 / Q_11)).sum()),
                     "c'est en nats : on demande des bits (log₂)": float((Q_11 * np.log(Q_11 / P_11)).sum())}'''),
        ("h", "is H(p, q') finite? True / False (ou \"vrai\" / \"faux\")", "False", ""),
    ]),
]

SUMMARY = r'''# Summary of the automatic checks
import re


def natural(sub_id):
    """CP1.2a before CP1.10a: the question number as a number, then the letter."""
    number, letter = re.match(r"CP1\.(\d+)(.*)", sub_id).groups()
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
    print(f"Propriétés du code (CP1.3, CP1.5) : {sum(checked)} sur {len(checked)}.")
    print("Note ensuite ta copie avec le barème de 03_examen_corrige.md : il donne aussi les points des démarches, "
          "des justifications et des questions rédigées, que cette vérification ne voit pas.")'''


WRAP = {"CP1.10a": "as_letters"}     # sub-answers normalized before the check (letters typed with separators)


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
        title = "# Checkpoint I · Examen blanc — notebook de l'examen"
        how = ("Ce notebook accompagne le sujet `01_examen_sujet.md`.\n\n"
               "- **Partie A, pendant l'examen** : les deux questions de code, CP1.3 (🐛, 1 point) et CP1.5 (🔨, "
               "1,5 point). Écris ton code dans les cellules à compléter, sans `mylearn`, sans les corrigés et sans "
               "assistant IA ; `help()` et la documentation officielle de NumPy sont permis. Rien n'est vérifié "
               "pendant l'examen : les cellules « essai » montrent seulement ce que fait ton code. Tes réponses "
               "chiffrées et tes explications vont sur ta feuille (`04_mes_reponses.md`).\n"
               "- **Partie B, après l'examen** : la vérification automatique de ton code et des réponses de ta "
               "feuille. Ses cellules ne vérifient rien tant que `EXAM_OVER` vaut `False`.\n\n"
               "« Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche ⏳ "
               "(partie A) ou ⏸️ (partie B).\n\n"
               "> Travaille dans **ta copie** (`mon_travail/checkpoints/partie_1/02_examen_notebook.ipynb`, créée "
               "par `python tools/start_chapter.py CP1`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge ouvre cette version du dépôt, "
               "qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique `00_setup/COLAB.md` §2.")
    else:
        title = "# Checkpoint I · Examen blanc — solutions (notebook exécuté)"
        how = ("Les solutions des deux questions de code (CP1.3 et CP1.5), exécutées, puis l'enregistrement des "
               "réponses que vérifie la partie B du notebook de l'examen (cellules marquées `answer`, récoltées "
               "par `tools/build_answers.py`). Le corrigé détaillé et le barème sont dans `03_examen_corrige.md`.")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell("demo"))
    cells.append(md("## Partie A · Pendant l'examen : les deux questions de code" if kind == "exercise"
                    else "## Partie A · Les deux questions de code"))
    cells.append(code(GIVEN))
    cells.append(md(CP13_MD))
    cells.append(code(CP13_BUG))
    if kind == "exercise":
        cells += [code(CP13_TODO), code(CP13_TRY)]
    else:
        cells += [code(CP13_SOLUTION), code(CP13_RECORD, tags=["answer"])]
    cells.append(md(CP15_MD))
    if kind == "exercise":
        cells += [code(CP15_TODO), code(CP15_TRY)]
        cells.append(md("---\n\n### 🛑 Fin de la partie A\n\nL'examen s'arrête ici. Rends ta copie (enregistre ce "
                        "notebook et ta feuille), puis, seulement après, passe à la partie B."))
        cells.append(md(PART_B_INTRO))
        cells.append(code(PART_B_FLAG))
        cells.append(md("### Le code de la partie A"))
        cells.append(code(CODE_CHECK_3))
        cells.append(code(CODE_CHECK_5))
        cells.append(md("### Les réponses de ta feuille"))
        for paper in PAPER:
            cells += paper_cell(kind, paper)
        cells.append(md("### Bilan"))
        cells.append(code(SUMMARY))
        cells.append(md("**Ce que cette vérification ne voit pas** : les démarches et les justifications ; les "
                        "explications (CP1.1, CP1.3 a, CP1.5 c–d, CP1.6 e et g, CP1.7 a–d et f, CP1.8 f, CP1.9 a, c et f, "
                        "CP1.10 c, CP1.11 d, CP1.12 à CP1.14), la formule de CP1.10 b (seule la borne est vérifiée) ; les "
                        "conclusions écrites à côté d'une valeur vérifiée "
                        "(CP1.4 f, CP1.9 e, CP1.11 g et h) ; et, pour CP1.3 et CP1.5, les nombres recopiés sur ta "
                        "feuille (seule ta fonction est vérifiée). Tout cela se corrige avec `03_examen_corrige.md`. "
                        "Ensuite : la remédiation du corrigé (relis la synthèse, `05_synthese.md`, pour les notions ratées), puis le "
                        "mini-projet."))
    else:
        cells += [code(CP15_SOLUTION), code(CP15_RECORD, tags=["answer"])]
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
