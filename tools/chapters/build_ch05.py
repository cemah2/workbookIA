#!/usr/bin/env python
"""Build the two notebooks of chapter 5 from a single source (used by Claude).

    python tools/chapters/build_ch05.py
    python tools/run_all_notebooks.py chapitres/ch05_courbes/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch05_courbes/03_notebook.ipynb

Part 0 checks the ✏️ paper exercises (5.1 to 5.5). Parts A to D are the notebook
exercises 5.11 to 5.25: numerical derivatives, the step h and the extrema of the solar
cycles (A), numerical gradients, a bug and level lines (B), gradient descent, the
learning rate, a saddle point and torch.autograd (C), a figure, parametrized tests,
critical points and the Rosenbrock challenge (D).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, guarded, md, paper_cells,  # noqa: E402
                         part_cells, setup_cell, write_notebook)

CHAPTER = "5"
FOLDER = "chapitres/ch05_courbes"


def indent(code: str, spaces: int = 4) -> str:
    """Indent every non-empty line of `code` (to put a block inside `with wb.attempt(...):`)."""
    return "\n".join((" " * spaces + line) if line.strip() else "" for line in code.splitlines())


# ---------------------------------------------------------------------------
# Part 0: checks of the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The paper exercises only use the numbers of their statements (02_exercices.md)
from fractions import Fraction as Fr

import numpy as np


def inverse(x):
    """5.1: f(x) = 1/x, exact with fractions."""
    return 1 / Fr(x)


def symmetric_slope(f, a, h):
    """Slope of the secant through the points of abscissas a - h and a + h."""
    return (f(a + h) - f(a - h)) / (2 * h)


A51 = Fr(2)
SYM_51 = {h: symmetric_slope(inverse, A51, Fr(h)) for h in ("1", "1/2", "1/10")}
FORWARD_51 = (inverse(Fr("2.1")) - inverse(A51)) / Fr("0.1")
BACKWARD_51 = (inverse(A51) - inverse(Fr("1.9"))) / Fr("0.1")
DERIV_51 = Fr(-1, 4)


def grad_52(x, y):
    """5.2: gradient of x² + xy + 2y²."""
    return np.array([2 * x + y, x + 4 * y], dtype=float)


G52 = grad_52(1, -1)
NORM_52 = float(np.linalg.norm(G52))
U52 = np.array([0.6, 0.8])


def descend_53(x, eta, steps, derivative, sign=-1.0):
    """5.3: the points x_1, ..., x_steps of a descent (sign -1) or an ascent (sign +1)."""
    points = []
    for _ in range(steps):
        x = x + sign * eta * derivative(x)
        points.append(x)
    return points


def f53(x):
    return 2 * (x - 1) ** 2


def df53(x):
    return 4 * (x - 1)


def dg53(x):
    return 4 - 2 * x


def f54(x):
    return x ** 3 - 3 * x


def saddle_steps_55(point, eta, steps, flipped_y=False):
    """5.5: descent on x² - y² (gradient (2x, -2y)); flipped_y uses the wrong sign for y."""
    x, y = point
    for _ in range(steps):
        x, y = x - eta * 2 * x, y - eta * (2 * y if flipped_y else -2 * y)
    return [x, y]'''

PAPER = [
    Paper("5.1", "La sécante qui se resserre sur la tangente", [
        ("a", "slope of the symmetric secant for h = 1, 3 decimals", 'float(SYM_51["1"])',
         r'''decimals=3, mistakes={"c'est la sécante avant, de 2 à 3 : la sécante symétrique passe par les points d'abscisses 1 et 3": float(inverse(3) - inverse(2)),
                     "les deux points sont distants de 2h : divise la différence des hauteurs par 2h, pas par h": float(2 * SYM_51["1"])}'''),
        ("b", "slope for h = 0.5, 4 decimals", 'float(SYM_51["1/2"])',
         r'''decimals=4, mistakes={"c'est la sécante avant, de 2 à 2,5 : la sécante symétrique passe par 1,5 et 2,5": float((inverse(Fr("2.5")) - inverse(2)) / Fr("0.5")),
                     "les deux points sont distants de 2h : divise par 2h, pas par h": float(2 * SYM_51["1/2"])}'''),
        ("c", "slope for h = 0.1, 5 decimals", 'float(SYM_51["1/10"])',
         r'''decimals=5, mistakes={"c'est la sécante avant (question e) : la sécante symétrique passe par 1,9 et 2,1": float(FORWARD_51),
                     "les deux points sont distants de 2h : divise par 2h, pas par h": float(2 * SYM_51["1/10"])}'''),
        ("d", "f'(2), the limit when h tends to 0", "float(DERIV_51)",
         r'''decimals=4, mistakes={"c'est la pente pour h = 0,1 (question c) : on demande la limite quand h tend vers 0": float(SYM_51["1/10"])}'''),
        ("e", "slope of the forward secant between 2 and 2.1, 4 decimals", "float(FORWARD_51)",
         r'''decimals=4, mistakes={"c'est la pente symétrique (question c) : on demande la sécante avant, entre 2 et 2,1": float(SYM_51["1/10"]),
                     "c'est la sécante arrière, entre 1,9 et 2 : on demande la sécante avant, entre 2 et 2,1": float(BACKWARD_51)}'''),
        ("f", "error of the forward slope / error of the symmetric slope, 1 decimal",
         "float(abs(FORWARD_51 - DERIV_51) / abs(SYM_51['1/10'] - DERIV_51))",
         r'''decimals=1, mistakes={"c'est l'inverse : mets l'écart de la pente avant au numérateur": float(abs(SYM_51["1/10"] - DERIV_51) / abs(FORWARD_51 - DERIV_51)),
                     "tu as utilisé les valeurs arrondies de c) et e) : garde les fractions exactes jusqu'au bout": (0.25 - 0.2381) / (0.25063 - 0.25)}'''),
    ]),
    Paper("5.2", "Gradient à la main et direction de plus grande pente", [
        ("a", "f(P)", "1 ** 2 + 1 * (-1) + 2 * (-1) ** 2",
         r'''mistakes={"vérifie le signe du terme xy en P, et que le carré de 2y² ne porte que sur y": 1 + 1 + 2,
           "2y² vaut 2 × (−1)² = 2 en P : n'oublie pas le facteur 2": 1 - 1 + 1}'''),
        ("b", "the gradient [df/dx, df/dy] at P, a list", "G52.tolist()",
         r'''mistakes={"tu as oublié le terme xy : il dépend des deux variables, il compte dans les deux dérivées partielles": [2.0, -4.0],
           "l'ordre demandé est [∂f/∂x, ∂f/∂y]": G52[::-1].tolist()}'''),
        ("c", "its norm, 3 decimals", "NORM_52",
         r'''decimals=3, mistakes={"c'est le carré de la norme : prends la racine carrée": NORM_52 ** 2,
                     "c'est la norme d'un gradient sans le terme xy": float(np.linalg.norm([2, -4]))}'''),
        ("d", "the unit vector of steepest DESCENT, a list (3 decimals)", "(-G52 / NORM_52).tolist()",
         r'''decimals=3, mistakes={"c'est la direction de plus grande MONTÉE : la descente va contre le gradient": (G52 / NORM_52).tolist(),
                     "un vecteur unitaire a une norme 1 : divise par la norme du gradient": (-G52).tolist()}'''),
        ("e", "the slope of f at P in the direction u = (0.6, 0.8), 1 decimal", "float(G52 @ U52)",
         r'''decimals=1, mistakes={"chaque composante du gradient se multiplie par la composante de u de même rang": float(G52[0] * U52[1] + G52[1] * U52[0]),
                     "c'est la plus grande pente (la norme du gradient) : on demande la pente dans la direction u": NORM_52}'''),
        ("f", "a unit vector with zero slope, first component positive, a list (3 decimals)",
         "(np.array([3.0, 1.0]) / np.sqrt(10)).tolist()",
         r'''decimals=3, mistakes={"l'énoncé demande la direction de première composante positive": (np.array([-3.0, -1.0]) / np.sqrt(10)).tolist(),
                     "un vecteur unitaire a une norme 1 : divise par sa norme": [3.0, 1.0],
                     "vérifie que son produit scalaire avec le gradient est nul : celui-ci ne l'est pas": (np.array([1.0, 3.0]) / np.sqrt(10)).tolist()}'''),
    ]),
    Paper("5.3", "Trois pas de descente de gradient à la main", [
        ("a", "f'(5)", "df53(5)",
         r'''mistakes={"la dérivée de 2(x − 1)² est 4(x − 1) : n'oublie pas le facteur 2 qui est devant": 2 * (5 - 1),
           "c'est f(5) : on demande la dérivée f'(5)": f53(5)}'''),
        ("b", "[x1, x2, x3]", "descend_53(5.0, 0.125, 3, df53)",
         r'''decimals=3, mistakes={"tu es monté : pour descendre, on RETIRE η f'(x)": descend_53(5.0, 0.125, 3, df53, sign=1.0),
           "la dérivée de 2(x − 1)² est 4(x − 1) : n'oublie pas le facteur 2": descend_53(5.0, 0.125, 3, lambda x: 2 * (x - 1)),
           "on demande les trois points SUIVANTS, x1, x2 et x3 : pas le point de départ x0": [5.0] + descend_53(5.0, 0.125, 2, df53),
           "x0 n'est pas à donner : la liste ne contient que x1, x2 et x3": [5.0] + descend_53(5.0, 0.125, 3, df53)}'''),
        ("c", "[f(x0), f(x1), f(x2), f(x3)]", "[f53(x) for x in [5.0] + descend_53(5.0, 0.125, 3, df53)]", "decimals=3"),
        ("d", "ascent on g(x) = 4x - x², [x1, x2, x3]", "descend_53(-1.0, 0.25, 3, dg53, sign=1.0)",
         r'''decimals=3, mistakes={"tu es descendu : pour monter, on AJOUTE η g'(x)": descend_53(-1.0, 0.25, 3, dg53)}'''),
        ("e", "with eta = 0.5: [x1, x2]", "descend_53(5.0, 0.5, 2, df53)",
         r'''mistakes={"c'est avec η = 0,125 : refais les deux pas avec η = 0,5": descend_53(5.0, 0.125, 2, df53)}'''),
        ("f", "the upper bound of the learning rates that converge", "0.5",
         r'''decimals=2, mistakes={"au-delà de cette valeur, x dépasse le minimum, mais s'en rapproche encore : écris la condition |1 − 4η| < 1": 0.25}'''),
        ("g", "the learning rate that reaches the minimum in one step", "0.25",
         r'''decimals=2, mistakes={"à cette valeur, la descente rebondit d'un côté à l'autre (e) : cherche le η pour lequel 1 − 4η = 0": 0.5}'''),
    ]),
    Paper("5.4", "Tableau de variations : extrema locaux et globaux de x³ − 3x", [
        ("a", "the points where f'(x) = 0, in increasing order (a list)", "[-1, 1]",
         r'''decimals=3, mistakes={"l'ordre demandé est croissant": [1, -1],
                     "ce sont les points où f s'annule : on demande ceux où sa DÉRIVÉE s'annule": [-np.sqrt(3), np.sqrt(3)],
                     "ce sont les trois points où f s'annule : on demande ceux où sa DÉRIVÉE s'annule": [-np.sqrt(3), 0, np.sqrt(3)]}'''),
        ("b", "the values of f at the points of a), same order (a list)", "[f54(-1), f54(1)]",
         r'''mistakes={"garde l'ordre de a), croissant": [f54(1), f54(-1)]}'''),
        ("c", "f'' at the smaller point of a)", "6 * (-1)",
         r'''mistakes={"6, c'est la dérivée seconde au plus GRAND des deux points (ou une dérivée seconde qui a perdu son x) : on la demande au plus petit": 6,
           "c'est la dérivée PREMIÈRE, nulle en ce point critique : dérive une seconde fois": 0}'''),
        ("d", "the global maximum on [-2.5, 2.5], 3 decimals", "f54(2.5)",
         r'''decimals=3, mistakes={"c'est le maximum LOCAL, en −1 : sur [−2,5 ; 2,5], compare aussi les valeurs aux bornes": f54(-1)}'''),
        ("e", "where it is reached (x)", "2.5",
         r'''decimals=1, mistakes={"le maximum local n'est pas le maximum global sur cet intervalle : regarde les bornes": -1,
                     "c'est l'abscisse du minimum global": -2.5}'''),
        ("f", "on [-2, 2], the number of points where f reaches its global maximum", "2",
         r'''mistakes={"f(2) vaut aussi 2 : compte tous les points où f atteint sa valeur maximale sur [−2 ; 2]": 1}'''),
        ("g", "where f'' = 0 (x)", "0",
         r'''mistakes={"1 annule f′ (c'est un point critique) : le point d'inflexion annule f″": 1,
           "−1 annule f′ (c'est un point critique) : le point d'inflexion annule f″": -1}'''),
    ]),
    Paper("5.5", "Point selle : x² − y² vu dans deux directions", [
        ("a", "the gradient at (0, 0), a list", "[0, 0]", ""),
        ("b", "[second derivative along x, along y]", "[2, -2]",
         r'''mistakes={"la dérivée seconde de t² vaut 2, pas 1": [1, -1],
           "l'ordre demandé : le long de l'axe des x d'abord": [-2, 2]}'''),
        ("c", "second derivative along theta = 30 degrees", "2 * np.cos(np.radians(60))",
         r'''decimals=1, mistakes={"cos² θ − sin² θ ne vaut pas cos θ : cherche la bonne formule de trigonométrie (0B)": 2 * np.cos(np.radians(30)),
                     "c'est le coefficient de t² : la dérivée seconde de a t² vaut 2a": np.cos(np.radians(60)),
                     "ta dérivée seconde est nulle : refais le développement de f(t cos θ, t sin θ), il reste un terme en t²": 0.0,
                     "ta calculatrice est en radians : passe-la en degrés": 2 * np.cos(60)}'''),
        ("d", "the angle in [0, 90] degrees where it is zero", "45",
         r'''mistakes={"ta réponse ressemble à un angle en radians : on le demande en degrés": 1}'''),
        ("e", "the point after two steps from (0.5, 0.1) with eta = 0.25, a list", "saddle_steps_55((0.5, 0.1), 0.25, 2)",
         r'''decimals=3, mistakes={"∂f/∂y = −2y : la descente retranche η × (−2y), donc elle MULTIPLIE y par 1 + 2η": saddle_steps_55((0.5, 0.1), 0.25, 2, flipped_y=True),
           "c'est après un seul pas : on en demande deux": saddle_steps_55((0.5, 0.1), 0.25, 1)}'''),
        ("f", "f at that point, 3 decimals", "(lambda p: p[0] ** 2 - p[1] ** 2)(saddle_steps_55((0.5, 0.1), 0.25, 2))",
         r'''decimals=3, mistakes={"c'est f au point de départ : on demande f après les deux pas": 0.5 ** 2 - 0.1 ** 2,
                     "c'est f au point d'une descente qui multiplie y par 0,5 : corrige d'abord e)": 0.125 ** 2 - 0.025 ** 2,
                     "c'est f après un seul pas : on en demande deux": 0.25 ** 2 - 0.15 ** 2}'''),
        ("g", "the point after three steps from (0.5, 0), a list", "saddle_steps_55((0.5, 0.0), 0.25, 3)",
         r'''decimals=4, mistakes={"c'est après deux pas : on en demande trois": saddle_steps_55((0.5, 0.0), 0.25, 2)}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes exercices papier (✏️ 5.1 à 5.5)\n\n"
               "Fais d'abord les exercices ✏️ de `02_exercices.md` **sur papier**, dans ta copie de "
               "`06_mes_reponses.md`. Reporte ensuite chaque réponse ici : **la valeur** que tu as trouvée "
               "(par exemple `42`, `0.125`, `[1, 2, 3]` ou `True`), pas l'expression Python, sinon tu ne vérifies "
               "rien. En Python, le séparateur décimal est un **point** (`0.125`) ; `0,125` sans guillemets "
               "serait un couple de deux nombres. Arrondis comme l'énoncé le demande, et seulement à la fin du "
               "calcul ; quand l'énoncé dit « valeurs exactes », donne-les sans arrondir. Les réponses pas encore "
               "remplies affichent ⏳. "
               "Les exercices ∂ 5.6 et 5.7 se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Part A: numerical derivatives, the step h, the extrema of the solar cycles (5.11 to 5.14)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools for the notebook exercises (parts A to D)
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch


def verdict(ex_id, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message."""
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")


def filled(*values):
    """True when none of the values is still `...` (or None): the answer has been written."""
    return all(value is not ... and value is not None for value in values)


def run_calculus_tests(keyword, impl="learner"):
    """Run the tests of mylearn.calculus selected by `keyword` (on YOUR code by default)."""
    command = [sys.executable, "-m", "pytest", "tests/test_ch05_calculus.py", "-k", keyword, "-q",
               "-p", "no:cacheprovider", "--color=no", "-rf", "--tb=no"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "1000"})   # long lines: the reason of each failure
    lines = result.stdout.strip().splitlines()
    failed = [line for line in lines if line.startswith("FAILED ")]
    for line in failed[:8]:                                  # the test, then the reason of its failure
        name, _, reason = line.removeprefix("FAILED tests/test_ch05_calculus.py::").partition(" - ")
        print(f"❌ {name}\n   {reason[:800]}")
    if len(failed) > 8:
        print(f"   ... and {len(failed) - 8} other failed test(s)")
    print("pytest:", lines[-1] if lines else result.stderr.strip()[-300:])


def error_name(func, *args, **kwargs):
    """Name of the exception raised by func(*args, **kwargs), or "no error"."""
    try:
        func(*args, **kwargs)
    except NotImplementedError:
        raise
    except Exception as error:  # noqa: BLE001 - we want the name of whatever is raised
        return type(error).__name__
    return "no error"


def rosenbrock(v):
    """Rosenbrock function (a = 1, b = 100) at the point v = [x, y]: a float, minimum 0 at (1, 1)."""
    return float(wb.synth.rosenbrock(v[0], v[1]))


def rosenbrock_gradient(v):
    """Its exact gradient at v = [x, y], from the formula (∂ 5.7): the array [df/dx, df/dy]."""
    return np.array(wb.synth.rosenbrock_grad(v[0], v[1]))


print("tools ready: rosenbrock(v) =", rosenbrock([-1.5, 2.0]), "at (-1.5, 2)")'''

MYLEARN_HOWTO = ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre `mon_travail/mylearn/calculus.py` "
                 "(créé par `python tools/start_chapter.py 5`), lis la docstring de chaque fonction, remplace les "
                 "`raise NotImplementedError(...)` par ton code et **enregistre**. NumPy est permis (`np.asarray`, "
                 "`np.ndim`, `np.array`, `np.zeros_like`, `np.eye`, `np.linalg.norm`…), SciPy et PyTorch non : "
                 "`scipy.signal.argrelextrema`, `torch.autograd` et `torch.optim.SGD` sont les **oracles** des tests. "
                 "Écris une petite fonction d'aide, par exemple `_check_step(h)`, qui lève une `ValueError` si `h` "
                 "n'est pas strictement positif : les trois fonctions qui prennent un pas `h` l'appelleront. "
                 "La cellule de vérification recharge ta librairie, vérifie quelques valeurs, puis lance les tests de tes "
                 "fonctions ; `python -m pytest tests/test_ch05_calculus.py -q` les lance tous dans un terminal.")

MYLEARN_SHORT = ("> **mylearn** : dans `mon_travail/mylearn/calculus.py`, mêmes règles qu'en 5.11 (NumPy permis, "
                 "SciPy et PyTorch non). Enregistre, puis relance la cellule de vérification.")

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="5")   # reload your saved file\n'

F_12 = r'''def f_12(x):
    """x³ - 2x, written with products only: the same bits on every computer."""
    return x * x * x - 2 * x


def df_12(x):
    """Its exact derivative, 3x² - 2."""
    return 3 * x * x - 2


X_12 = 2.2                     # f'(2.2) = 12.52
KS_12 = np.arange(1, 16)       # the steps h = 10**-k, for k = 1, 2, ..., 15'''

EXPERIMENT_12 = r'''steps_12 = 10.0 ** -KS_12
central_12 = [abs((f_12(X_12 + h) - f_12(X_12 - h)) / (2 * h) - df_12(X_12)) for h in steps_12]
forward_12 = [abs((f_12(X_12 + h) - f_12(X_12)) / h - df_12(X_12)) for h in steps_12]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.loglog(steps_12, central_12, "o-", label="central difference")
ax.loglog(steps_12, forward_12, "s-", label="forward difference")
ax.invert_xaxis()                                       # h decreases from left to right
ax.set(xlabel="step h (log scale, decreasing)", ylabel="error |estimate - f'(2.2)| (log scale)")
ax.legend()
plt.show()
print(f"your predictions: a) k = {prediction_5_12a}   b) k = {prediction_5_12b}   "
      f"c) {prediction_5_12c}   d) {prediction_5_12d}")'''

PLOT_13 = r'''fig, ax = plt.subplots(figsize=(7.5, 4.2))
steps_13 = 10.0 ** -KS_12
eps_13 = np.finfo(float).eps
ax.loglog(steps_13, central_13, "o-", label="central difference (your function)")
ax.loglog(steps_13, forward_13, "s-", label="forward difference (your function)")
ax.loglog(steps_13, steps_13 ** 2 + eps_13 / steps_13, "--", color="C0", alpha=0.6, label="model h² + ε/h")
ax.loglog(steps_13, steps_13 + eps_13 / steps_13, "--", color="C1", alpha=0.6, label="model h + ε/h")
ax.invert_xaxis()
ax.set(xlabel="step h (log scale, decreasing)", ylabel="error (log scale)")
ax.legend(fontsize=8)
plt.show()'''

PLOT_14 = r'''fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(years_14, smooth_14, lw=1.2, label="13-month centred mean")
ax.plot(years_14[minima_14], smooth_14[minima_14], "v", label="minima (order 60)")
ax.plot(years_14[maxima_14], smooth_14[maxima_14], "^", label="maxima (order 60)")
ax.set(xlabel="year", ylabel="sunspots")
ax.legend(loc="upper left", fontsize=8)
plt.show()'''

PART_A = Part("A", "Dériver sur ordinateur : le pas h et les cycles solaires",
              "Fiche §5.2 et §5.3, puis « au-delà du livre » (1) et (2). Tu écris tes premières fonctions de "
              "`mylearn.calculus` (les dérivées numériques), tu prévois puis mesures l'effet du pas $h$, et tu repères "
              "les cycles solaires avec les extrema locaux. La cellule ci-dessous charge les outils de tout le "
              "notebook : exécute-la d'abord.",
              given=PART_A_GIVEN, exercises=[
    Ex("5.11", "🔨", 2, 25, "Dérivées numériques : première et seconde",
       "écrire la pente par différences finies (avant, arrière, centrée) et la dérivée seconde, avec leurs contrôles.",
       "Ex 5.1 · fiche §5.3, au-delà du livre (1)", thread="synthétique", tracks="R, M, C", mylearn="calculus.py",
       body=MYLEARN_HOWTO + r"""

Écris `numerical_derivative(f, x, h=1e-5, method="central")` et `second_derivative(f, x, h=1e-4)` (lis leurs docstrings) :
- lève une `ValueError` si `h` n'est pas strictement positif, ou si `method` n'est pas `"central"`, `"forward"` ou `"backward"` ;
- applique la formule demandée : elle marche telle quelle sur un tableau de points, pourvu que `f` accepte les tableaux (comme `np.sin`) ;
- pour un `x` scalaire (`np.ndim(x) == 0`), renvoie un `float` Python ; sinon (un tableau ou une liste), convertis d'abord `x` avec `np.asarray(x, dtype=float)`, et renvoie un tableau de flottants de la même forme.

Vérifications sur $f(x) = x\,e^{-x}$ (`f_11`), au point $x = 0{,}2$. La cellule de vérification appelle tes fonctions :
a) la pente **centrée**, avec $h = 0{,}01$ ;
b) la pente **avant**, avec le même pas $h = 0{,}01$ ;
c) la dérivée seconde, avec le pas par défaut ;
puis deux contrôles d'erreurs et les tests des deux fonctions.

Dans tes notes : la dérivée exacte est $f'(x) = (1 - x)\,e^{-x}$. Compare-lui a) et b) : de combien chacune s'en écarte-t-elle, avec le même pas ? Même question pour c), avec $f''(x) = (x - 2)\,e^{-x}$.""",
       given=r'''def f_11(x):
    """x e^(-x): works on numbers and on NumPy arrays."""
    return x * np.exp(-x)


X_11 = 0.2''',
       check=RELOAD + r'''with wb.attempt("5.11"):
    wb.check("5.11a", mylearn.calculus.numerical_derivative(f_11, X_11, h=0.01), computed=True)
    wb.check("5.11b", mylearn.calculus.numerical_derivative(f_11, X_11, h=0.01, method="forward"), computed=True)
    wb.check("5.11c", mylearn.calculus.second_derivative(f_11, X_11), computed=True)
    verdict("5.11", error_name(mylearn.calculus.numerical_derivative, f_11, X_11, h=0.0) == "ValueError",
            "un pas nul est refusé (ValueError).",
            "numerical_derivative(f_11, 0.2, h=0.0) doit lever une ValueError : le pas doit être strictement positif.")
    verdict("5.11", error_name(mylearn.calculus.numerical_derivative, f_11, X_11, method="middle") == "ValueError",
            "une méthode inconnue est refusée (ValueError).",
            "numerical_derivative(f_11, 0.2, method=\"middle\") doit lever une ValueError : la méthode est inconnue.")
    run_calculus_tests("test_numerical_derivative_ or test_second_derivative_")''',
       solution=r'''slope_11 = mylearn.calculus.numerical_derivative(f_11, X_11, h=0.01)
forward_11 = mylearn.calculus.numerical_derivative(f_11, X_11, h=0.01, method="forward")
curvature_11 = mylearn.calculus.second_derivative(f_11, X_11)
exact_11, exact2_11 = (1 - X_11) * np.exp(-X_11), (X_11 - 2) * np.exp(-X_11)
print(f"h = 0.01: central {slope_11:.10f} (error {abs(slope_11 - exact_11):.1e}) · forward {forward_11:.10f} "
      f"(error {abs(forward_11 - exact_11):.1e})")
print(f"second derivative {curvature_11:.10f} (error {abs(curvature_11 - exact2_11):.1e})")
print(error_name(mylearn.calculus.numerical_derivative, f_11, X_11, h=0.0),
      error_name(mylearn.calculus.numerical_derivative, f_11, X_11, method="middle"))
run_calculus_tests("test_numerical_derivative_ or test_second_derivative_", impl="ref")''',
       record=r'''wb.record("5.11a", slope_11, decimals=4, mistakes={"c'est f(0,2), la hauteur de la courbe : on demande sa pente": float(f_11(X_11)),
                                                    "tu as divisé par h au lieu de 2h : les deux points de la différence centrée sont à 2h l'un de l'autre": 2 * slope_11,
                                                    "c'est la différence AVANT, (f(x + h) − f(x)) / h : la méthode \"central\" utilise f(x + h) et f(x − h)": forward_11,
                                                    "c'est la différence ARRIÈRE, (f(x) − f(x − h)) / h : la méthode \"central\" utilise f(x + h) et f(x − h)": mylearn.calculus.numerical_derivative(f_11, X_11, h=0.01, method="backward")})
wb.record("5.11b", forward_11, decimals=4, mistakes={"c'est la différence ARRIÈRE, (f(x) − f(x − h)) / h : on demande la différence avant, (f(x + h) − f(x)) / h": mylearn.calculus.numerical_derivative(f_11, X_11, h=0.01, method="backward"),
                                                      "c'est la pente centrée (ou une pente avec un pas trop petit) : on demande la différence avant, avec h = 0,01": slope_11})
wb.record("5.11c", curvature_11, decimals=4, mistakes={"c'est la pente : on demande la dérivée seconde, (f(x + h) − 2 f(x) + f(x − h)) / h²": slope_11,
                                                        "tu as divisé par 2h² : la différence seconde se divise par h²": curvature_11 / 2})''',
       note="Avec le même pas $h = 0{,}01$, la pente centrée s'écarte de la dérivée exacte d'environ $3{,}8 \\times "
            "10^{-5}$, la pente avant d'environ $7{,}3 \\times 10^{-3}$, presque exactement $\\frac{h}{2}\\,|f''(0{,}2)| "
            "\\approx 0{,}005 \\times 1{,}47$ : à pas égal, la différence centrée est environ 200 fois plus précise "
            "(erreur en $h^2$ contre $h$, ∂ 5.6). Avec son pas par défaut, $10^{-5}$, elle est même juste à "
            "$4 \\times 10^{-11}$ près. La dérivée seconde, avec "
            "$h = 10^{-4}$, est juste à environ $5 \\times 10^{-9}$ près : la division par $h^2$ amplifie les "
            "arrondis, d'où un pas par défaut plus grand. La référence est dans `solutions/mylearn_ref/calculus.py` : "
            "lis-la **après** avoir réussi les tests."),

    Ex("5.12", "🔮", 2, 15, "Quel pas h choisir ? Prédire la courbe d'erreur",
       "prévoir comment l'erreur d'une différence finie varie avec le pas $h$, de $10^{-1}$ à $10^{-15}$.",
       "Ex 5.11 · fiche, au-delà du livre (1) · fais ta prédiction **avant** de lire « Au-delà du livre (2) », la "
       "question 7 de ∂ 5.6 et son corrigé",
       thread="synthétique", tracks="M, C", hypothesis=True,
       body=r"""On estime la pente de $f(x) = x^3 - 2x$ (`f_12`) en $x = 2{,}2$, où elle vaut exactement $f'(2{,}2) = 12{,}52$, avec les pas $h = 10^{-k}$ pour $k = 1, 2, \ldots, 15$, par la différence centrée et par la différence avant. Pour chaque pas, on mesure l'erreur $|\text{estimation} - 12{,}52|$.

**Sans rien exécuter**, prévois :
a) `prediction_5_12a` : le $k$ du pas qui donne la plus petite erreur pour la différence **centrée** (un entier de 1 à 15) ;
b) `prediction_5_12b` : le même, pour la différence **avant** ;
c) `prediction_5_12c` : pour la différence centrée, l'erreur avec $h = 10^{-15}$ est-elle plus grande qu'avec $h = 10^{-1}$ ? (`True` ou `False`) ;
d) `prediction_5_12d` : chacune à son meilleur pas, la différence centrée est-elle plus précise que la différence avant ? (`True` ou `False`).

Écris ton hypothèse (cellule 📝), puis tes quatre prédictions. Ensuite seulement, exécute l'**Expérience**, qui trace les deux courbes d'erreur. Rien n'est vérifié automatiquement ici : compare toi-même, puis note ce que tu retiens ; l'exercice suivant mesure la même courbe avec ta fonction.""",
       given=F_12,
       todo=r'''prediction_5_12a = ...   # an integer from 1 to 15
prediction_5_12b = ...   # an integer from 1 to 15
prediction_5_12c = ...   # True or False
prediction_5_12d = ...   # True or False''',
       solution=r'''prediction_5_12a, prediction_5_12b, prediction_5_12c, prediction_5_12d = 5, 8, True, True''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", guarded(EXPERIMENT_12, ["prediction_5_12a", "prediction_5_12b", "prediction_5_12c",
                                               "prediction_5_12d"],
                               "⏳ Ex 5.12 : écris d'abord tes quatre prédictions, puis relance cette cellule.")),
              ("todo_md", "📝 **Ce que je retiens** (comparaison avec mes prédictions ; pourquoi la courbe "
                          "remonte-t-elle à droite ? pourquoi pas au même endroit pour les deux différences ?) : …"),
              ("solution_md", "**Ce qu'il faut retenir.** Les deux courbes descendent d'abord en ligne droite : "
                              "c'est l'erreur de troncature, en $h^2$ pour la différence centrée (pente 2 en échelle "
                              "log-log) et en $h$ pour la différence avant (pente 1). Puis elles remontent, de façon "
                              "irrégulière : c'est l'erreur d'arrondi, qui grandit comme $\\varepsilon/h$. Le meilleur "
                              "pas est vers $h = 10^{-5}$ pour la différence centrée et vers $10^{-8}$ pour la "
                              "différence avant : la troncature de la différence avant diminue plus lentement, elle "
                              "reste plus longtemps la plus grande des deux erreurs. À son meilleur pas, la "
                              "différence centrée est plus de cent fois plus précise. Et un pas minuscule est un "
                              "désastre : avec $h = 10^{-15}$, l'erreur dépasse celle de $h = 0{,}1$.")],
       note="Le dilemme du pas est général : l'erreur totale est la somme d'une troncature qui baisse avec $h$ et d'un "
            "arrondi qui monte. C'est pourquoi `numerical_derivative` prend $h = 10^{-5}$ par défaut (le bon ordre de "
            "grandeur pour la différence centrée, à une fonction près), et pourquoi les *gradient checks* du ch. 18 "
            "se font en `float64`."),

    Ex("5.13", "🔬", 2, 25, "Erreur de troncature contre erreur d'arrondi",
       "mesurer les deux erreurs d'une différence finie, et retrouver le meilleur pas en minimisant un modèle de "
       "l'erreur.",
       "Ex 5.12 · ∂ 5.6 · fiche, au-delà du livre (2)", thread="synthétique", tracks="M, C",
       body=r"""Mesure maintenant la courbe de 5.12 avec **ta** fonction `numerical_derivative`. Écris `errors_13(f, df, x, ks, method)`, qui renvoie le tableau des erreurs $|\text{pente estimée} - f'(x)|$ pour les pas $h = 10^{-k}$, $k$ parcourant `ks` (`df` est la dérivée exacte, une fonction ; attention, `10 ** -k` refuse un exposant entier négatif de NumPy : écris `10.0 ** -k`). La cellule de vérification l'appelle sur `f_12` en $x = 2{,}2$, pour $k = 1, \ldots, 15$, et en tire :
a) le $k$ du meilleur pas, pour la différence centrée ;
b) le même, pour la différence avant ;
c) la **pente** de la courbe d'erreur centrée entre $h = 10^{-1}$ et $h = 10^{-3}$, en coordonnées logarithmiques : $\frac{\log_{10} e(10^{-1}) - \log_{10} e(10^{-3})}{\log_{10} 10^{-1} - \log_{10} 10^{-3}}$ ;
d) la même pente, pour la différence avant ;
puis elle trace tes deux courbes.

Ensuite, un modèle simple de l'erreur totale. Avec un pas $h$, l'erreur de troncature de la différence centrée se comporte comme $h^2$, et son erreur d'arrondi comme $\frac{\varepsilon}{h}$, où $\varepsilon \approx 2{,}2 \times 10^{-16}$ (`np.finfo(float).eps`) ; pour la différence avant, comme $h$ et $\frac{\varepsilon}{h}$ (∂ 5.6, question 7 ; on oublie les constantes). Le meilleur pas minimise la somme : annule sa dérivée par rapport à $h$ (§5.3).
e) `log_h_central_13` : $\log_{10}$ du pas qui minimise $h^2 + \frac{\varepsilon}{h}$ (2 décimales) ;
f) `log_h_forward_13` : $\log_{10}$ du pas qui minimise $h + \frac{\varepsilon}{h}$ (2 décimales).

Dans tes notes : que disent les pentes c) et d) de la façon dont l'erreur de troncature diminue ? Le modèle de e) et f) retrouve-t-il les meilleurs pas mesurés en a) et b) ? Pourquoi la partie des petits pas est-elle si irrégulière ?""",
       todo=r'''def errors_13(f, df, x, ks, method):
    """|numerical_derivative(f, x, h=10**-k, method) - df(x)| for every k of ks, with YOUR function (an array)."""
    raise NotImplementedError("errors_13() is not written yet")


log_h_central_13 = ...   # e)
log_h_forward_13 = ...   # f)''',
       check=RELOAD + r'''with wb.attempt("5.13"):
    central_13 = np.asarray(errors_13(f_12, df_12, X_12, KS_12, "central"), dtype=float)
    forward_13 = np.asarray(errors_13(f_12, df_12, X_12, KS_12, "forward"), dtype=float)
    wb.check("5.13a", int(KS_12[np.argmin(central_13)]), computed=True)
    wb.check("5.13b", int(KS_12[np.argmin(forward_13)]), computed=True)
    wb.check("5.13c", (np.log10(central_13[0]) - np.log10(central_13[2])) / 2, computed=True)
    wb.check("5.13d", (np.log10(forward_13[0]) - np.log10(forward_13[2])) / 2, computed=True)
''' + indent(PLOT_13) + r'''
wb.check("5.13e", log_h_central_13)
wb.check("5.13f", log_h_forward_13)''',
       solution=r'''def errors_13(f, df, x, ks, method):
    """|numerical_derivative(f, x, h=10**-k, method) - df(x)| for every k of ks, with YOUR function (an array)."""
    return np.array([abs(mylearn.calculus.numerical_derivative(f, x, h=10.0 ** -k, method=method) - df(x)) for k in ks])


central_13 = errors_13(f_12, df_12, X_12, KS_12, "central")
forward_13 = errors_13(f_12, df_12, X_12, KS_12, "forward")
best_central_13, best_forward_13 = int(KS_12[np.argmin(central_13)]), int(KS_12[np.argmin(forward_13)])
slope_central_13 = float((np.log10(central_13[0]) - np.log10(central_13[2])) / 2)
slope_forward_13 = float((np.log10(forward_13[0]) - np.log10(forward_13[2])) / 2)
eps = np.finfo(float).eps
log_h_central_13 = float(np.log10((eps / 2) ** (1 / 3)))    # 2h - eps/h² = 0  ->  h³ = eps / 2
log_h_forward_13 = float(np.log10(np.sqrt(eps)))           # 1 - eps/h² = 0   ->  h² = eps
print(best_central_13, best_forward_13, slope_central_13, slope_forward_13, log_h_central_13, log_h_forward_13)
''' + PLOT_13,
       record=r'''wb.record("5.13a", best_central_13, mistakes={"15 donne la plus GRANDE erreur (ou la plus négative, si tu as oublié la valeur absolue) : on cherche la plus petite erreur |estimation − f'(x)|": 15})
wb.record("5.13b", best_forward_13, mistakes={"c'est le meilleur pas de la différence CENTRÉE : errors_13 doit transmettre `method` à numerical_derivative": best_central_13,
                                              "15 donne la plus GRANDE erreur (ou la plus négative, si tu as oublié la valeur absolue) : on cherche la plus petite erreur |estimation − f'(x)|": 15})
wb.record("5.13c", slope_central_13, decimals=4)
wb.record("5.13d", slope_forward_13, decimals=4, mistakes={"c'est la pente de la différence CENTRÉE : errors_13 doit transmettre `method` à numerical_derivative": slope_central_13})
wb.record("5.13e", log_h_central_13, decimals=2, mistakes={"c'est log₁₀(ε^(1/3)) : la dérivée de h² + ε/h est 2h − ε/h², elle s'annule pour h³ = ε/2": float(np.log10(eps ** (1 / 3))),
                                                            "c'est log₁₀(√ε), le meilleur pas de la différence AVANT (f) : ici, la troncature est en h²": log_h_forward_13})
wb.record("5.13f", log_h_forward_13, decimals=2, mistakes={"c'est log₁₀(√(2ε)) : la dérivée de h + ε/h est 1 − ε/h², elle s'annule pour h² = ε, sans facteur 2": float(np.log10(np.sqrt(2 * eps))),
                                                            "c'est log₁₀(ε) : le meilleur pas n'est pas ε, il annule la dérivée 1 − ε/h²": float(np.log10(eps)),
                                                            "c'est le meilleur pas de la différence CENTRÉE (e) : ici, la troncature est en h": log_h_central_13})''',
       note="Entre $h = 10^{-1}$ et $10^{-3}$, la courbe centrée a une pente de 2 : diviser $h$ par 10 divise l'erreur "
            "par 100 (pour $x^3 - 2x$, l'erreur de troncature vaut même exactement $h^2$, ∂ 5.6). La courbe avant a une "
            "pente de 1 (un peu plus, à cause du terme en $h^2$). Le modèle place le meilleur pas en $10^{-5{,}32}$ et "
            "$10^{-7{,}83}$ : $k = 5$ et $k = 8$ une fois arrondis, exactement les meilleurs pas mesurés. Les "
            "constantes oubliées (ici $|f'''|/6 = 1$ et $|f(2{,}2)| \\approx 6{,}2$) déplacent un peu l'optimum, mais "
            "pas son ordre de grandeur. Du côté des petits pas, l'erreur d'arrondi dépend des derniers bits de "
            "$f(x + h)$ et $f(x - h)$ : elle est irrégulière, presque aléatoire, ce qui rend ces pas inutilisables."),

    Ex("5.14", "🔨", 2, 30, "Les maxima des cycles solaires",
       "repérer les extrema locaux d'une série échantillonnée, et mesurer l'effet du lissage et de la largeur de la zone "
       "comparée.",
       "Ex 5.11 · Ex 1.12 (taches solaires) · fiche §5.2, §5.3 · livre §5.3 (figures 5.5 et 5.6)",
       thread="taches solaires", tracks="C", mylearn="calculus.py",
       body=MYLEARN_SHORT + r"""

Au ch. 1 (1.12), un outil fourni repérait les sommets de la série lissée des taches solaires. Écris le tien, `find_local_extrema(y, order=1)` (lis sa docstring) : l'échantillon $i$ est un minimum local s'il est **strictement** plus petit que chacun de ses voisins, jusqu'à `order` échantillons de chaque côté (un maximum : strictement plus grand). Les voisins qui tomberaient hors du tableau sont ignorés, mais le premier et le dernier échantillon ne sont jamais des extrema. Renvoie deux tableaux d'indices entiers, triés : les minima, puis les maxima. Lève une `ValueError` si `y` n'est pas à une dimension, ou si `order` n'est pas un entier au moins égal à 1.

C'est la définition « par voisinage » de la fiche (§5.3), pour une courbe échantillonnée : `order` règle la largeur du voisinage. Avec `order=1`, on retrouve les points où la série change de sens, ceux où s'arrête la marche du livre (figures 5.5 et 5.6), sauf sur un palier de valeurs égales (question e). Les données : `raw_14`, les nombres mensuels de taches depuis 1749, et `smooth_14`, leur moyenne mobile **centrée** sur 13 mois (le mois, les 6 précédents et les 6 suivants), sans ses valeurs manquantes ; `dates_14` donne `[année, mois]` pour chaque valeur de `smooth_14`, et `years_14` l'année décimale. La cellule de vérification appelle ta fonction :
a) le nombre de maxima locaux de la série **brute**, avec `order=1` ;
b) le nombre de minima de la série lissée, avec `order=6` ;
c) avec `order=60` (cinq ans de chaque côté), sur la série lissée : `[année, mois]` du dernier minimum, puis du dernier maximum (un tableau 2 × 2) ;
d) avec `order=60`, la plus longue durée entre deux minima successifs, en années ;
puis les tests de `find_local_extrema`, et une figure.

e) `zero_months_14` : le nombre de mois où la série lissée vaut **exactement** 0.

Dans tes notes : la durée de d) correspond-elle à un vrai cycle ? Que s'est-il passé à cette époque, et pourquoi ta fonction ne voit-elle pas le minimum qui manque ? Pourquoi le dernier maximum de c) est-il encore incertain ?""",
       given=r'''sun_14 = wb.datasets.load_sunspots(definitive_only=True)        # monthly sunspot numbers since 1749 (ch. 1)
raw_14 = sun_14["sunspots"].to_numpy()
centred_14 = sun_14["sunspots"].rolling(13, center=True).mean()  # the month, the 6 before and the 6 after
kept_14 = centred_14.notna().to_numpy()                          # the first 6 and the last 6 months have no value
smooth_14 = centred_14.to_numpy()[kept_14]
years_14 = sun_14["decimal_year"].to_numpy()[kept_14]
dates_14 = sun_14[["year", "month"]].to_numpy()[kept_14]         # [year, month] of every value of smooth_14
print(f"{len(raw_14)} months; {len(smooth_14)} smoothed values, from {dates_14[0].tolist()} to {dates_14[-1].tolist()}")''',
       todo=r'''zero_months_14 = ...   # e)''',
       check=RELOAD + r'''with wb.attempt("5.14"):
    wb.check("5.14a", len(mylearn.calculus.find_local_extrema(raw_14)[1]), computed=True)
    wb.check("5.14b", len(mylearn.calculus.find_local_extrema(smooth_14, order=6)[0]), computed=True)
    minima_14, maxima_14 = mylearn.calculus.find_local_extrema(smooth_14, order=60)
    wb.check("5.14c", [dates_14[minima_14[-1]], dates_14[maxima_14[-1]]], computed=True)
    wb.check("5.14d", np.max(np.diff(years_14[minima_14])), computed=True)
    run_calculus_tests("test_find_local_extrema_")
''' + indent(PLOT_14) + r'''
wb.check("5.14e", zero_months_14)''',
       solution=r'''raw_minima_14, raw_maxima_14 = mylearn.calculus.find_local_extrema(raw_14)
minima_6_14, maxima_6_14 = mylearn.calculus.find_local_extrema(smooth_14, order=6)
minima_14, maxima_14 = mylearn.calculus.find_local_extrema(smooth_14, order=60)
last_14 = [dates_14[minima_14[-1]].tolist(), dates_14[maxima_14[-1]].tolist()]
gaps_14 = np.diff(years_14[minima_14])
longest_14 = float(gaps_14.max())
zero_months_14 = int(np.sum(smooth_14 == 0))
print(len(raw_maxima_14), len(minima_6_14), last_14, longest_14, zero_months_14)
print("the longest gap goes from", years_14[minima_14][np.argmax(gaps_14)], "to", years_14[minima_14][np.argmax(gaps_14) + 1])
print("months at exactly 0:", dates_14[smooth_14 == 0].tolist())
run_calculus_tests("test_find_local_extrema_", impl="ref")
''' + PLOT_14,
       record=r'''def non_strict_minima_14(y, order):
    """The minima with <= instead of <: a plateau gives several minima (the wrong definition, for the mistakes)."""
    return np.array([i for i in range(1, len(y) - 1)
                     if all(y[i] <= y[j] for j in range(max(0, i - order), min(len(y), i + order + 1)) if j != i)])


def non_strict_maxima_14(y, order):
    """The maxima with >= instead of >: the wrong definition, for the mistakes."""
    return np.array([i for i in range(1, len(y) - 1)
                     if all(y[i] >= y[j] for j in range(max(0, i - order), min(len(y), i + order + 1)) if j != i)])


wb.record("5.14a", len(raw_maxima_14), mistakes={"c'est le nombre de MINIMA (le premier tableau renvoyé) : on demande les maxima": len(raw_minima_14),
                                                 "tes comparaisons ne sont pas strictes : deux mois égaux côte à côte ne sont pas des maxima (relis la docstring)": len(non_strict_maxima_14(raw_14, 1))})
wb.record("5.14b", len(minima_6_14), mistakes={"c'est le nombre de MAXIMA (le second tableau renvoyé) : on demande les minima": len(maxima_6_14),
                                               "c'est le nombre de minima avec order=1 : ta fonction doit comparer chaque point à ses `order` voisins de chaque côté": len(mylearn.calculus.find_local_extrema(smooth_14)[0]),
                                               "tes comparaisons ne sont pas strictes : un plateau ne donne aucun minimum (relis la docstring)": len(non_strict_minima_14(smooth_14, 6))})
wb.record("5.14c", last_14, mistakes={"l'ordre demandé : le dernier minimum, puis le dernier maximum": last_14[::-1]})
wb.record("5.14d", longest_14, decimals=4, mistakes={"tes comparaisons ne sont pas strictes : sur un plateau, aucun point n'est strictement plus petit que ses voisins, il n'y a pas d'extremum (relis la docstring)": float(np.max(np.diff(years_14[non_strict_minima_14(smooth_14, 60)])))})
wb.record("5.14e", zero_months_14, mistakes={"c'est le nombre de mois BRUTS à 0 : on demande la série lissée, smooth_14": int(np.sum(raw_14 == 0))})''',
       note="Sur la série brute, le bruit crée des centaines de « sommets » : un point plus haut que ses deux voisins "
            "immédiats n'a rien d'un maximum du cycle. Le lissage et une zone plus large (`order=60`, cinq ans de chaque "
            "côté) ne gardent que les vrais extrema : le premier minimum tombe en 1755, au début du cycle que l'on "
            "numérote 1, et le dernier en décembre 2019, au début du cycle 25, en cours. Le dernier maximum (octobre 2024) "
            "n'a que 11 mois lissés après lui : les mois à venir pourraient encore le dépasser. La durée de 25 ans de d) "
            "n'est pas un cycle : c'est le **minimum de Dalton**, où la série lissée vaut exactement 0 pendant neuf mois "
            "de 1810. Sur ce plateau, aucun mois n'est strictement plus bas que ses voisins : la définition stricte ne "
            "voit pas de minimum, et deux cycles fusionnent. Pour le voir, il faudrait traiter les plateaux (par "
            "exemple, garder le milieu d'une suite de valeurs égales), ce que fait `scipy.signal.find_peaks` avec "
            "les maxima."),
])

# ---------------------------------------------------------------------------
# Part B: numerical gradients, a bug, level lines (5.15 to 5.17)
# ---------------------------------------------------------------------------
COLLEAGUE_16 = r'''def gradient_colleague(f, x, h=1e-5):
    """The colleague's numerical gradient: central differences, one coordinate after the other."""
    x = np.asarray(x)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x[i] += h
        f_plus = f(x)
        x[i] -= 2 * h
        f_minus = f(x)
        x[i] += h
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


print("gradient at (0.1, 0.2):", gradient_colleague(rosenbrock, np.array([0.1, 0.2])))
start_16 = np.array([-1, 1])                    # the starting point of a descent, written without decimal points
print("gradient at (-1, 1):   ", gradient_colleague(rosenbrock, start_16))'''

MAP_17 = r'''def g_17(x, y):
    """x³ - 3x - y³ + 3y (works on grids of points)."""
    return x ** 3 - 3 * x - y ** 3 + 3 * y


POINTS_17 = {"A": (1.0, 1.0), "B": (1.0, -1.0), "C": (-1.0, 1.0), "D": (0.0, 0.0), "E": (-1.0, -1.0), "F": (1.8, -0.2)}
X_17, Y_17 = np.meshgrid(np.linspace(-2.2, 2.2, 201), np.linspace(-2.2, 2.2, 201))


def draw_map_17(ax):
    """Filled level lines of g_17 (light = high), the lines themselves and the six points; returns the colour map."""
    colours = ax.contourf(X_17, Y_17, g_17(X_17, Y_17), levels=24, cmap="viridis")
    ax.contour(X_17, Y_17, g_17(X_17, Y_17), levels=colours.levels[::2], colors="white", linewidths=0.5)
    for letter, (x, y) in POINTS_17.items():
        ax.plot(x, y, "o", color="red")
        ax.annotate(letter, (x, y), xytext=(6, 6), textcoords="offset points", color="red", fontsize=13,
                    weight="bold")
    ax.set_aspect("equal")
    ax.set(xlabel="x", ylabel="y")
    return colours


fig, ax = plt.subplots(figsize=(6.5, 5.5))
fig.colorbar(draw_map_17(ax), ax=ax, label="g(x, y)")
plt.show()
print("F =", POINTS_17["F"])'''

ARROWS_17 = r'''fig, ax = plt.subplots(figsize=(6.5, 5.5))
draw_map_17(ax)
GX_17, GY_17 = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
ax.quiver(GX_17, GY_17, 3 * GX_17 ** 2 - 3, -3 * GY_17 ** 2 + 3, color="white")   # the gradient at each point
ax.set_title("the gradient of g, drawn on its level lines")
plt.show()'''

PART_B = Part("B", "Le gradient : le calculer, le déboguer, le lire sur une carte",
              "Fiche §5.4. Tu écris le gradient numérique et tu l'essaies sur la vallée de Rosenbrock et sur une "
              "matrice de poids, tu débogues la version d'un collègue, puis tu lis le gradient sur une carte de lignes "
              "de niveau.",
              exercises=[
    Ex("5.15", "🔨", 2, 25, "numerical_gradient sur la vallée de Rosenbrock",
       "écrire le gradient numérique d'une fonction de plusieurs variables, pour un point de forme quelconque, sans "
       "toucher au point reçu.",
       "Ex 5.11 · ✏️ 5.2 · fiche §5.4 (calculer un gradient sur ordinateur)", thread="Rosenbrock",
       tracks="R, M, C", mylearn="calculus.py",
       body=MYLEARN_SHORT + r"""

Écris `numerical_gradient(f, x, h=1e-5)` (lis sa docstring) : pour chaque position $i$ de `x`, la différence centrée $\frac{f(\mathbf{x} + h\,\mathbf{e}_i) - f(\mathbf{x} - h\,\mathbf{e}_i)}{2h}$, où $\mathbf{e}_i$ vaut 1 à la position $i$ et 0 ailleurs.
- Travaille sur une **copie en flottants** : `point = np.array(x, dtype=float)` (contrairement à `np.asarray`, `np.array` copie toujours). Le tableau de l'appelant ne doit jamais changer, pas même pendant les appels à `f`.
- `x` peut avoir n'importe quelle forme : une matrice de poids, par exemple. `point.reshape(-1)` en donne une vue à plat (modifier la vue modifie `point`), et `np.zeros_like(point)` prépare un gradient de la même forme.
- Pour chaque position : garde l'ancienne valeur, pose `ancienne + h`, évalue `f`, pose `ancienne - h`, évalue `f`, puis **remets l'ancienne valeur** telle quelle (ne la recalcule pas : `+ h`, `- 2 * h` puis `+ h` ne redonnent pas toujours exactement le même nombre).

La cellule de vérification appelle ta fonction :
a) le gradient de la fonction de Rosenbrock (`rosenbrock`, $a = 1$, $b = 100$) au point $(-0{,}5 ;\ 0{,}5)$ ; elle le compare au gradient exact, `rosenbrock_gradient` ;
b) le gradient de la loss $L(W) = \sum_i \big((W\mathbf{v})_i - t_i\big)^2$ (`loss_15`) par rapport à la **matrice** $W$, au point `W_15` (une matrice 2 × 3) ;
puis le contrôle que `W_15` n'a pas changé, et les tests de `numerical_gradient`.

Dans tes notes : combien d'appels à `loss_15` a-t-il fallu pour b) ? Vérifie b) à la main avec la formule $\nabla_W L = 2\,(W\mathbf{v} - \mathbf{t})\,\mathbf{v}^\top$ (0B).""",
       given=r'''W_15 = np.array([[1.0, 0.0, -1.0],
                 [0.5, 2.0, 0.0]])
V_15 = np.array([1.0, 2.0, 3.0])
T_15 = np.array([0.0, 1.0])


def loss_15(W):
    """Sum of the squared errors of W @ V_15 against the targets T_15: a function of the matrix W."""
    return float(np.sum((W @ V_15 - T_15) ** 2))''',
       check=RELOAD + r'''with wb.attempt("5.15"):
    grad_15 = np.asarray(mylearn.calculus.numerical_gradient(rosenbrock, [-0.5, 0.5]), dtype=float)
    wb.check("5.15a", grad_15, computed=True)
    exact_15 = rosenbrock_gradient([-0.5, 0.5])
    verdict("5.15", grad_15.shape == (2,) and np.allclose(grad_15, exact_15, rtol=1e-6),
            f"ton gradient numérique colle au gradient exact {exact_15.tolist()}.",
            f"ton gradient numérique s'écarte du gradient exact {exact_15.tolist()}.")
    saved_15 = W_15.copy()
    wb.check("5.15b", mylearn.calculus.numerical_gradient(loss_15, W_15), computed=True)
    verdict("5.15", np.array_equal(W_15, saved_15), "W_15 n'a pas changé.",
            "ta fonction a modifié W_15 : travaille sur une copie, np.array(x, dtype=float).")
    W_15[...] = saved_15                                      # put W_15 back, in case it was modified
    run_calculus_tests("test_numerical_gradient_")''',
       solution=r'''grad_15 = mylearn.calculus.numerical_gradient(rosenbrock, [-0.5, 0.5])
calls_15 = []
grad_W_15 = mylearn.calculus.numerical_gradient(lambda W: calls_15.append(1) or loss_15(W), W_15)
by_hand_15 = 2 * np.outer(W_15 @ V_15 - T_15, V_15)
print(grad_15, rosenbrock_gradient([-0.5, 0.5]))
print(grad_W_15.round(6), f"{len(calls_15)} calls of the loss", np.allclose(grad_W_15, by_hand_15), sep="\n")
run_calculus_tests("test_numerical_gradient_", impl="ref")''',
       record=r'''wb.record("5.15a", grad_15, decimals=4, mistakes={"tu as divisé par h au lieu de 2h : les deux points sont à 2h l'un de l'autre": 2 * grad_15,
                                                   "l'ordre est [∂f/∂x, ∂f/∂y]": grad_15[::-1]})
wb.record("5.15b", grad_W_15, decimals=4, mistakes={"tu as divisé par h au lieu de 2h": 2 * grad_W_15})''',
       note="Il faut deux évaluations par coordonnée : 4 pour a), 12 pour b), et $2n$ pour $n$ paramètres (🧮 5.9). Le "
            "gradient de b) a la forme de $W$ : chaque poids reçoit sa dérivée partielle. À la main, "
            "$W\\mathbf{v} = (-2 ;\\ 4{,}5)$, donc $W\\mathbf{v} - \\mathbf{t} = (-2 ;\\ 3{,}5)$ et "
            "$2\\,(W\\mathbf{v} - \\mathbf{t})\\,\\mathbf{v}^\\top$ redonne exactement b). En $(-0{,}5 ;\\ 0{,}5)$, la "
            "pente de Rosenbrock est déjà forte : la vallée $y = x^2$ passe juste en dessous, en $y = 0{,}25$."),

    Ex("5.16", "🐛", 2, 20, "Le gradient qui abîme son entrée",
       "comprendre pourquoi un gradient numérique doit travailler sur une copie en flottants, et corriger la version "
       "d'un collègue.",
       "Ex 5.15 · fiche §5.4 et les pièges classiques", thread="Rosenbrock", tracks="C",
       body=r"""Un collègue a écrit son propre gradient numérique, `gradient_colleague` (cellule ci-dessous). Il l'a essayé sur des points à virgule, comme $(0{,}1 ;\ 0{,}2)$ : les valeurs sont justes. Puis il lance une descente de gradient depuis le point $(-1, 1)$, écrit sans virgule, `np.array([-1, 1])`. La cellule affiche ce que renvoie sa fonction dans les deux cas.

a) `exact_16` : le vrai gradient de Rosenbrock en $(-1, 1)$ (une liste ; avec `rosenbrock_gradient`, ou à la main avec ∂ 5.7) ;
b) `after_16` : que contient `start_16` **après** l'appel de la cellule ? (une liste de deux entiers : affiche-le) ;
c) `changed_16` : crée un point à virgule, `point_16 = np.array([0.1, 0.2])`, et appelle **une fois** `gradient_colleague(rosenbrock, point_16)`. Combien de ses deux coordonnées ont changé ? (un entier ; compare-les aux valeurs d'origine avec `==` : `print` arrondit l'affichage) ;
d) corrige : écris `gradient_fixed_16(f, x, h=1e-5)`, la fonction du collègue en changeant le moins de lignes possible. La vérification l'appelle sur le point entier $(-1, 1)$ (le gradient obtenu est vérifié) et sur un point à virgule, et contrôle qu'aucun des deux points n'est modifié.

Dans tes notes : explique les deux défauts de la fonction du collègue, et pourquoi il ne voyait rien sur ses points à virgule. Le défaut de c) est minuscule : pourquoi est-il quand même dangereux ?""",
       given=COLLEAGUE_16,
       todo=r'''exact_16 = ...     # a) a list
after_16 = ...     # b) a list of two integers
changed_16 = ...   # c) an integer


def gradient_fixed_16(f, x, h=1e-5):
    """The colleague's function, corrected: the same gradient for [-1, 1] as for [-1.0, 1.0], x never modified."""
    raise NotImplementedError("gradient_fixed_16() is not written yet")''',
       check=r'''wb.check("5.16a", exact_16)
wb.check("5.16b", after_16)
wb.check("5.16c", changed_16)
with wb.attempt("5.16"):
    integer_16, decimal_16 = np.array([-1, 1]), np.array([0.1, 0.2])
    wb.check("5.16d", gradient_fixed_16(rosenbrock, integer_16), computed=True)
    gradient_fixed_16(rosenbrock, decimal_16)
    verdict("5.16", np.array_equal(integer_16, [-1, 1]) and np.array_equal(decimal_16, [0.1, 0.2]),
            "les points reçus ne changent plus, pas même d'un arrondi.",
            "ta version modifie encore le point reçu : travaille sur une copie en flottants.")''',
       solution=r'''exact_16 = rosenbrock_gradient([-1.0, 1.0]).tolist()
after_16 = start_16.tolist()
point_16 = np.array([0.1, 0.2])
gradient_colleague(rosenbrock, point_16)
changed_16 = int(np.sum(point_16 != np.array([0.1, 0.2])))
print(exact_16, after_16, changed_16, repr(point_16[0]), repr(point_16[1]))


def gradient_fixed_16(f, x, h=1e-5):
    """The colleague's function, corrected: the same gradient for [-1, 1] as for [-1.0, 1.0], x never modified."""
    x = np.array(x, dtype=float)          # the only change: a float COPY (np.zeros_like(x) is then float too)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x[i] += h
        f_plus = f(x)
        x[i] -= 2 * h
        f_minus = f(x)
        x[i] += h
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


fixed_16 = gradient_fixed_16(rosenbrock, np.array([-1, 1]))
print(fixed_16)''',
       record=r'''wb.record("5.16a", exact_16, mistakes={"c'est le résultat du collègue : calcule le gradient exact, avec rosenbrock_gradient ou ∂ 5.7": [0, 5000000]})
wb.record("5.16b", after_16, mistakes={"regarde vraiment : affiche start_16 après l'appel, il a changé": [-1, 1]})
wb.record("5.16c", changed_16, mistakes={"print arrondit l'affichage : compare avec ==, par exemple point_16 == np.array([0.1, 0.2])": 0})
wb.record("5.16d", fixed_16, decimals=4, mistakes={"le point est encore un tableau d'ENTIERS : chaque décalage de h est tronqué (convertis avec dtype=float)": [0, 5000000]})''',
       note="Deux défauts. (1) `np.asarray(x)` ne copie pas : la fonction modifie le tableau de l'appelant. "
            "(2) Elle garde son type : avec un tableau d'entiers, `x[i] += h` donne $-1 + 10^{-5} = -0{,}99999$, "
            "arrondi vers zéro en l'entier 0 ; le point devient $(0, 0)$, les « dérivées » sont absurdes, "
            "et le gradient lui-même est un tableau d'entiers. Sur des flottants, le calcul est juste, et le tableau "
            "ne bouge que d'un arrondi (`0.09999999999999999` au lieu de `0.1`) : invisible à l'affichage, mais "
            "une descente de gradient qui partage ce tableau avec son appelant s'en trouve modifiée en douce, et un "
            "test d'égalité exacte échoue sans raison apparente. Une seule ligne corrige les deux défauts : "
            "`x = np.array(x, dtype=float)`, une copie en flottants. Garder l'ancienne valeur et la remettre telle "
            "quelle, comme en 5.15, évite en plus la petite dérive de la copie."),

    Ex("5.17", "📈", 2, 20, "Lire des lignes de niveau : où pointe le gradient ?",
       "lire sur une carte de lignes de niveau les extrema, les points selles, la pente et la direction du gradient.",
       "Ex 5.15 · ✏️ 5.2, 5.5 · fiche §5.4 (pente dans une direction, lignes de niveau, points critiques)",
       thread="synthétique", tracks="R, M",
       body=r"""La carte ci-dessous représente $g(x, y) = x^3 - 3x - y^3 + 3y$ par ses **lignes de niveau** : les couleurs claires sont hautes, les sombres basses (voir la barre de couleurs). Six points sont marqués de A à F. **Sans calculer de dérivée** (sauf en f), lis la carte :
a) `max_17` : la lettre du maximum local ;
b) `min_17` : la lettre du minimum local ;
c) `saddles_17` : les lettres des points selles (un ensemble de chaînes, par exemple `{"X", "Y"}`) ;
d) `steeper_17` : en lequel des deux points D et F la pente est-elle la plus forte ? (`"D"` ou `"F"`) ;
e) `uphill_D_17` : en D, vers où la montée est-elle la plus raide ? Une des huit directions de la boussole, le nord en haut : `"N"`, `"NE"`, `"E"`, `"SE"`, `"S"`, `"SO"`, `"O"` (ouest) ou `"NO"` ;
f) `grad_F_17` : enfin, calcule le gradient de $g$ en F (une liste, 2 décimales), avec les coordonnées de F affichées sous la carte.

Une fois tes six réponses écrites, la dernière cellule superpose à la carte les flèches du gradient : regarde leur longueur, leur sens, et leur angle avec les lignes de niveau.""",
       given=MAP_17,
       todo=r'''max_17 = ...        # a) a letter
min_17 = ...        # b) a letter
saddles_17 = ...    # c) a set of letters
steeper_17 = ...    # d) "D" or "F"
uphill_D_17 = ...   # e) "N", "NE", "E", "SE", "S", "SO", "O" or "NO"
grad_F_17 = ...     # f) a list''',
       check=r'''for letter, answer in zip("abcdef", [max_17, min_17, saddles_17, steeper_17, uphill_D_17, grad_F_17]):
    wb.check(f"5.17{letter}", answer)''',
       solution=r'''max_17, min_17, saddles_17, steeper_17, uphill_D_17 = "C", "B", {"A", "E"}, "F", "NO"
x_F, y_F = POINTS_17["F"]
grad_F_17 = [3 * x_F ** 2 - 3, -3 * y_F ** 2 + 3]
print(grad_F_17, "norm at F:", np.hypot(*grad_F_17), "norm at D:", np.hypot(-3, 3))''',
       record=r'''wb.record("5.17a", max_17, mistakes={"c'est le minimum : les couleurs claires sont les hauteurs": "B",
                                      "A est un point selle : la surface y monte dans une direction et descend dans l'autre": "A",
                                      "E est un point selle : la surface y monte dans une direction et descend dans l'autre": "E"})
wb.record("5.17b", min_17, mistakes={"c'est le maximum : les couleurs sombres sont les creux": "C",
                                     "A est un point selle, pas un creux : la surface y monte dans une direction et descend dans l'autre": "A",
                                     "E est un point selle, pas un creux : la surface y monte dans une direction et descend dans l'autre": "E"})
wb.record("5.17c", saddles_17, mistakes={"il y a deux points selles : là où les lignes de niveau se croisent en X": {"A"},
                                         "il y a deux points selles : regarde aussi l'autre croisement en X": {"E"},
                                         "D n'est pas un point critique : les lignes de niveau y sont régulières, le gradient n'y est pas nul": {"A", "E", "D"}})
wb.record("5.17d", steeper_17, mistakes={"les lignes de niveau sont plus serrées en F : la pente y est plus forte": "D"})
wb.record("5.17e", uphill_D_17, mistakes={"c'est la direction de plus grande DESCENTE : le gradient pointe vers la montée": "SE",
                                          "regarde où sont les couleurs claires autour de D : vers le maximum C": "NE",
                                          "c'est la bonne direction, mais écrite en anglais : utilise les initiales françaises (O pour ouest)": "NW",
                                          "le gradient en D a deux composantes non nulles : la direction est une diagonale": "N",
                                          "le gradient en D a deux composantes non nulles : la plus grande montée suit une diagonale": "O"})
wb.record("5.17f", grad_F_17, decimals=2, mistakes={"la dérivée de −y³ est −3y² : ∂g/∂y = −3y² + 3": [6.72, -2.88],
                                                    "n'oublie pas la dérivée de −3x : ∂g/∂x = 3x² − 3": [9.72, 2.88],
                                                    "l'ordre est [∂g/∂x, ∂g/∂y]": [2.88, 6.72]})''',
       after=[("md", "**La carte et ses gradients** : exécute la cellule une fois tes six réponses écrites."),
              ("code", guarded(ARROWS_17, ["max_17", "min_17", "saddles_17", "steeper_17", "uphill_D_17", "grad_F_17"],
                               "⏳ Ex 5.17 : écris d'abord tes six réponses, puis relance cette cellule."))],
       note="Le gradient $(3x^2 - 3,\\ -3y^2 + 3)$ s'annule en quatre points : C $(-1, 1)$, le maximum ($g = 4$) ; "
            "B $(1, -1)$, le minimum ($g = -4$) ; A et E, deux points selles ($g = 0$), où les lignes de niveau se "
            "croisent en X. En D, le gradient vaut $(-3, 3)$ : il pointe vers le nord-ouest, vers C, et sa norme vaut "
            "$\\sqrt{18} \\approx 4{,}24$. En F, il vaut $(6{,}72 ;\\ 2{,}88)$, de norme 7,31 : les lignes y sont plus "
            "serrées. Partout, les flèches coupent les lignes de niveau à angle droit, et elles sont longues là où les "
            "lignes sont serrées (fiche §5.4)."),
])

# ---------------------------------------------------------------------------
# Part C: gradient descent, the learning rate, a saddle point, autograd (5.18 to 5.21)
# ---------------------------------------------------------------------------
PLOT_18 = r'''wb.plot.plot_contour(lambda x, y: (x + 1) ** 2 + 4 * (y - 0.5) ** 2, xlim=(-2, 1.5), ylim=(-0.5, 1.5),
                     path=path_18, log=False, minimum=(-1, 0.5), levels=15, title="15 steps of gradient descent on q")
plt.show()'''

DRAW_19 = r'''steps_19 = np.arange(51)
colours_19 = dict(zip(LRS_19, ["#f58518", "#e45756", "#54a24b", "#000000", "#b279a2"]))   # one colour per lr
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 4.3))
for lr, curve in curves_19.items():
    ax_left.semilogy(steps_19, curve, color=colours_19[lr], label=f"lr = {lr}")
ax_left.set(xlabel="step", ylabel="distance to the minimum (log scale)")
ax_left.legend(fontsize=8)
wb.plot.plot_contour(lambda x, y: x ** 2 + 10 * y ** 2, xlim=(-2.5, 2.5), ylim=(-1.6, 1.6), ax=ax_right, log=False,
                     minimum=(0, 0), resolution=150, levels=15)
for lr in LRS_19[:4]:                                     # lr = 0.11 leaves the frame: see the left panel
    path = mylearn.calculus.gradient_descent(grad_bowl_19, START_19, lr=lr, n_steps=50)[1]
    ax_right.plot(path[:, 0], path[:, 1], ".-", color=colours_19[lr], lw=1, ms=4, label=f"lr = {lr}")
ax_right.set(xlim=(-2.5, 2.5), ylim=(-1.6, 1.6), title="the first four descents")
ax_right.set_aspect("equal")
ax_right.legend(fontsize=7, loc="lower left")
plt.show()'''

S_20 = r'''def grad_s_20(v):
    """Gradient of s(x, y) = x² + y⁴/4 - y²/2: a saddle at (0, 0), minima at (0, 1) and (0, -1)."""
    return np.array([2 * v[0], v[1] ** 3 - v[1]])


def descend_20(start, lr=0.1, n_steps=2000):
    """Plain gradient descent on s (no mylearn here): the array of the n_steps + 1 points visited."""
    points = [np.array(start, dtype=float)]
    for _ in range(n_steps):
        points.append(points[-1] - lr * grad_s_20(points[-1]))
    return np.array(points)'''

EXPERIMENT_20 = r'''fig, (ax_y, ax_s) = plt.subplots(1, 2, figsize=(12.5, 3.8))
for colour, start in enumerate([(0.0, 0.0), (0.5, 1e-6), (0.5, -1e-6), (0.5, 1e-12)]):
    path = descend_20(start)
    beyond = np.flatnonzero(np.abs(path[:, 1]) > 0.5)
    first = int(beyond[0]) if beyond.size else None
    print(f"start {start}: after 2,000 steps ({path[-1, 0]:.3f}, {path[-1, 1]:.3f}); first step with |y| > 0.5: {first}")
    height = path[:, 0] ** 2 + path[:, 1] ** 4 / 4 - path[:, 1] ** 2 / 2
    ax_s.plot(height[:400], color=f"C{colour}", ls="--" if start[1] < 0 else "-", label=f"start {start}")
    if start[1] != 0:                                    # |y| = 0 has no logarithm
        ax_y.semilogy(np.abs(path[:400, 1]), color=f"C{colour}", ls="--" if start[1] < 0 else "-",
                      label=f"start {start}")
ax_y.set(xlabel="step", ylabel="|y| (log scale)")
ax_s.set(xlabel="step", ylabel="height s(x, y)")
ax_y.legend(fontsize=8)
ax_s.legend(fontsize=8)
plt.show()
print(f"your predictions: a) {prediction_5_20a}   b) {prediction_5_20b}   c) {prediction_5_20c}")'''

TORCH_21 = r'''def torch_gradient(f, x):
    """Gradient of f at x by automatic differentiation, in float64 (to compare with a numerical gradient)."""
    point = torch.tensor(x, dtype=torch.float64, requires_grad=True)   # PyTorch records what is done with point
    f(point).backward()                                                # chain rule, from f(point) back to point
    return point.grad.numpy()                                          # df/dpoint, as a NumPy array


POINT_21 = [-0.8, 0.9]'''

COST_21 = r'''weights_21 = np.linspace(-1.0, 1.0, 2000)          # 2,000 variables


def valley_21(w):
    """Rosenbrock in 2,000 dimensions; the same code works on NumPy arrays and on PyTorch tensors."""
    return (100 * (w[1:] - w[:-1] ** 2) ** 2 + (1 - w[:-1]) ** 2).sum()


with wb.attempt("5.21"):
    mylearn.calculus.numerical_gradient(valley_21, weights_21[:3])      # first, make sure your function is written
    with wb.timer("numerical gradient, 2,000 variables"):
        numeric_big_21 = mylearn.calculus.numerical_gradient(valley_21, weights_21)
    with wb.timer("torch.autograd, 2,000 variables"):
        auto_big_21 = torch_gradient(valley_21, weights_21)
    print(f"largest gap between the two gradients: {np.max(np.abs(numeric_big_21 - auto_big_21)):.1e}")'''

PART_C = Part("C", "Descendre : le learning rate, les points selles, autograd",
              "Fiche §5.4 et « au-delà du livre » (3). Tu écris la descente de gradient, tu mesures l'effet du learning "
              "rate sur un bol, tu prévois ce que fait une descente partie d'un point selle, puis tu compares ton "
              "gradient numérique au gradient de PyTorch.",
              exercises=[
    Ex("5.18", "🔨", 2, 30, "gradient_descent, et sa version qui monte",
       "écrire la descente (et la montée) de gradient, avec le chemin parcouru et un critère d'arrêt.",
       "Ex 5.15 · ✏️ 5.3 · fiche §5.4 (descendre une surface pas à pas)", thread="synthétique",
       tracks="R, M, C", mylearn="calculus.py",
       body=MYLEARN_SHORT + r"""

Écris `gradient_descent(grad, x0, lr=0.01, n_steps=100, tol=None, maximize=False)` (lis sa docstring) :
- lève une `ValueError` si `lr <= 0`, si `n_steps < 0`, ou si `tol` est donné et négatif ;
- pars d'une copie en flottants de `x0`, et range chaque point visité dans une liste (des **copies** : `x.copy()`) ;
- à chaque pas : évalue le gradient au point courant ; si `tol` est donné et que la norme euclidienne de ce gradient (`np.linalg.norm`, sur toutes ses composantes) est **strictement** inférieure à `tol`, arrête-toi sans bouger ; sinon, fais le pas $\mathbf{x} \leftarrow \mathbf{x} - \eta\,\nabla f(\mathbf{x})$, ou $\mathbf{x} + \eta\,\nabla f(\mathbf{x})$ si `maximize` ;
- renvoie le dernier point et le tableau du chemin (`np.array` de la liste), de forme `(nombre de pas faits + 1, *x0.shape)`.

Vérifications sur deux fonctions dont les gradients sont fournis : $q(\mathbf{v}) = (v_0 + 1)^2 + 4\,(v_1 - 0{,}5)^2$, de minimum $(-1 ;\ 0{,}5)$, et $m(\mathbf{v}) = -(v_0 - 1)^2 - 2\,(v_1 + 2)^2$, de maximum $(1 ;\ -2)$. La cellule de vérification appelle ta fonction :
a) le point atteint sur $q$ depuis $(1, 1)$, avec `lr=0.1` et 15 pas ;
b) le nombre de pas faits sur $q$ depuis $(1, 1)$, avec `lr=0.1`, `n_steps=1000` et `tol=1e-3` ;
c) le point atteint en **montant** $m$ depuis $(0, 0)$, avec `lr=0.2` et 3 pas ;
puis deux contrôles, les tests de `gradient_descent`, et la trajectoire de a) sur la carte de $q$.

Dans tes notes : en a), pourquoi $v_1$ est-il déjà au minimum alors que $v_0$ en est encore loin ? Regarde le facteur qui multiplie, à chaque pas, l'écart de chaque coordonnée au minimum (✏️ 5.3).""",
       given=r'''def grad_q_18(v):
    """Gradient of q(v) = (v0 + 1)² + 4 (v1 - 0.5)²."""
    return np.array([2 * (v[0] + 1), 8 * (v[1] - 0.5)])


def grad_m_18(v):
    """Gradient of m(v) = -(v0 - 1)² - 2 (v1 + 2)²."""
    return np.array([-2 * (v[0] - 1), -4 * (v[1] + 2)])''',
       check=RELOAD + r'''with wb.attempt("5.18"):
    end_18, path_18 = mylearn.calculus.gradient_descent(grad_q_18, [1.0, 1.0], lr=0.1, n_steps=15)
    wb.check("5.18a", end_18, computed=True)
    _, stopped_18 = mylearn.calculus.gradient_descent(grad_q_18, [1.0, 1.0], lr=0.1, n_steps=1000, tol=1e-3)
    wb.check("5.18b", len(stopped_18) - 1, computed=True)
    wb.check("5.18c", mylearn.calculus.gradient_descent(grad_m_18, [0.0, 0.0], lr=0.2, n_steps=3, maximize=True)[0],
             computed=True)
    start_18 = np.array([1.0, 1.0])
    mylearn.calculus.gradient_descent(grad_q_18, start_18, lr=0.1, n_steps=5)
    verdict("5.18", np.array_equal(start_18, [1.0, 1.0]), "le point de départ de l'appelant n'a pas bougé.",
            "ta fonction a modifié x0 : pars d'une copie, np.array(x0, dtype=float).")
    verdict("5.18", error_name(mylearn.calculus.gradient_descent, grad_q_18, [1.0, 1.0], lr=0.0) == "ValueError",
            "un learning rate nul est refusé (ValueError).",
            "gradient_descent(grad_q_18, [1.0, 1.0], lr=0.0) doit lever une ValueError.")
    run_calculus_tests("test_gradient_descent_")
''' + indent(PLOT_18),
       solution=r'''end_18, path_18 = mylearn.calculus.gradient_descent(grad_q_18, [1.0, 1.0], lr=0.1, n_steps=15)
_, stopped_18 = mylearn.calculus.gradient_descent(grad_q_18, [1.0, 1.0], lr=0.1, n_steps=1000, tol=1e-3)
n_done_18 = len(stopped_18) - 1
top_18 = mylearn.calculus.gradient_descent(grad_m_18, [0.0, 0.0], lr=0.2, n_steps=3, maximize=True)[0]
print(end_18, n_done_18, top_18)
print("factors per step:", round(1 - 0.1 * 2, 2), "for v0 and", round(1 - 0.1 * 8, 2), "for v1")
run_calculus_tests("test_gradient_descent_", impl="ref")
''' + PLOT_18,
       record=r'''wrong_way_18 = mylearn.calculus.gradient_descent(grad_q_18, [1.0, 1.0], lr=0.1, n_steps=15, maximize=True)[0]
wb.record("5.18a", end_18, decimals=4, mistakes={"tu es monté : pour descendre, on RETIRE lr × gradient": wrong_way_18,
                                                 "c'est le point après 14 pas : n_steps pas donnent n_steps + 1 points dans le chemin": mylearn.calculus.gradient_descent(grad_q_18, [1.0, 1.0], lr=0.1, n_steps=14)[0]})
wb.record("5.18b", n_done_18, mistakes={"ta descente fait un pas de trop : teste tol AVANT de bouger, au point courant": n_done_18 + 1,
                                        "tol n'est pas pris en compte : avant chaque pas, arrête-toi si la norme du gradient est < tol": 1000})
wb.record("5.18c", top_18, decimals=4, mistakes={"tu es descendu : avec maximize=True, on AJOUTE lr × gradient": mylearn.calculus.gradient_descent(grad_m_18, [0.0, 0.0], lr=0.2, n_steps=3)[0]})''',
       note="En a), l'écart de $v_0$ au minimum est multiplié par $1 - 2\\eta = 0{,}8$ à chaque pas, celui de $v_1$ par "
            "$1 - 8\\eta = 0{,}2$ : après 15 pas, $0{,}8^{15} \\approx 0{,}035$, mais $0{,}2^{15} \\approx 3 \\times "
            "10^{-11}$. La direction la plus courbée converge le plus vite (fiche, « un learning rate pour plusieurs "
            "courbures »). En b), 38 pas suffisent pour que la norme du gradient passe sous $10^{-3}$. La montée de c) "
            "est la descente de $-m$ ; le paramètre `maximize` porte le même nom dans `torch.optim.SGD`, l'oracle des "
            "tests."),

    Ex("5.19", "🔬", 2, 25, "Learning rate sur un bol : trop petit, juste, trop grand",
       "mesurer l'effet du learning rate sur une surface dont les deux directions n'ont pas la même courbure.",
       "Ex 5.18 · Ex 1.17 · ✏️ 5.3 · fiche §5.4 (un learning rate pour plusieurs courbures)", thread="synthétique",
       tracks="C",
       body=r"""Le bol $b(\mathbf{v}) = v_0^2 + 10\,v_1^2$ est dix fois plus courbé selon $v_1$ que selon $v_0$. Avec ta fonction `gradient_descent`, descends-le depuis $(2, 1)$, 50 pas, avec chacun des learning rates de `LRS_19` : 0,01 ; 0,05 ; 0,09 ; 0,1 et 0,11.

Écris `distances_19(lr)` : la distance au minimum $(0, 0)$ de chacun des 51 points du chemin, départ compris (un tableau de 51 nombres ; `np.linalg.norm(..., axis=1)`). La vérification contrôle tes distances avec la formule exacte (à chaque pas, $v_0$ est multiplié par $1 - 2\eta$ et $v_1$ par $1 - 20\eta$, ✏️ 5.3), puis trace les distances pas à pas, en échelle logarithmique, et les cinq trajectoires sur la carte du bol.

Réponds dans la cellule 📝 :
- quels learning rates convergent ? Lequel est le plus rapide, et pourquoi n'est-ce pas le plus grand ?
- que fait la descente avec 0,1 ? avec 0,11 ? Quelle direction fixe le plus grand learning rate possible, et laquelle fixe la vitesse ?
- avec 0,01, combien de pas faudrait-il, à peu près, pour arriver à moins de 0,01 du minimum ?""",
       given=r'''def grad_bowl_19(v):
    """Gradient of b(v) = v0² + 10 v1²."""
    return np.array([2 * v[0], 20 * v[1]])


START_19 = [2.0, 1.0]
LRS_19 = [0.01, 0.05, 0.09, 0.1, 0.11]''',
       todo=r'''def distances_19(lr):
    """Distances to (0, 0) of the 51 points of 50 steps of YOUR gradient_descent on the bowl, from START_19."""
    raise NotImplementedError("distances_19() is not written yet")''',
       check=RELOAD + r'''with wb.attempt("5.19"):
    curves_19 = {lr: np.asarray(distances_19(lr), dtype=float) for lr in LRS_19}
    exact_19 = {lr: np.hypot(2 * (1 - 2 * lr) ** np.arange(51), (1 - 20 * lr) ** np.arange(51)) for lr in LRS_19}
    right_19 = all(curve.shape == (51,) and np.allclose(curve, exact_19[lr], rtol=1e-9)
                   for lr, curve in curves_19.items())
    verdict("5.19", right_19,
            "tes distances suivent la formule exacte : lis les graphiques, puis réponds dans la cellule 📝.",
            "distances_19(lr) doit renvoyer les 51 distances au minimum (départ compris) de 50 pas de gradient_descent "
            "depuis START_19 ; les graphiques s'afficheront ensuite.")
    if right_19:
''' + indent(DRAW_19, 8),
       solution=r'''def distances_19(lr):
    """Distances to (0, 0) of the 51 points of 50 steps of YOUR gradient_descent on the bowl, from START_19."""
    _, path = mylearn.calculus.gradient_descent(grad_bowl_19, START_19, lr=lr, n_steps=50)
    return np.linalg.norm(path, axis=1)


curves_19 = {lr: distances_19(lr) for lr in LRS_19}
for lr, curve in curves_19.items():
    print(f"lr = {lr:<5}: factors {1 - 2 * lr:+.2f} (v0) and {1 - 20 * lr:+.2f} (v1); distance after 50 steps {curve[-1]:.3g}")
print("steps to get within 0.01 with lr = 0.01:", np.log(0.01 / 2) / np.log(0.98))
''' + DRAW_19,
       after=[("todo_md", "📝 **Mes réponses** (learning rates qui convergent, le plus rapide ; 0,1 et 0,11 ; la "
                          "direction qui fixe le learning rate maximal, celle qui fixe la vitesse ; le nombre de pas "
                          "avec 0,01) : …")],
       note="Avec 0,01, la descente est sûre mais lente : $v_0$ n'est multiplié que par 0,98 par pas (il reste 0,73 "
            "après 50 pas, et il faudrait environ 260 pas pour arriver à 0,01). Avec 0,05, $v_1$ tombe à 0 en un seul "
            "pas (facteur $1 - 20\\eta = 0$), mais $v_0$ ne gagne qu'un facteur 0,9. Avec 0,09, les deux facteurs "
            "valent 0,82 et $-0{,}8$ : $v_1$ oscille en s'amortissant, et c'est le plus rapide. Le meilleur choix "
            "égalise les deux facteurs en valeur absolue, $1 - 2\\eta = -(1 - 20\\eta)$, soit $\\eta = 1/11 "
            "\\approx 0{,}091$. Avec 0,1, le facteur de $v_1$ vaut exactement $-1$ : $v_1$ saute de 1 à $-1$ "
            "indéfiniment, et la distance reste bloquée vers 1. Avec 0,11, il vaut $-1{,}2$ : la descente diverge. "
            "La direction la plus courbée ($v_1$) fixe le plus grand learning rate possible, la plus plate ($v_0$) "
            "fixe la vitesse. C'est tout le problème de la vallée de Rosenbrock (🏆 5.25)."),

    Ex("5.20", "🔮", 2, 15, "Démarrer pile sur un point selle",
       "prévoir ce que fait une descente de gradient partie d'un point selle, ou d'un point très proche.",
       "Ex 5.18 · ✏️ 5.5 · fiche §5.4 (les points où le gradient s'annule, 🕰️ points selles)", thread="synthétique",
       tracks="C", hypothesis=True,
       body=r"""La surface $s(x, y) = x^2 + \frac{y^4}{4} - \frac{y^2}{2}$ a un point selle en $(0, 0)$ et deux minima, en $(0, 1)$ et $(0, -1)$ ; son gradient vaut $(2x,\ y^3 - y)$. On la descend avec `lr=0.1`, pendant 2 000 pas. **Sans rien exécuter**, prévois :
a) `prediction_5_20a` : partie **exactement** de $(0, 0)$, la descente y reste-t-elle ? (`True` ou `False`) ;
b) `prediction_5_20b` : partie de $(0{,}5 ;\ 10^{-6})$, un millionième au-dessus de l'axe des $x$, où finit-elle ? `1` pour $(0, 1)$, `-1` pour $(0, -1)$, `0` si elle reste près de $(0, 0)$ ;
c) `prediction_5_20c` : depuis ce même départ, combien de pas faut-il pour que $|y|$ dépasse 0,5 ? Choisis l'ordre de grandeur : `10`, `100`, `1000` ou `10000`.

Écris ton hypothèse (cellule 📝), puis tes trois prédictions ; la vérification ne regarde que tes prédictions. Ensuite seulement, exécute l'**Expérience** : quatre départs, dont $(0{,}5 ;\ -10^{-6})$ et $(0{,}5 ;\ 10^{-12})$.""",
       given=S_20,
       todo=r'''prediction_5_20a = ...   # True or False
prediction_5_20b = ...   # 1, -1 or 0
prediction_5_20c = ...   # 10, 100, 1000 or 10000''',
       check=r'''wb.check("5.20a", prediction_5_20a)
wb.check("5.20b", prediction_5_20b)
try:                                                        # c) is one of four proposed orders of magnitude
    off_scale_20 = filled(prediction_5_20c) and float(str(prediction_5_20c).replace(",", ".")) not in (10, 100, 1000, 10000)
except ValueError:
    off_scale_20 = False
if off_scale_20:
    print("❌ Ex 5.20c : réponds par l'un des quatre ordres de grandeur proposés : 10, 100, 1000 ou 10000.")
else:
    wb.check("5.20c", prediction_5_20c)''',
       solution=r'''prediction_5_20a, prediction_5_20b, prediction_5_20c = True, 1, 100''',
       record=r'''wb.record("5.20a", prediction_5_20a, mistakes={"en (0, 0), le gradient est exactement nul : chaque pas retranche 0, et rien ne bouge": False})
wb.record("5.20b", prediction_5_20b, mistakes={"le départ est au-dessus de l'axe (y > 0) : rien ne fait changer y de signe": -1,
                                                "le point selle repousse dans la direction y : un écart minuscule grandit à chaque pas": 0})
wb.record("5.20c", prediction_5_20c, mistakes={"trop tôt : près du point selle, y n'est multiplié que par 1,1 environ à chaque pas, et il doit passer de 0,000001 à 0,5": 10,
                                                "trop tard : y est multiplié par 1,1 environ à chaque pas ; combien de fois faut-il multiplier par 1,1 pour gagner un facteur 500 000 ?": 1000,
                                                "beaucoup trop tard : y grandit de 10 % environ à chaque pas, c'est une croissance exponentielle": 10000})''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", guarded(EXPERIMENT_20, ["prediction_5_20a", "prediction_5_20b", "prediction_5_20c"],
                               "⏳ Ex 5.20 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Près du point selle, $|y|$ est petit et $y^3$ est négligeable devant $y$ : un pas fait alors "
                     "$y \\leftarrow y - \\eta\\,(y^3 - y) \\approx (1 + \\eta)\\,y = 1{,}1\\,y$.\n\n"
                     "d) `extra_steps_5_20d` : avec ce facteur, combien de pas de plus faut-il pour que $|y|$ dépasse "
                     "0,5 quand le départ est $10^{-12}$ au lieu de $10^{-6}$, un million de fois plus près de l'axe ? "
                     "(un entier, arrondi au plus proche ; compare-le ensuite à l'expérience)"),
              ("todo", "extra_steps_5_20d = ...   # d) an integer"),
              ("check", 'wb.check("5.20d", extra_steps_5_20d)'),
              ("solution", "extra_steps_5_20d = round(np.log(1e6) / np.log(1.1))     # 1.1 ** n = 10**6\n"
                           "print(extra_steps_5_20d, np.log(1e6) / np.log(1.1))"),
              ("record", r'''wb.record("5.20d", extra_steps_5_20d, mistakes={"tu as remplacé ln(1,1) par 0,1 : sa valeur est 0,0953, garde-la": round(np.log(1e6) / 0.1),
                                                 "c'est le nombre TOTAL de pas depuis 10⁻¹² : on demande les pas EN PLUS": 284,
                                                 "c'est le nombre de pas depuis 10⁻⁶ : on demande les pas EN PLUS quand on part de 10⁻¹²": 139,
                                                 "ln(10⁶) / ln(1,1) vaut environ 144,95 : arrondis au plus proche, pas vers le bas": 144})''')],
       note="En $(0, 0)$, le gradient est exactement nul : la descente ne bouge pas, alors que ce n'est pas un minimum. "
            "Un millionième suffit pour en sortir, mais lentement : $y$ grandit de 10 % par pas, il faut 139 pas pour "
            "atteindre 0,5, puis la descente file vers le minimum $(0, 1)$, du côté où elle était. Pendant ces pas, la "
            "hauteur $s$ reste presque constante : un **plateau**, qui ressemble à une convergence. Partir un million "
            "de fois plus près de l'axe coûte 145 pas de plus ($\\ln 10^6 / \\ln 1{,}1$) : le temps passé près d'un "
            "point selle croît comme le logarithme de la distance initiale, ce qui reste lent en grande dimension "
            "(Du et coll., 2017, 🕰️ de la fiche). Le bruit de la descente stochastique (ch. 19) suffit à quitter la "
            "position exacte du point selle."),

    Ex("5.21", "📦", 2, 20, "Le même gradient avec torch.autograd",
       "calculer un gradient par différentiation automatique avec PyTorch, le comparer au gradient numérique, et voir "
       "ce qu'il donne en un coin.",
       "Ex 5.15 · fiche, au-delà du livre (3) (🧮 PyTorch en cinq lignes) · quiz Q9", thread="Rosenbrock",
       tracks="R, C",
       body=r"""La fonction `torch_gradient` ci-dessous calcule un gradient par **différentiation automatique** : PyTorch enregistre les opérations faites avec le tenseur `point`, puis `backward()` applique la règle de la chaîne en remontant ces opérations (fiche, au-delà du livre (3)). Il faut lui donner une fonction écrite avec des opérations que PyTorch sait dériver : `+`, `-`, `*`, `/`, `**`, `torch.exp`… mais pas les fonctions de NumPy.

Écris `rosenbrock_torch(p)` : la fonction de Rosenbrock ($a = 1$, $b = 100$) d'un tenseur `p` de deux composantes, `p[0]` et `p[1]`, avec ces opérations. La cellule de vérification l'appelle :
a) le gradient par autograd au point $(-0{,}8 ;\ 0{,}9)$ ; elle le compare ensuite à ton `numerical_gradient`.

La fonction $\max(0, x)$ (la ReLU des réseaux de neurones, ch. 17) a un point anguleux en 0 (quiz Q9) : sa pente vaut 0 à gauche et 1 à droite. PyTorch permet de l'écrire de plusieurs façons. Avec `torch_gradient` et ta fonction `numerical_derivative`, calcule quatre « pentes en 0 » :
b) `corner_21` : la liste [pente de `torch.relu(x)` en 0, pente de `torch.clamp(x, min=0)` en 0, pente de `torch.maximum(x, torch.zeros_like(x))` en 0, différence centrée de `max(0, x)` en 0].

La dernière cellule compare enfin les durées des deux méthodes sur une fonction de 2 000 variables. Dans tes notes : combien d'évaluations de la fonction coûte un gradient numérique pour $n$ variables ? Que penser de la « pente en 0 » de la ReLU ?""",
       given=TORCH_21,
       todo=r'''def rosenbrock_torch(p):
    """Rosenbrock function (a = 1, b = 100) of a tensor p = [x, y], with operations that PyTorch can differentiate."""
    raise NotImplementedError("rosenbrock_torch() is not written yet")


corner_21 = ...   # b) [torch.relu, torch.clamp(x, min=0), torch.maximum(x, torch.zeros_like(x)), central difference of max(0, x)]''',
       check=RELOAD + r'''with wb.attempt("5.21"):
    auto_21 = torch_gradient(rosenbrock_torch, POINT_21)
    wb.check("5.21a", auto_21, computed=True)
    numeric_21 = np.asarray(mylearn.calculus.numerical_gradient(rosenbrock, POINT_21), dtype=float)
    print(f"autograd {auto_21} · your numerical gradient {numeric_21} · "
          f"largest gap {np.max(np.abs(auto_21 - numeric_21)):.1e}")
wb.check("5.21b", corner_21)''',
       solution=r'''def rosenbrock_torch(p):
    """Rosenbrock function (a = 1, b = 100) of a tensor p = [x, y], with operations that PyTorch can differentiate."""
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


auto_21 = torch_gradient(rosenbrock_torch, POINT_21)
numeric_21 = mylearn.calculus.numerical_gradient(rosenbrock, POINT_21)
print(auto_21, numeric_21, f"largest gap {np.max(np.abs(auto_21 - numeric_21)):.1e}")
corner_21 = [float(torch_gradient(lambda p: torch.relu(p[0]), [0.0])[0]),
             float(torch_gradient(lambda p: torch.clamp(p[0], min=0), [0.0])[0]),
             float(torch_gradient(lambda p: torch.maximum(p[0], torch.zeros_like(p[0])), [0.0])[0]),
             mylearn.calculus.numerical_derivative(lambda x: max(0.0, x), 0.0)]
print(corner_21)''',
       record=r'''wb.record("5.21a", auto_21, decimals=4, mistakes={"le terme de la vallée est multiplié par b = 100 : (1 − x)² + 100 (y − x²)²": torch_gradient(lambda p: (1 - p[0]) ** 2 + (p[1] - p[0] ** 2) ** 2, POINT_21)})
wb.record("5.21b", corner_21, decimals=1, mistakes={"les trois écritures PyTorch ne donnent pas la même pente en 0 : calcule-les vraiment, l'une après l'autre": [0.0, 0.0, 0.0, 0.5],
                                                    "torch.clamp et torch.maximum ne donnent pas la même pente que torch.relu en 0 : calcule-les vraiment": [1.0, 1.0, 1.0, 0.5],
                                                    "c'est la différence AVANT : la différence centrée de max(0, x) en 0 vaut (h − 0) / (2h)": [0.0, 1.0, 0.5, 1.0],
                                                    "c'est la différence ARRIÈRE : la différence centrée de max(0, x) en 0 vaut (h − 0) / (2h)": [0.0, 1.0, 0.5, 0.0]})''',
       after=[("md", "**Le prix d'un gradient** : la même comparaison sur la fonction de Rosenbrock à 2 000 variables "
                     "(`valley_21`), dont le code marche aussi bien sur un tableau NumPy que sur un tenseur. Exécute la "
                     "cellule et compare les deux durées."),
              ("code", COST_21)],
       note="Au point $(-0{,}8 ;\\ 0{,}9)$, autograd donne $(79{,}6 ;\\ 52)$, et la différence centrée le retrouve à "
            "environ $10^{-8}$ près (son erreur de troncature et d'arrondi). Autograd n'a pas de pas $h$ : son gradient "
            "est exact aux arrondis près. En 0, la ReLU n'a pas de dérivée : chaque opération de PyTorch a sa "
            "convention, alors que les trois écritures calculent la même fonction : 0 pour `torch.relu`, 1 pour "
            "`torch.clamp(x, min=0)`, 0,5 pour `torch.maximum`, qui partage la pente entre ses deux branches ; la "
            "différence centrée donne la moyenne des deux pentes, 0,5. En théorie, une entrée réelle ne tombe presque "
            "jamais pile sur 0 et le choix ne compte pas ; en `float32`, il modifie pourtant souvent les gradients "
            "calculés (quiz Q9 et fiche, 🕰️ ReLU). Enfin, le gradient numérique de 2 000 variables demande 4 000 évaluations de "
            "la fonction, contre une évaluation et un passage en arrière pour autograd : sur un réseau d'un million de "
            "paramètres, l'écart devient énorme (🧮 5.9)."),
])

# ---------------------------------------------------------------------------
# Part D: a figure, parametrized tests, critical points, the challenge (5.22 to 5.25)
# ---------------------------------------------------------------------------
DRAW_22 = r'''fig = plt.figure(figsize=(12, 4.8))
try:
    draw_surface_22(fig.add_subplot(1, 2, 1, projection="3d"), path_22)
    draw_map_22(fig.add_subplot(1, 2, 2), path_22)
except BaseException:
    plt.close(fig)                    # no empty figure while the drawing functions are not written
    raise
plt.tight_layout()
plt.show()'''

VERSIONS_23 = r'''def gradient_ok_23(f, x, h=1e-5):
    """A correct numerical gradient, to test your tests."""
    point = np.array(x, dtype=float)
    flat, grad = point.reshape(-1), np.zeros_like(point)
    flat_grad = grad.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + h
        f_plus = f(point)
        flat[i] = old - h
        f_minus = f(point)
        flat[i] = old
        flat_grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


def gradient_bug_1_23(f, x, h=1e-5):
    """Bug 1: the forward difference instead of the central one."""
    point = np.array(x, dtype=float)
    flat, grad = point.reshape(-1), np.zeros_like(point)
    flat_grad = grad.reshape(-1)
    f_here = f(point)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + h
        flat_grad[i] = (f(point) - f_here) / h
        flat[i] = old
    return grad


def gradient_bug_2_23(f, x, h=1e-5):
    """Bug 2: copies x but keeps its type, so an integer point stays an integer array."""
    point = np.array(x)
    flat, grad = point.reshape(-1), np.zeros(point.shape)
    flat_grad = grad.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + h
        f_plus = f(point)
        flat[i] = old - h
        f_minus = f(point)
        flat[i] = old
        flat_grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


def gradient_bug_3_23(f, x, h=1e-5):
    """Bug 3: only right for a vector; for a matrix, point[i] is a whole row."""
    point = np.array(x, dtype=float)
    grad = np.zeros(len(point))
    for i in range(len(point)):
        old = point[i].copy()
        point[i] = old + h
        f_plus = f(point)
        point[i] = old - h
        f_minus = f(point)
        point[i] = old
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad'''

CHECK_23 = r'''if not filled(my_tests_23):
    print("⏳ Ex 5.23 : tests pas encore écrits.")
else:
    marked_23 = [test for test in my_tests_23
                 if any(mark.name == "parametrize" for mark in getattr(test, "pytestmark", []))]
    verdict("5.23", len(my_tests_23) >= 3 and len(marked_23) >= 1,
            f"{len(my_tests_23)} tests, dont {len(marked_23)} paramétré(s) par @pytest.mark.parametrize.",
            "il faut au moins trois tests, dont au moins un décoré par @pytest.mark.parametrize.")
    good_23 = wb.run_pytest(my_tests_23, subject=gradient_ok_23, name="numerical_gradient")
    verdict("5.23", good_23.ok, f"tes tests passent sur une version juste ({good_23.passed} cas).",
            "tes tests doivent tous passer sur une version juste (voir le détail au-dessus).")
    if good_23.ok:
        for bug in [gradient_bug_1_23, gradient_bug_2_23, gradient_bug_3_23]:
            result = wb.run_pytest(my_tests_23, subject=bug, name="numerical_gradient", quiet=True)
            verdict("5.23", result.failed + result.errors > 0, f"{bug.__name__} est attrapé ({result.failed} cas en échec).",
                    f"tes tests passent sur {bug.__name__}, qui est faux : ajoute un test qui le fait échouer.")'''

CHALLENGE_25 = r'''START_25 = (-1.5, 2.0)                   # a tuple: nothing can change it
TARGET_25 = np.array([1.0, 1.0])


def grade_25(phases, start=(-1.5, 2.0), target=(1.0, 1.0)):
    """Play the phases with YOUR gradient_descent from `start`: (total steps, final distance to `target`, then, during
    10,000 more steps with the last learning rate, the worst distance and the distance at the end)."""
    point, target = np.array(start, dtype=float), np.array(target, dtype=float)
    for lr, n_steps in phases:
        point, _ = mylearn.calculus.gradient_descent(rosenbrock_gradient, np.array(point, dtype=float), lr=lr,
                                                    n_steps=int(n_steps))
    final = float(np.linalg.norm(np.asarray(point, dtype=float) - target))
    _, stay = mylearn.calculus.gradient_descent(rosenbrock_gradient, np.array(point, dtype=float), lr=phases[-1][0],
                                                n_steps=10_000)
    distances = np.linalg.norm(np.asarray(stay, dtype=float) - target, axis=1)
    return sum(int(n_steps) for _, n_steps in phases), final, float(distances.max()), float(distances[-1])'''

GRADE_25 = r'''    phases_25 = [tuple(phase) for phase in schedule_25()]
    if not (1 <= len(phases_25) <= 3 and all(len(phase) == 2 and np.ndim(phase[0]) == 0 and np.ndim(phase[1]) == 0
                                                and phase[0] > 0 and float(phase[1]).is_integer() and phase[1] >= 100
                                                for phase in phases_25)):
        verdict("5.25", False, "", "schedule_25() doit renvoyer une liste de 1 à 3 phases (lr, n_steps) : lr un nombre "
                "> 0, n_steps un entier au moins égal à 100.")
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            total_25, final_25, worst_25, end_25 = grade_25(phases_25)
        print(f"{len(phases_25)} phase(s), {total_25} steps: final distance {final_25:.2e}; during the next 10,000 "
              f"steps: worst distance {worst_25:.2e}, distance at the end {end_25:.2e}")
        verdict("5.25", total_25 <= 10_000 and final_25 < 1e-3 and worst_25 < 1e-3 and end_25 < 1e-4,
                f"défi réussi : (1, 1) atteint à {final_25:.1e} près en {total_25} pas, et tenu.",
                "il faut au plus 10 000 pas et une distance finale inférieure à 1e-3 ; puis, pendant les 10 000 pas "
                "suivants, ne jamais s'éloigner à 1e-3 ou plus, et finir à moins de 1e-4.")'''

PART_D = Part("D", "Dessiner, tester, classer, et un défi",
              "Fiche §5.4 et les pièges classiques. Tu dessines la goutte d'eau du livre sur la vallée de Rosenbrock, "
              "tu protèges un gradient numérique par des tests de propriétés, tu classes des points critiques, puis tu "
              "règles une descente pour atteindre le fond de la vallée.",
              exercises=[
    Ex("5.22", "🎨", 2, 30, "L'eau qui descend la surface (figure 5.18)",
       "dessiner une descente de gradient sur une surface en trois dimensions et sur sa carte, avec les directions de "
       "plus grande descente.",
       "Ex 5.18 · fiche §5.4 (descendre une surface pas à pas, 🧮 lignes de niveau et flèches) · livre figure 5.18",
       thread="Rosenbrock", tracks="C",
       body=r"""Le livre montre la descente comme une goutte d'eau qui suit, pas après pas, la direction de plus grande descente (figures 5.16 et 5.18). Refais cette image sur la vallée de Rosenbrock : 3 000 pas de descente depuis $(-1{,}2 ;\ 1)$, le départ classique, avec `lr=0.001` et le gradient exact `rosenbrock_gradient`.

1. Écris `draw_surface_22(ax, path)` : sur l'axe 3D `ax`, la surface sur $[-2, 2] \times [-1, 3]$ (`np.meshgrid`, puis `ax.plot_surface(X, Y, Z, cmap=..., alpha=0.6)`), et par-dessus le chemin, à la hauteur de la surface (`ax.plot(xs, ys, zs)`), départ et arrivée marqués. En hauteur, prends $\log_{10}(1 + f)$ : les parois montent jusqu'à 2 500 et écraseraient la vallée (le logarithme garde les mêmes directions de descente).
2. Écris `draw_map_22(ax, path)` : sur un axe ordinaire, la carte de la même surface (`wb.plot.plot_contour(wb.synth.rosenbrock, ..., ax=ax)` trace les lignes de niveau et le chemin), et, tous les 300 pas, une flèche dans la direction de plus grande descente $-\nabla f$ (`ax.quiver` ; divise chaque flèche par sa norme pour qu'elles aient toutes la même longueur).

La vérification calcule le chemin avec ta fonction `gradient_descent`, contrôle sa forme, puis trace les deux panneaux avec tes fonctions. Compare avec la figure du corrigé, puis réponds dans tes notes : où en est la goutte après 3 000 pas ? Pourquoi descend-elle si vite au début, et si lentement ensuite ?""",
       given=r'''START_22 = [-1.2, 1.0]
LR_22, STEPS_22 = 0.001, 3000''',
       todo=r'''def draw_surface_22(ax, path):
    """The surface log10(1 + f) on [-2, 2] x [-1, 3] (3-D axis ax), with the path drawn on it."""
    raise NotImplementedError("draw_surface_22() is not written yet")


def draw_map_22(ax, path):
    """The map of the same surface, the path, and a unit arrow of -gradient every 300 steps."""
    raise NotImplementedError("draw_map_22() is not written yet")''',
       check=RELOAD + r'''with wb.attempt("5.22"):
    end_22, path_22 = mylearn.calculus.gradient_descent(rosenbrock_gradient, START_22, lr=LR_22, n_steps=STEPS_22)
    verdict("5.22", np.shape(path_22) == (STEPS_22 + 1, 2),
            f"le chemin a {STEPS_22 + 1} points ; la goutte s'arrête en ({end_22[0]:.3f} ; {end_22[1]:.3f}).",
            "gradient_descent doit renvoyer ici un chemin de forme (3001, 2) : reprends 5.18.")
''' + indent(DRAW_22),
       solution=r'''def draw_surface_22(ax, path):
    """The surface log10(1 + f) on [-2, 2] x [-1, 3] (3-D axis ax), with the path drawn on it."""
    X, Y = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-1, 3, 60))
    ax.plot_surface(X, Y, np.log10(1 + wb.synth.rosenbrock(X, Y)), cmap="viridis", alpha=0.6, linewidth=0,
                    antialiased=False)
    path = np.asarray(path)
    heights = np.log10(1 + wb.synth.rosenbrock(path[:, 0], path[:, 1]))
    ax.plot(path[:, 0], path[:, 1], heights, color="red", lw=2)
    ax.scatter(path[0, 0], path[0, 1], heights[0], color="red", s=40, label="start")
    ax.scatter(path[-1, 0], path[-1, 1], heights[-1], color="black", marker="s", s=40, label="after 3,000 steps")
    ax.view_init(elev=40, azim=-110)
    ax.set(xlabel="x", ylabel="y", zlabel="log10(1 + f)")
    ax.legend(loc="upper left", fontsize=8)


def draw_map_22(ax, path):
    """The map of the same surface, the path, and a unit arrow of -gradient every 300 steps."""
    path = np.asarray(path)
    wb.plot.plot_contour(wb.synth.rosenbrock, xlim=(-2, 2), ylim=(-1, 3), path=path, ax=ax, minimum=(1, 1), levels=15)
    marks = path[::300]
    arrows = -np.array([rosenbrock_gradient(point) for point in marks])
    arrows = arrows / np.linalg.norm(arrows, axis=1, keepdims=True)
    ax.quiver(marks[:, 0], marks[:, 1], arrows[:, 0], arrows[:, 1], color="white", scale=15, width=0.006)
    ax.set_title("3,000 steps with lr = 0.001; an arrow of -gradient every 300 steps", fontsize=10)


end_22, path_22 = mylearn.calculus.gradient_descent(rosenbrock_gradient, START_22, lr=LR_22, n_steps=STEPS_22)
print("after 3,000 steps:", end_22, "f =", rosenbrock(end_22), "· after 1 step: f =", rosenbrock(path_22[1]))
''' + DRAW_22,
       note="Le premier pas fait tomber $f$ de 24,2 à 5,4 : la goutte part de la paroi, où la pente est très forte, et "
            "tombe presque d'un coup au fond de la vallée. Ensuite, elle suit le fond en courbe, où la pente est faible : "
            "après 3 000 pas, elle n'est qu'en $(0{,}84 ;\\ 0{,}71)$, avec $f \\approx 0{,}025$, encore loin de $(1, 1)$. "
            "Les flèches de $-\\nabla f$ pointent presque perpendiculairement au fond de la vallée, vers lui, et non "
            "vers le minimum : la goutte est sans cesse ramenée au fond, et n'avance que par la petite composante qui "
            "le longe (fiche, ⚠️ les limites de l'image de l'eau)."),

    Ex("5.23", "🛠️", 2, 20, "Tests de propriétés paramétrés avec pytest",
       "tester une fonction numérique par des propriétés vraies sur beaucoup d'entrées, avec "
       "`@pytest.mark.parametrize`.",
       "Ex 5.15 · 0A.62 (pytest, parametrize) · Ex 4.23", thread="—", tracks="C",
       body=r"""Pour une fonction numérique, on ne connaît pas toujours la valeur exacte attendue ; on connaît en revanche des **propriétés** qu'elle doit respecter sur beaucoup d'entrées. C'est l'idée des **tests de propriétés** : un même test, répété sur plusieurs entrées avec `@pytest.mark.parametrize` (0A.62). Pour un gradient numérique, par exemple :
- sur une forme quadratique $f(\mathbf{x}) = \mathbf{x}^\top A\,\mathbf{x} + \mathbf{b}^\top\mathbf{x}$, le gradient exact est $(A + A^\top)\,\mathbf{x} + \mathbf{b}$ (0B), et la différence centrée le retrouve aux arrondis près ; on peut le vérifier en dimension 1, 2, 5…, avec des $A$, $\mathbf{b}$ et $\mathbf{x}$ tirés au hasard avec une graine ;
- un point écrit en entiers donne le même gradient que le même point écrit en flottants ;
- le gradient a la forme du point, même quand le point est une matrice.

Écris au moins **trois** tests pytest qui appellent `numerical_gradient(f, x)`, dont **au moins un** paramétré par `@pytest.mark.parametrize`, et range-les dans la liste `my_tests_23`. Comme en 4.23, la vérification lance vraiment pytest : tes tests doivent **passer** sur une version juste (`gradient_ok_23`) et **échouer** sur chacune des trois versions buggées de la cellule ci-dessous (lis-les). Pour comparer des flottants, prends une tolérance serrée, `np.allclose(..., atol=1e-6)` par exemple : une tolérance trop large laisserait passer un bug. Tes tests peuvent utiliser `numerical_gradient`, `np`, `math` et `pytest` ; seules tes fonctions de test sont copiées dans le fichier que lance pytest : définis tout ce dont elles ont besoin (une fonction $f$, une matrice) **à l'intérieur** de chaque test.""",
       given=VERSIONS_23,
       todo=r'''# Write your test functions here; each one calls numerical_gradient(f, x).
# @pytest.mark.parametrize("n", [1, 2, 5])
# def test_something(n):
#     ...
#     assert ...

my_tests_23 = ...   # the list of your test functions''',
       check=CHECK_23,
       solution=r'''@pytest.mark.parametrize("n, seed", [(1, 0), (2, 1), (5, 2)])
def test_matches_the_gradient_of_a_quadratic_form(n, seed):
    rng = np.random.default_rng(seed)
    A, b, x = rng.normal(size=(n, n)), rng.normal(size=n), rng.normal(size=n)
    result = numerical_gradient(lambda v: v @ A @ v + b @ v, x)
    assert np.allclose(result, (A + A.T) @ x + b, atol=1e-6)


def test_an_integer_point_gives_the_same_gradient():
    def f(v):
        return v[0] ** 2 * v[1] + 3 * v[1]
    assert np.allclose(numerical_gradient(f, np.array([-1, 2])), numerical_gradient(f, np.array([-1.0, 2.0])),
                       atol=1e-6)


def test_the_gradient_of_a_matrix_has_its_shape():
    W = np.arange(6.0).reshape(2, 3)
    result = numerical_gradient(lambda M: np.sum(M ** 2), W)
    assert result.shape == (2, 3)
    assert np.allclose(result, 2 * W, atol=1e-6)


my_tests_23 = [test_matches_the_gradient_of_a_quadratic_form, test_an_integer_point_gives_the_same_gradient,
               test_the_gradient_of_a_matrix_has_its_shape]
''' + CHECK_23,
       note="Chaque propriété attrape un bug. La forme quadratique, avec une tolérance de $10^{-6}$, attrape la "
            "différence avant (bug 1) : son erreur, environ $h\\,A_{ii}$, vaut quelques $10^{-5}$, alors que la "
            "différence centrée est exacte sur un polynôme de degré 2, aux arrondis près (∂ 5.6). Le point entier "
            "attrape le bug 2 (les décalages de $h$ sont tronqués), à condition de choisir un point où la vraie dérivée "
            "n'est pas nulle : en $(0, 0)$, les décalages tronqués donnent un gradient nul qui peut être juste par "
            "hasard. La matrice attrape le bug 3. Un seul cas « calculé à la main » n'en aurait attrapé qu'un ; les "
            "propriétés vérifiées sur plusieurs entrées tirées au hasard (avec une graine, pour que l'échec soit "
            "reproductible) couvrent bien plus de situations. La bibliothèque *Hypothesis* pousse l'idée plus loin : "
            "elle choisit elle-même les entrées, et réduit un échec au plus petit exemple qui le provoque."),

    Ex("5.24", "🔨", 3, 35, "Minimum, maximum, selle ou plat : classify_critical_point",
       "classer un point critique par les signes des dérivées secondes dans plusieurs directions, sans matrice "
       "hessienne.",
       "Ex 5.11, Ex 5.15 · ✏️ 5.5 · fiche §5.4 (les points où le gradient s'annule)", thread="synthétique",
       tracks="M, C", mylearn="calculus.py",
       body=MYLEARN_SHORT + r"""

Écris `classify_critical_point(f, x, h=1e-3, tol=1e-6, grad_tol=1e-4)` (lis sa docstring) :
- `x` doit être à une dimension, sinon `ValueError` ; si la norme de `numerical_gradient(f, x)` (ta fonction de 5.15, avec son pas par défaut) dépasse `grad_tol`, `x` n'est pas un point critique : `ValueError` ;
- les directions $\mathbf{d}$ : les axes $\mathbf{e}_i$, puis les diagonales $\mathbf{e}_i + \mathbf{e}_j$ et $\mathbf{e}_i - \mathbf{e}_j$ pour $i < j$ (`np.eye(n)` donne les $\mathbf{e}_i$) ;
- dans chaque direction, la différence seconde $D_\mathbf{d} = \frac{f(\mathbf{x} + h\mathbf{d}) - 2f(\mathbf{x}) + f(\mathbf{x} - h\mathbf{d})}{h^2}$ ; une valeur telle que $|D_\mathbf{d}| \le$ `tol` compte comme nulle ;
- renvoie `"minimum"` si toutes les $D_\mathbf{d}$ sont $>$ `tol`, `"maximum"` si toutes sont $<$ `-tol`, `"saddle"` s'il y en a des deux signes, et `"flat"` sinon (on ne peut pas conclure).

Vérifications sur $h(\mathbf{v}) = v_0^4 + v_1^4 - 4\,v_0 v_1$ (`h_24`, dont les points critiques sont $(0, 0)$, $(1, 1)$ et $(-1, -1)$) et sur $k(\mathbf{v}) = v_0^3 + v_1^2$ (`k_24`). La cellule de vérification appelle ta fonction :
a) en $(0, 0)$, pour $h$ ;
b) en $(1, 1)$, pour $h$ ;
c) en $(0, 0)$, pour $k$ ;
d) le nom de l'erreur levée en $(1, 0)$, pour $h$ ;
puis les tests de `classify_critical_point`.

Dans tes notes : en $(0, 0)$, $h$ ne varie le long des axes qu'en $v^4$ : quelles directions décident du verdict ? Le résultat de c) dit-il que $(0, 0)$ est un minimum de $k$ ? Que vaut $k$ en $(-0{,}1 ;\ 0)$ ?""",
       given=r'''def h_24(v):
    """v0⁴ + v1⁴ - 4 v0 v1: critical points (0, 0), (1, 1) and (-1, -1)."""
    return v[0] ** 4 + v[1] ** 4 - 4 * v[0] * v[1]


def k_24(v):
    """v0³ + v1²."""
    return v[0] ** 3 + v[1] ** 2''',
       check=RELOAD + r'''with wb.attempt("5.24"):
    wb.check("5.24a", mylearn.calculus.classify_critical_point(h_24, [0.0, 0.0]), computed=True)
    wb.check("5.24b", mylearn.calculus.classify_critical_point(h_24, [1.0, 1.0]), computed=True)
    wb.check("5.24c", mylearn.calculus.classify_critical_point(k_24, [0.0, 0.0]), computed=True)
    wb.check("5.24d", error_name(mylearn.calculus.classify_critical_point, h_24, [1.0, 0.0]), computed=True)
    run_calculus_tests("test_classify_critical_point_")''',
       solution=r'''kinds_24 = [mylearn.calculus.classify_critical_point(h_24, [0.0, 0.0]),
            mylearn.calculus.classify_critical_point(h_24, [1.0, 1.0]),
            mylearn.calculus.classify_critical_point(k_24, [0.0, 0.0])]
error_24 = error_name(mylearn.calculus.classify_critical_point, h_24, [1.0, 0.0])
hessian_24 = np.array([[12.0, -4.0], [-4.0, 12.0]])          # the second derivatives of h at (1, 1), by hand
print(kinds_24, error_24, "eigenvalues of the Hessian of h at (1, 1):", np.linalg.eigvalsh(hessian_24),
      "k(-0.1, 0) =", k_24([-0.1, 0.0]))
run_calculus_tests("test_classify_critical_point_", impl="ref")''',
       record=r'''wb.record("5.24a", kinds_24[0], mistakes={"tu n'as regardé que les axes, où h ne varie qu'en v⁴ : le long des diagonales, h descend dans un sens (v0 = v1) et monte dans l'autre": "minimum",
                                          "le long des axes, h est presque plate ; regarde aussi les diagonales, où elle descend dans un sens et monte dans l'autre": "flat"})
wb.record("5.24b", kinds_24[1], mistakes={"en (1, 1), h monte dans toutes les directions : vérifie le signe de tes différences secondes": "saddle"})
wb.record("5.24c", kinds_24[2], mistakes={"le long de l'axe v0, k vaut v³ : la différence seconde y est nulle, on ne peut pas conclure (et ce n'est pas un minimum : k(−0,1 ; 0) < 0)": "minimum"})
wb.record("5.24d", error_24, mistakes={"(1, 0) n'est pas un point critique (le gradient y vaut (4, −4)) : la fonction doit le refuser par une ValueError": "no error"})''',
       note="En $(0, 0)$, $h$ vaut $v^4$ le long de chaque axe : la différence seconde y est minuscule ($2h^2 = 2 "
            "\\times 10^{-6}$ avec $h = 10^{-3}$, juste au-dessus de `tol`). Ce sont les diagonales qui tranchent : "
            "le long de $v_0 = v_1$, $h \\approx -4t^2$ descend ; le long de $v_0 = -v_1$, $h \\approx 4t^2$ monte : "
            "c'est une selle. En $(1, 1)$, la matrice des dérivées secondes $\\begin{pmatrix} 12 & -4 \\\\ -4 & 12 "
            "\\end{pmatrix}$ a deux valeurs propres positives (8 et 16) : un minimum. En $(0, 0)$, $k$ monte "
            "dans la direction $v_1$, mais vaut $v_0^3$ le long de l'axe $v_0$ : la différence seconde y est nulle, "
            "et la fonction répond « plat », c'est-à-dire « je ne peux pas conclure ». À raison : $k(-0{,}1 ;\\ 0) = "
            "-0{,}001 < 0 = k(0, 0)$, ce n'est pas un minimum. Comme pour $x^3$ en 0 (piège classique de la fiche), "
            "une dérivée seconde nulle ne permet pas de conclure. Regarder les axes et les diagonales ne suffit pas "
            "toujours : une forme quadratique peut monter le long de toutes ces directions et descendre entre elles "
            "(voir `05_solutions.md`) ; la matrice hessienne (ch. 19) règle le cas général."),

    Ex("5.25", "🏆", 3, 60, "Atteindre le fond de la vallée de Rosenbrock",
       "régler un learning rate, ou un petit programme de learning rates, pour atteindre et tenir le minimum d'une "
       "vallée mal conditionnée.",
       "Ex 5.18, Ex 5.19 · ∂ 5.7 · fiche §5.4 (Rosenbrock)", thread="Rosenbrock", tracks="C",
       body=r"""**Défi.** Pars de $(-1{,}5 ;\ 2)$ et descends la vallée de Rosenbrock ($a = 1$, $b = 100$) jusqu'au minimum $(1, 1)$, avec ta fonction `gradient_descent` et le gradient exact `rosenbrock_gradient`. Écris `schedule_25()`, qui renvoie ton **programme** : une liste de 1 à 3 phases `(lr, n_steps)`, jouées l'une après l'autre, chaque phase repartant du point où la précédente s'est arrêtée. Règles :
- chaque phase compte au moins 100 pas, et le total au plus **10 000 pas** ;
- à la fin, le point est à moins de $10^{-3}$ de $(1, 1)$ (distance euclidienne) ;
- et il y **reste** : 10 000 pas de plus avec le learning rate de la dernière phase ne doivent jamais l'en éloigner de $10^{-3}$ ou plus, et doivent finir à moins de $10^{-4}$ de $(1, 1)$.

Commence par mesurer, avec un seul learning rate : combien de pas faut-il avec 0,001 ? avec 0,0015 ? avec 0,002 ? Pistes : la leçon du bol de 5.19 (le plus grand learning rate possible dépend de la courbure la plus forte) s'applique au fond de la vallée, où la courbure la plus forte vaut environ 1 000 (∂ 5.7) ; et la règle « il y reste » n'est pas là pour rien.""",
       given=CHALLENGE_25,
       todo=r'''def schedule_25():
    """Your programme: a list of 1 to 3 phases (lr, n_steps), played one after the other from START_25."""
    raise NotImplementedError("schedule_25() is not written yet")''',
       check=RELOAD + r'''with wb.attempt("5.25"):
''' + GRADE_25,
       solution=r'''def steps_to_reach_25(lr, max_steps=20_000):
    """Number of steps of YOUR gradient_descent from START_25 until the distance to (1, 1) is < 1e-3 (None if never)."""
    with np.errstate(over="ignore", invalid="ignore"):
        _, path = mylearn.calculus.gradient_descent(rosenbrock_gradient, START_25, lr=lr, n_steps=max_steps)
        close = np.flatnonzero(np.linalg.norm(path - TARGET_25, axis=1) < 1e-3)
    return int(close[0]) if close.size else None


for lr in (0.001, 0.0015, 0.0018, 0.0019, 0.00194, 0.00195, 0.002, 0.0021):
    print(f"lr = {lr}: {steps_to_reach_25(lr)} steps")
print("lr = 0.002 reaches (1, 1), but does it stay? (steps, final distance, worst and last distances afterwards):",
      grade_25([(0.002, 6000)]))
print("eigenvalues of the Hessian at (1, 1):", np.linalg.eigvalsh([[802.0, -400.0], [-400.0, 200.0]]))


def schedule_25():
    """Your programme: a list of 1 to 3 phases (lr, n_steps), played one after the other from START_25."""
    return [(0.0019, 8500)]


print("two phases, (0.002, 5000) then (0.0019, 1000):", grade_25([(0.002, 5000), (0.0019, 1000)]))
with wb.attempt("5.25"):
''' + GRADE_25,
       note="Avec un seul learning rate : 0,001 demande environ 17 700 pas, 0,0015 environ 11 500 ; il faut monter "
            "vers 0,0018 (9 500 pas) ou 0,0019 (8 200 pas) pour tenir dans le budget. Près de la limite, le nombre de "
            "pas varie de façon irrégulière (3 300 pour 0,00194, 4 700 pour 0,00195, 5 400 pour 0,002) : partie de la "
            "paroi très raide, où le gradient vaut environ 160, la goutte fait d'abord quelques sauts énormes, et "
            "l'endroit où elle retombe dans la vallée change beaucoup avec $\\eta$. Avec 0,00194, elle atterrit "
            "presque sur $(1, 1)$ en 6 pas ; avec 0,0019, vers $(0{,}4 ;\\ 0{,}6)$, et il lui reste toute la vallée "
            "à parcourir. Avec 0,002, la goutte atteint $(1, 1)$, mais n'y reste pas : une oscillation en travers de "
            "la vallée grandit, jusqu'à $4 \\times 10^{-3}$. Au fond de la vallée, la matrice des dérivées secondes $\\begin{pmatrix} 802 & -400 \\\\ -400 "
            "& 200 \\end{pmatrix}$ a pour valeurs propres environ 1 001,6 (en travers) et 0,4 (le long du fond) : "
            "comme sur le bol de 5.19, l'oscillation en travers ne s'amortit que si $\\eta \\times 1\\,001{,}6 < 2$, "
            "soit $\\eta < 0{,}001997$. Juste au-dessus, elle grandit si lentement que 10 000 pas ne le montrent pas ; "
            "à 0,002, elle se voit déjà. Le long du fond, chaque pas ne multiplie l'écart que par $1 - 0{,}4\\,\\eta "
            "\\approx 0{,}9992$ : d'où les milliers de pas. Un programme en deux phases réussit aussi : (0,002 ; 5 000), "
            "puis (0,0019 ; 1 000), qui amortit l'oscillation. C'est l'idée des programmes de learning rate "
            "(*schedules*) et des optimiseurs adaptatifs du ch. 19 : la vallée de Rosenbrock y reviendra."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 5.1 à 5.5 | vérifier tes exercices papier ✏️ | ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")          # a "|" in a title must not split the table row
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if kind == "exercise":
        title = "# 5 · Courbes et surfaces — notebook d'exercices"
        how = ("La partie 0 vérifie tes exercices papier. Chaque exercice de code : un énoncé, une cellule à "
               "compléter (les `...` et les `raise NotImplementedError`), puis une cellule de vérification "
               "(`wb.check`, ou les tests de ta librairie `mylearn`). « Exécuter tout » va jusqu'au bout même si rien "
               "n'est rempli : ce qui n'est pas fait affiche ⏳. Les questions « dans tes notes » qui n'ont pas de "
               "cellule 📝 se notent dans la section « Notes sur le notebook » de ta copie de `06_mes_reponses.md`. "
               "Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch05_courbes/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 5`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 5 · Courbes et surfaces — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Estimer une dérivée et un gradient par différences finies, et choisir le pas $h$ (troncature contre "
               "arrondi).\n"
               "- Repérer les extrema d'une courbe échantillonnée, et classer les points critiques d'une surface "
               "(minimum, maximum, point selle).\n"
               "- Écrire et régler une descente de gradient : le learning rate, les courbures, les points selles.\n"
               "- Comparer le gradient numérique et la différentiation automatique (`torch.autograd`).\n\n"
               "**Rappel express.** Pente centrée $\\frac{f(x + h) - f(x - h)}{2h}$ (erreur en $h^2$), avant "
               "$\\frac{f(x + h) - f(x)}{h}$ (erreur en $h$) ; dérivée seconde $\\frac{f(x + h) - 2f(x) + f(x - h)}{h^2}$ ; "
               "gradient $\\nabla f = \\left(\\frac{\\partial f}{\\partial x_1}, \\ldots, \\frac{\\partial f}{\\partial "
               "x_n}\\right)$, perpendiculaire aux lignes de niveau, vers la plus grande montée ; descente "
               "$\\mathbf{x} \\leftarrow \\mathbf{x} - \\eta\\,\\nabla f(\\mathbf{x})$. En Python : `np.array(x, "
               "dtype=float)` (une copie), `np.linalg.norm`, `np.meshgrid`, `ax.contour`, `ax.quiver`, "
               "`torch.tensor(..., requires_grad=True)` et `.backward()`.")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu estimer une dérivée par différences finies, et expliquer comment choisir le pas $h$ ?\n"
               "2. Sais-tu calculer un gradient, le lire sur une carte de lignes de niveau, et classer un point "
               "critique ?\n"
               "3. Sais-tu écrire une descente de gradient, et reconnaître un learning rate trop petit ou trop grand ?\n\n"
               "**Pour aller plus loin** : les références de la fiche (la vidéo de 3Blue1Brown sur la descente de "
               "gradient, l'article de *Distill* sur les vallées mal proportionnées, l'introduction à `torch.autograd`). "
               "La suite : le ch. 6 (information et entropie), puis "
               "le ch. 18, où `numerical_gradient` vérifiera tes rétropropagations, et le ch. 19, où la vallée de "
               "Rosenbrock et `gradient_descent` serviront à comparer les optimiseurs.")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT, PAPER_INTRO)
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
