#!/usr/bin/env python
"""Build the two notebooks of chapter 2 from a single source (used by Claude).

    python tools/chapters/build_ch02.py
    python tools/run_all_notebooks.py chapitres/ch02_stats/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch02_stats/03_notebook.ipynb

Part 0 checks the ✏️ paper exercises (2.1 to 2.7); parts A to D hold the notebook
exercises 2.13 to 2.32 (mylearn.stats, the bootstrap, covariance and correlation, Anscombe).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, guarded, md, paper_cells,  # noqa: E402
                         part_cells, setup_cell, write_notebook)

CHAPTER = "2"
FOLDER = "chapitres/ch02_stats"

# ---------------------------------------------------------------------------
# Part 0: checks of the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The paper exercises only use the numbers of their statements (02_exercices.md)
import math

import numpy as np

SALARIES = np.array([2300, 12000, 2100, 1900, 2800, 2400, 3500, 2100, 2600])     # 2.1
CARS = np.array([176, 144, 224, 96, 160])            # 2.2: sedans, pick-ups, minivans, SUVs, estates
CAR_TYPES = ["berline", "pick-up", "monospace", "SUV", "break"]
LOADING = np.array([4, 7, 6, 3, 10])                  # 2.4
STUDY = np.array([1, 2, 3, 4, 5])                     # 2.7: hours of revision
GRADES = np.array([2, 3, 5, 4, 6])                    # 2.7: grades'''

PAPER = [
    Paper("2.1", "Moyenne, médiane et mode d'une liste de salaires", [
        ("a", "the mean salary, rounded to the euro", "round(SALARIES.mean())",
         'mistakes={"il y a 9 salariés : divise la somme par 9": round(SALARIES.sum() / 10), '
         '"c\'est la médiane : on demande la moyenne": 2400}'),
        ("b", "the median", "int(np.median(SALARIES))",
         'mistakes={"trie d\'abord les salaires : la médiane est la valeur du milieu de la liste TRIÉE": 2800, '
         '"c\'est la moyenne : on demande la médiane": round(SALARIES.mean())}'),
        ("c", "the mode", "2100",
         'mistakes={"le mode est la valeur la plus FRÉQUENTE, pas celle du milieu": 2400}'),
        ("d", "how many employees earn LESS than the mean", "int((SALARIES < SALARIES.mean()).sum())",
         'mistakes={"on demande ceux qui gagnent MOINS que la moyenne": 1, '
         '"compare chaque salaire à la moyenne, pas à la médiane": 4}'),
        ("e", "the new mean with 30 000 instead of 12 000, rounded to the euro",
         "round(np.where(SALARIES == 12000, 30000, SALARIES).mean())",
         'mistakes={"la moyenne dépend de TOUS les salaires : recalcule-la avec le nouveau": round(SALARIES.mean())}'),
        ("f", "the new median", "int(np.median(np.where(SALARIES == 12000, 30000, SALARIES)))",
         'mistakes={"c\'est la nouvelle moyenne : on demande la médiane": '
         'round(np.where(SALARIES == 12000, 30000, SALARIES).mean()), '
         '"trie d\'abord les salaires : la médiane est la valeur du milieu de la liste TRIÉE": 2800}'),
    ]),
    Paper("2.2", "De la casse de voitures à la distribution de probabilité", [
        ("a", "P(minivan), 2 decimals", "CARS[2] / CARS.sum()",
         'decimals=2, mistakes={"divise le nombre de monospaces par le nombre TOTAL de voitures": 224}'),
        ("b", "the 5 probabilities in the order of the statement, a list", "(CARS / CARS.sum()).tolist()",
         'decimals=2, mistakes={"ce sont les comptages : divise chacun par le total": CARS.tolist(), '
         '"exprime les probabilités entre 0 et 1, pas en pourcentage": (100 * CARS / CARS.sum()).tolist()}'),
        ("c", "the angle of the minivan slice, in degrees (1 decimal)", "360 * CARS[2] / CARS.sum()",
         'decimals=1, mistakes={"c\'est un pourcentage : un tour complet de roue fait 360°": 28, '
         '"multiplie la probabilité par 360° (un tour complet)": 0.28}'),
        ("d", "P(not an SUV), 2 decimals", "1 - CARS[3] / CARS.sum()",
         'decimals=2, mistakes={"c\'est la probabilité de tirer un SUV : on demande l\'événement contraire": 0.12}'),
        ("e", "the mean number of pick-ups in 50 draws", "50 * CARS[1] / CARS.sum()",
         'decimals=1, mistakes={"c\'est la probabilité d\'un seul tirage : multiplie-la par le nombre de tirages": 0.18}'),
        ("f", "the cumulative sums of the probabilities, a list", "np.cumsum(CARS / CARS.sum()).tolist()",
         'decimals=2, mistakes={"ce sont les probabilités elles-mêmes : additionne-les au fur et à mesure": '
         '(CARS / CARS.sum()).tolist()}'),
        ("g", "the car type chosen by u = 0.71 (a string)",
         "CAR_TYPES[int(np.searchsorted(np.cumsum(CARS / CARS.sum()), 0.71, side='right'))]",
         'mistakes={"le type choisi est le premier dont la somme cumulée est strictement plus grande que u": "monospace", '
         '"compare u aux sommes cumulées de la question f, dans l\'ordre de l\'énoncé": "break"}'),
    ]),
    Paper("2.3", "La règle 68-95-99,7 sur les nageoires des manchots", [
        ("a", "the interval holding about 68 % of the Adelie, a list [start, end] in mm", "[190 - 6.5, 190 + 6.5]",
         'decimals=1, mistakes={"c\'est l\'intervalle à 2 écarts-types (95 %)": [177, 203], '
         '"c\'est l\'intervalle à 3 écarts-types (99,7 %)": [170.5, 209.5]}'),
        ("b", "the interval holding about 95 % of them", "[190 - 2 * 6.5, 190 + 2 * 6.5]",
         'decimals=1, mistakes={"c\'est l\'intervalle à 1 écart-type (68 %)": [183.5, 196.5], '
         '"c\'est l\'intervalle à 3 écarts-types (99,7 %)": [170.5, 209.5]}'),
        ("c", "the proportion above 203 mm (3 decimals)", "(1 - 0.95) / 2",
         'decimals=3, mistakes={"les 5 % hors de l\'intervalle se partagent entre les DEUX côtés (symétrie)": 0.05, '
         '"c\'est la part EN DESSOUS de 203 mm": 0.975, "c\'est la part DANS l\'intervalle": 0.95, '
         '"l\'énoncé demande d\'utiliser la règle 68-95-99,7, pas une table de la loi normale": 0.023}'),
        ("d", "the expected number of Adelie above 203 mm among 151 (integer)", "round(151 * 0.025)",
         'mistakes={"seuls ceux AU-DESSUS de 203 mm comptent, pas ceux des deux côtés": round(151 * 0.05), '
         '"arrondis au plus proche, ne tronque pas (et utilise la règle 68-95-99,7, pas une table)": 3}'),
        ("e", "the z-score of a 210 mm flipper (2 decimals)", "(210 - 190) / 6.5",
         'decimals=2, mistakes={"arrondis au plus proche, ne tronque pas": 3.07, '
         '"divise l\'écart à la moyenne par l\'écart-type": 20}'),
    ]),
    Paper("2.4", "Variance : diviser par N ou par N − 1 ?", [
        ("a", "the mean", "int(LOADING.mean())", ""),
        ("b", "the sum of the squared deviations", "int(((LOADING - LOADING.mean()) ** 2).sum())",
         'mistakes={"élève chaque écart au carré AVANT d\'additionner (la somme des écarts vaut toujours 0)": 0, '
         '"ce sont les écarts en valeur absolue : il faut leurs CARRÉS": int(np.abs(LOADING - LOADING.mean()).sum())}'),
        ("c", "the variance with ddof = 0", "LOADING.var()",
         'decimals=2, mistakes={"ddof = 0 : divise par n, pas par n − 1": LOADING.var(ddof=1), '
         '"c\'est l\'écart-type : on demande la variance": LOADING.std()}'),
        ("d", "the variance with ddof = 1", "LOADING.var(ddof=1)",
         'decimals=2, mistakes={"ddof = 1 : divise par n − 1, pas par n": LOADING.var()}'),
        ("e", "the standard deviation with ddof = 0 (3 decimals)", "LOADING.std()",
         'decimals=3, mistakes={"c\'est la variance : prends sa racine carrée": LOADING.var(), '
         '"c\'est l\'écart-type avec ddof = 1": LOADING.std(ddof=1)}'),
        ("f", "the standard deviation with ddof = 1 (3 decimals)", "LOADING.std(ddof=1)",
         'decimals=3, mistakes={"c\'est la variance : prends sa racine carrée": LOADING.var(ddof=1), '
         '"c\'est l\'écart-type avec ddof = 0": LOADING.std()}'),
        ("g", "the mean of the squares", "(LOADING ** 2).mean()",
         'decimals=2, mistakes={"c\'est le carré de la moyenne : on demande la moyenne des carrés": 36, '
         '"c\'est la somme des carrés : divise-la par le nombre de valeurs": int((LOADING ** 2).sum())}'),
        ("h", "the variance (ddof = 0) after adding 100 to every value", "(LOADING + 100).var()",
         'decimals=2, mistakes={"ajouter 100 décale les valeurs ET leur moyenne : les écarts ne changent pas": 106}'),
        ("i", "the standard deviation (ddof = 0) in milliseconds (1 decimal)", "(LOADING * 1000).std()",
         'decimals=1, mistakes={"c\'est la variance : elle est multipliée par 1 000², l\'écart-type par 1 000": '
         '(LOADING * 1000).var(), "multiplier les valeurs par 1 000 multiplie aussi les écarts": LOADING.std(), '
         '"tu es reparti de l\'écart-type arrondi : pars de sa valeur exacte (la racine de la variance)": 2449.0}'),
    ]),
    Paper("2.5", "Espérances : Bernoulli, multinoulli et jeu de hasard", [
        ("a", "E[X] for the coin (heads = 1)", "0.3", "decimals=2"),
        ("b", "Var(X), 2 decimals", "0.3 * (1 - 0.3)",
         'decimals=2, mistakes={"c\'est l\'espérance : la variance d\'une loi de Bernoulli vaut p(1 − p)": 0.3, '
         '"c\'est p² : la variance vaut p(1 − p)": 0.09}'),
        ("c", "the mean number of heads in 200 throws", "200 * 0.3",
         'decimals=1, mistakes={"c\'est l\'espérance d\'un seul lancer : multiplie par le nombre de lancers": 0.3}'),
        ("d", "the expected value of a fair 20-sided die", "sum(range(1, 21)) / 20",
         'decimals=1, mistakes={"la moyenne de 1, 2, …, 20 vaut (1 + 20) / 2": 10, '
         '"divise la somme des faces par le nombre de faces": sum(range(1, 21))}'),
        ("e", "the expected class number (1 decimal)", "1 * 0.1 + 2 * 0.2 + 3 * 0.3 + 4 * 0.4",
         'decimals=1, mistakes={"pondère chaque numéro par SA probabilité (ce n\'est pas la moyenne de 1, 2, 3, 4)": 2.5}'),
        ("f", "the one-hot vector of class 3, a list", "[0, 0, 1, 0]",
         'mistakes={"les classes sont numérotées de 1 à 4 : la classe 3 est la troisième composante": [0, 0, 0, 1]}'),
        ("g", "the limit of the average one-hot vector, a list", "[0.1, 0.2, 0.3, 0.4]",
         'decimals=2, mistakes={"les classes ne sortent pas toutes aussi souvent : pense à la fréquence de chacune": '
         '[0.25, 0.25, 0.25, 0.25]}'),
        ("h", "the mean gain per game, stake deducted (2 decimals)", "20 / 20 + 5 * 3 / 20 - 2",
         'decimals=2, mistakes={"déduis la mise de 2 € du gain moyen": 20 / 20 + 5 * 3 / 20, '
         '"la mise n\'est jamais rendue : tu paies 2 € à chaque partie, même quand tu gagnes": 0.15}'),
        ("i", "the organiser's mean gain over 400 games, in euros", "-400 * (20 / 20 + 5 * 3 / 20 - 2)",
         'decimals=1, mistakes={"le gain de l\'organisateur est l\'opposé de celui du joueur": 400 * (20 / 20 + 5 * 3 / 20 - 2), '
         '"c\'est le gain moyen d\'une seule partie : multiplie par le nombre de parties": 0.25}'),
    ]),
    Paper("2.6", "Compter les tirages avec et sans remise", [
        ("a", "without replacement, order ignored: number of new datasets", "math.comb(5, 2)",
         'mistakes={"sans tenir compte de l\'ordre, AB et BA forment le même jeu": 20}'),
        ("b", "with replacement, order ignored", "math.comb(5 + 2 - 1, 2)",
         'mistakes={"sans tenir compte de l\'ordre, AB et BA ne comptent qu\'une fois": 25, '
         '"avec remise, AA, BB… sont possibles aussi": 10}'),
        ("c", "with replacement, order counted", "5 ** 2",
         'mistakes={"ici l\'ordre compte : AB et BA sont deux suites différentes": 15, '
         '"avec remise, un exemple peut sortir deux fois (AA)": 20}'),
        ("d", "7 examples, draw 3 without replacement, order ignored", "math.comb(7, 3)",
         'mistakes={"sans tenir compte de l\'ordre : chaque groupe de 3 correspond à 3! = 6 suites": 210, '
         '"sans remise, un exemple ne peut pas sortir deux fois": 343}'),
        ("e", "number of possible bootstrap sequences of the 5 examples", "5 ** 5",
         'mistakes={"il y a 5 tirages, pas 2": 25, "avec remise : chacun des 5 tirages a 5 issues possibles": 120}'),
        ("f", "P(a given example is in none of the 5 draws), 4 decimals", "(1 - 1 / 5) ** 5",
         'decimals=4, mistakes={"on veut l\'absence à CHACUN des 5 tirages (indépendants)": 0.8, '
         '"c\'est la probabilité d\'apparaître AU MOINS une fois": 1 - 0.8 ** 5, '
         '"c\'est la probabilité d\'être tiré à un tirage donné": 0.2, '
         '"c\'est la limite quand n devient très grand ; ici n = 5 : calcule la valeur exacte": math.exp(-1)}'),
        ("g", "the same probability for 1 000 draws among 1 000 (4 decimals)", "(1 - 1 / 1000) ** 1000",
         'decimals=4, mistakes={"c\'est la limite quand n devient infini : calcule la valeur exacte pour n = 1 000": '
         'math.exp(-1)}'),
        ("h", "the mean share of examples drawn at least once (4 decimals)", "1 - (1 - 1 / 1000) ** 1000",
         'decimals=4, mistakes={"c\'est la part des exemples ABSENTS du rééchantillon": (1 - 1 / 1000) ** 1000, '
         '"c\'est la limite 1 − e^(−1) : calcule la valeur exacte pour n = 1 000": 1 - math.exp(-1)}'),
    ]),
    Paper("2.7", "Covariance et corrélation de cinq points à la main", [
        ("a", "the means of x and y, a list", "[STUDY.mean(), GRADES.mean()]", "decimals=1"),
        ("b", "the sum of the products of the deviations",
         "float(((STUDY - STUDY.mean()) * (GRADES - GRADES.mean())).sum())",
         'decimals=1, mistakes={"additionne les PRODUITS des écarts de x et de y, pas les écarts eux-mêmes": 0, '
         '"c\'est déjà la covariance (divisée par n) : on demande la somme des produits": 1.8}'),
        ("c", "the covariance (ddof = 0)", "np.cov(STUDY, GRADES, ddof=0)[0, 1]",
         'decimals=2, mistakes={"divise la somme des produits par le nombre de points": 9, '
         '"ddof = 0 : divise par n, pas par n − 1": np.cov(STUDY, GRADES, ddof=1)[0, 1]}'),
        ("d", "the covariance with ddof = 1 (2 decimals)", "np.cov(STUDY, GRADES, ddof=1)[0, 1]",
         'decimals=2, mistakes={"ddof = 1 : divise par n − 1, pas par n": np.cov(STUDY, GRADES, ddof=0)[0, 1]}'),
        ("e", "the standard deviations of x and y, a list (3 decimals)", "[STUDY.std(), GRADES.std()]",
         'decimals=3, mistakes={"ce sont les variances : prends leur racine carrée": [STUDY.var(), GRADES.var()]}'),
        ("f", "the correlation r (3 decimals)", "np.corrcoef(STUDY, GRADES)[0, 1]",
         'decimals=3, mistakes={"tu as mélangé ddof = 1 (covariance) et ddof = 0 (écarts-types) : une corrélation ne '
         'dépasse jamais 1": np.cov(STUDY, GRADES, ddof=1)[0, 1] / (STUDY.std() * GRADES.std()), '
         '"divise la covariance par le PRODUIT des deux écarts-types": 0.45, '
         '"tu as mélangé ddof = 0 (covariance) et ddof = 1 (écarts-types) : garde le même ddof partout": '
         'np.cov(STUDY, GRADES, ddof=0)[0, 1] / (STUDY.std(ddof=1) * GRADES.std(ddof=1))}'),
        ("g", "the covariance with x in minutes", "np.cov(STUDY * 60, GRADES, ddof=0)[0, 1]",
         'decimals=1, mistakes={"la covariance dépend de l\'unité de x : elle change": 1.8, '
         '"seul x est multiplié par 60, pas y": 1.8 * 3600}'),
        ("h", "the correlation with x in minutes (3 decimals)", "np.corrcoef(STUDY * 60, GRADES)[0, 1]",
         'decimals=3, mistakes={"c\'est la covariance : la corrélation divise par les écarts-types": 108}'),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes exercices papier (✏️ de 2.1 à 2.7)\n\n"
               "Fais d'abord les exercices ✏️ de `02_exercices.md` **sur papier**, dans ta copie de "
               "`06_mes_reponses.md`. Reporte ensuite chaque réponse ici : **la valeur** que tu as trouvée "
               "(par exemple `42`, `0.125`, `[7, -2]` ou `\"texte\"`), pas l'expression Python, sinon tu ne vérifies "
               "rien. Arrondis comme l'énoncé le demande. Les réponses pas encore remplies affichent ⏳. "
               "L'exercice ∂ 2.8 se corrige avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Part A: summarising data (2.13 to 2.18)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools and data for the notebook exercises (parts A to D)
import doctest
import inspect
import math
import re

import pytest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

penguins = wb.datasets.load_penguins()                    # the 344 penguins, a few values missing
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
measured = penguins.dropna(subset=MEASURES).reset_index(drop=True)   # the 342 penguins with their 4 measures
X_measures = measured[MEASURES].to_numpy()                # shape (342, 4)
mass = measured["body_mass_g"].to_numpy()                 # in grams
flipper = measured["flipper_length_mm"].to_numpy()        # in mm
P_CARS = np.array([176, 144, 224, 96, 160]) / 800         # the scrapyard of 2.2: probability of each type
CAR_TYPES = ["berline", "pick-up", "monospace", "SUV", "break"]


def verdict(ex_id, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message."""
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")


def fr(value, decimals=2):
    """A number written the French way, for the messages: fr(0.25) -> '0,25'."""
    return f"{value:.{decimals}f}".replace(".", ",")


def run_stats_tests(keyword, impl="learner"):
    """Run the tests of mylearn.stats selected by `keyword` (on YOUR code by default)."""
    command = [sys.executable, "-m", "pytest", "tests/test_ch02_stats.py", "-k", keyword, "-q",
               "-p", "no:cacheprovider", "--color=no", "-rf", "--tb=line"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "200"})   # long lines: the reason of each failure
    lines = result.stdout.strip().splitlines()
    for line in [line for line in lines if line.startswith("FAILED")][:8]:
        print(line[:200])
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


print(f"{len(penguins)} penguins, {len(measured)} with their 4 measures")'''

MYLEARN_HOWTO = ("> **Mode d'emploi mylearn** (comme en 0A et en 0B) : ouvre `mon_travail/mylearn/stats.py` (créé par "
                 "`python tools/start_chapter.py 2`), lis la docstring de chaque fonction, remplace les "
                 "`raise NotImplementedError(...)` par ton code et **enregistre**. NumPy est permis (`np.asarray`, "
                 "`.sum()`, `np.sort`, `np.sqrt`, `np.cumsum`…), mais pas la fonction qui ferait tout le travail à ta "
                 "place : `np.mean`, `np.median`, `np.var`, `np.std`, `np.percentile`, `np.histogram`, `np.cov`, "
                 "`np.corrcoef` et `rng.choice` sont les **oracles** des tests. Une fonction peut réutiliser tes autres "
                 "fonctions (`std` appelle `variance`). Chaque fonction commence par convertir et vérifier son entrée "
                 "(`np.asarray(x, dtype=float)` ; `ValueError` si c'est vide ou s'il y a des NaN, `np.isnan`) : écris "
                 "une petite fonction d'aide, par exemple `_as_numbers(x)`, et appelle-la dans les fonctions de calcul "
                 "(pas dans `mode` ni dans `sample`, qui acceptent aussi des chaînes de caractères). La cellule de "
                 "vérification recharge ta librairie, vérifie quelques valeurs, puis lance les tests de ces "
                 "fonctions ; `python -m pytest tests/test_ch02_stats.py -q` les lance tous dans un terminal.")

MYLEARN_SHORT = ("> **mylearn** : dans `mon_travail/mylearn/stats.py`, mêmes règles qu'en 2.13 (NumPy permis, sauf les "
                 "fonctions oracles). Enregistre, puis relance la cellule de vérification.")

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="2")   # reload your saved file\n'

SNIPPETS_2_14 = r'''print("a)", np.random.default_rng(7).integers(1, 7, size=10))
print("  ", np.random.default_rng(7).integers(1, 7, size=10))

rng = np.random.default_rng(7)
print("b)", rng.integers(1, 7, size=10))
print("  ", rng.integers(1, 7, size=10))

print("c)", np.random.default_rng().integers(1, 7, size=10))
print("  ", np.random.default_rng().integers(1, 7, size=10))

np.random.seed(0)
print("d)", np.random.default_rng().integers(1, 7, size=10))
np.random.seed(0)
print("  ", np.random.default_rng().integers(1, 7, size=10))

rng = np.random.default_rng(7)
first, rest = rng.random(3), rng.random(5)
print("e)", np.concatenate([first, rest]).round(3))
print("  ", np.random.default_rng(7).random(8).round(3))

rng = np.random.default_rng(7)
noise = rng.normal(size=3)       # a new line added at the top of an experiment...
print("f)", rng.permutation(6))
rng = np.random.default_rng(7)
print("  ", rng.permutation(6))  # ...compared with the old version, without it'''


PART_A = Part("A", "Résumer des données : centre, dispersion, histogramme, lois",
              "Fiche §2.2 et §2.3. Tu écris les premières fonctions de `mylearn.stats` (centre, dispersion, "
              "histogramme) et tu les appliques aux 342 manchots dont les quatre mesures sont connues ; puis tu "
              "compares les lois usuelles et la règle 68-95-99,7 à des tirages et à de vraies mesures.",
              given=PART_A_GIVEN, exercises=[
    Ex("2.13", "🔨", 2, 30, "Tendances centrales : `mean`, `median`, `mode`",
       "écrire la moyenne, la médiane et le mode, et voir laquelle résiste à une valeur aberrante.",
       "Ex 2.1 · 0A (NumPy : `axis`, tri, `np.unique`) · fiche §2.2", thread="Penguins", tracks="R, M, C",
       body=MYLEARN_HOWTO + r"""

Écris `mean`, `median` et `mode`.
- `mean(x, axis=None)` : la somme divisée par le nombre de valeurs. Avec `axis`, une moyenne par ligne ou par colonne : `x.sum(axis=axis)` divisé par `x.shape[axis]` (0A).
- `median(x, axis=None)` : trie (`np.sort(x, axis=...)`), puis prends la valeur du milieu, ou la moyenne des deux valeurs du milieu si leur nombre est pair. Le long d'un axe, `np.take(x_sorted, k, axis=axis)` prend l'élément n° `k` de chaque ligne ou de chaque colonne.
- `mode(x)` : compte chaque valeur (`np.unique(x, return_counts=True)`), puis garde **toutes** celles qui atteignent le plus grand compte, triées (fiche, ⚠️ « Le mode quand tout est à égalité »). Elle doit marcher aussi sur des chaînes de caractères : ne convertis pas en `float` dans `mode`.

Vérifications, sur les 342 manchots de `measured` (la cellule de vérification les calcule avec tes fonctions : il n'y a rien à recopier) :
a) `mean(mass)`, la masse moyenne (1 décimale) ;
b) `median(mass)` ;
c) `mode(mass)` : combien de manchots ont cette masse (la vérification l'affiche) ? Pourquoi une masse aussi « ronde » ?
d) `mean(X_measures, axis=0)` : la moyenne de chacune des 4 mesures (1 décimale) ;
e) le `mode` de la colonne `species` ;
f) une erreur de saisie ajoute un manchot de 57 000 g (au lieu de 5 700 g) : de combien la moyenne et la médiane augmentent-elles ? La vérification calcule `[hausse de la moyenne, hausse de la médiane]` avec tes fonctions (1 décimale).

Puis les tests de ces trois fonctions.""",
       check=RELOAD + r'''with wb.attempt("2.13"):
    st = mylearn.stats
    wb.check("2.13a", st.mean(mass))
    wb.check("2.13b", st.median(mass))
    mode_13 = st.mode(mass)
    wb.check("2.13c", mode_13)
    print(f"   the mode {mode_13} is the mass of {int(np.sum(mass == mode_13[0]))} penguins")
    wb.check("2.13d", st.mean(X_measures, axis=0))
    wb.check("2.13e", st.mode(measured["species"].to_numpy()))
    with_typo = np.append(mass, 57_000)
    wb.check("2.13f", [st.mean(with_typo) - st.mean(mass), st.median(with_typo) - st.median(mass)])
    run_stats_tests("test_mean_ or test_median_ or test_mode_")''',
       solution=r'''st = mylearn.stats
with_typo = np.append(mass, 57_000)
print("mean:", st.mean(mass), "· median:", st.median(mass), "· mode:", st.mode(mass))
print("mean of each measure:", st.mean(X_measures, axis=0).round(1), "· most frequent species:",
      st.mode(measured["species"].to_numpy()))
print("increase of the mean:", st.mean(with_typo) - st.mean(mass), "· of the median:",
      st.median(with_typo) - st.median(mass))
print(pd.Series(mass).value_counts().head(4).to_dict())       # the most frequent masses: round numbers
run_stats_tests("test_mean_ or test_median_ or test_mode_", impl="ref")''',
       record=r'''wb.record("2.13a", st.mean(mass), decimals=1, mistakes={"c'est la médiane : on demande la moyenne": st.median(mass)})
wb.record("2.13b", st.median(mass), decimals=1, mistakes={"c'est la moyenne : on demande la médiane": st.mean(mass),
                                                            "trie les masses avant de prendre les deux valeurs du milieu": (mass[170] + mass[171]) / 2})
wb.record("2.13c", st.mode(mass), decimals=1, mistakes={"c'est le NOMBRE d'apparitions : mode renvoie la valeur (ou les valeurs) la plus fréquente": [12.0]})
wb.record("2.13d", st.mean(X_measures, axis=0), decimals=1)
wb.record("2.13e", str(st.mode(measured["species"].to_numpy())[0]))
with_typo = np.append(mass, 57_000)
increases_13 = [st.mean(with_typo) - st.mean(mass), st.median(with_typo) - st.median(mass)]
wb.record("2.13f", increases_13, decimals=1, mistakes={"ce sont les nouvelles valeurs : on demande de combien elles AUGMENTENT (fais la différence)": [st.mean(with_typo), st.median(with_typo)]})''',
       note="La moyenne (4 202 g) dépasse la médiane (4 050 g) : la distribution des masses est étirée vers la "
            "droite par les Gentoo, plus lourds. Le mode, 3 800 g, n'est porté que par 12 manchots sur 342 : les "
            "masses sont arrondies à 25 g près (et le plus souvent à 50 g), si bien que les valeurs « rondes » se "
            "répètent ; sur une mesure continue, le mode dépend de l'arrondi plus que des manchots. Une seule erreur "
            "de saisie fait monter la moyenne de 154 g et laisse la médiane à 4 050 g : la médiane est **robuste**. "
            "La référence est dans `solutions/mylearn_ref/stats.py` : lis-la **après** avoir réussi les tests."),

    Ex("2.14", "🔮", 1, 10, "Graine fixée ou graine libre ?",
       "prévoir quand deux tirages pseudo-aléatoires redonnent les mêmes nombres.",
       "0A (`np.random.default_rng`) · fiche §2.2.1 · livre §2.2.1", thread="synthétique", tracks="C",
       hypothesis=True,
       body=r"""Chaque cas ci-dessous affiche deux lignes de tirages. **Sans rien exécuter**, prévois pour chacun si les deux lignes seront identiques : `True` (oui, identiques) ou `False` (non, différentes).

```python
# a)
print(np.random.default_rng(7).integers(1, 7, size=10))
print(np.random.default_rng(7).integers(1, 7, size=10))

# b)
rng = np.random.default_rng(7)
print(rng.integers(1, 7, size=10))
print(rng.integers(1, 7, size=10))

# c)
print(np.random.default_rng().integers(1, 7, size=10))
print(np.random.default_rng().integers(1, 7, size=10))

# d)
np.random.seed(0)
print(np.random.default_rng().integers(1, 7, size=10))
np.random.seed(0)
print(np.random.default_rng().integers(1, 7, size=10))

# e)
rng = np.random.default_rng(7)
first, rest = rng.random(3), rng.random(5)
print(np.concatenate([first, rest]))
print(np.random.default_rng(7).random(8))

# f)
rng = np.random.default_rng(7)
noise = rng.normal(size=3)       # a new line added at the top of an experiment...
print(rng.permutation(6))
rng = np.random.default_rng(7)
print(rng.permutation(6))        # ...compared with the old version, without it
```

Écris ton hypothèse (cellule 📝), puis tes six prédictions ; la vérification ne regarde que tes prédictions. Ensuite seulement, exécute l'**Expérience**.""",
       todo=r'''prediction_2_14a = ...   # True if the two lines will be identical, False otherwise
prediction_2_14b = ...
prediction_2_14c = ...
prediction_2_14d = ...
prediction_2_14e = ...
prediction_2_14f = ...''',
       check=r'''for letter, prediction in zip("abcdef", [prediction_2_14a, prediction_2_14b, prediction_2_14c,
                                         prediction_2_14d, prediction_2_14e, prediction_2_14f]):
    wb.check(f"2.14{letter}", prediction)''',
       solution=r'''prediction_2_14a, prediction_2_14b, prediction_2_14c = True, False, False
prediction_2_14d, prediction_2_14e, prediction_2_14f = False, True, False''',
       record=r'''wb.record("2.14a", prediction_2_14a, mistakes={"deux générateurs créés avec la même graine partent du même état : ils produisent la même suite": False})
wb.record("2.14b", prediction_2_14b, mistakes={"un générateur avance à chaque appel : le second appel continue la suite, il ne la recommence pas": True})
wb.record("2.14c", prediction_2_14c, mistakes={"sans graine, NumPy en prend une nouvelle, imprévisible, à chaque création d'un générateur": True})
wb.record("2.14d", prediction_2_14d, mistakes={"np.random.seed ne règle que l'état global de l'ANCIENNE interface (np.random.rand…) ; regarde ce que reçoit default_rng() entre ses parenthèses": True})
wb.record("2.14e", prediction_2_14e, mistakes={"un générateur produit UNE suite de nombres ; demander 3 puis 5 nombres, ou 8 d'un coup, parcourt-il la même suite ?": False})
wb.record("2.14f", prediction_2_14f, mistakes={"même graine, mais pas le même ordre d'appels : la ligne ajoutée a déjà consommé des nombres de la suite": True})''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", guarded(SNIPPETS_2_14, [f"prediction_2_14{letter}" for letter in "abcdef"],
                               "⏳ Ex 2.14 : écris d'abord tes six prédictions, puis relance cette cellule."))],
       note="Une graine fixe le point de départ d'**un** générateur ; chaque appel avance dans la suite (b), et "
            "c'est l'ordre des appels qui compte (f) ; ici, leur découpage ne change rien (e), mais ce n'est pas "
            "garanti pour toutes les méthodes. Sans graine, chaque générateur part "
            "d'un état imprévisible (c). `np.random.seed` ne règle que l'état global de l'ancienne interface, "
            "que `default_rng()` n'utilise pas (d) : c'est pour le code ancien que `wb.setup` le fixe aussi. En "
            "pratique : `rng = np.random.default_rng(seed)`, passé explicitement aux fonctions ; et un générateur "
            "par usage (découpage, initialisation…) si l'on veut qu'une ligne ajoutée ici ne change pas les tirages "
            "de là-bas (`rng.spawn(2)` en fabrique plusieurs, indépendants, à partir d'un seul)."),

    Ex("2.15", "🔨", 3, 40, "Dispersion : `variance`, `std`, `percentile`, `zscore`",
       "écrire variance, écart-type, percentiles et z-scores, et les appliquer aux mesures des manchots.",
       "Ex 2.13, Ex 2.4 · fiche §2.3.2 (🧮 ddof, 🧮 percentiles)", thread="Penguins", tracks="R, M, C",
       body=MYLEARN_SHORT + r"""

Écris `variance`, `std`, `percentile` et `zscore`.
- `variance(x, ddof=0, axis=None)` : la somme des carrés des écarts **à la moyenne**, divisée par $n - \mathrm{ddof}$ ; `ValueError` si $n - \mathrm{ddof} \le 0$. Calcule d'abord les écarts, puis élève-les au carré : la formule « moyenne des carrés − carré de la moyenne » de 0B est juste sur le papier, mais elle perd des chiffres sur ordinateur quand les valeurs sont grandes et peu dispersées (un test le vérifie). Avec `axis`, garde la dimension réduite pour soustraire la moyenne (`keepdims=True`, ou `np.expand_dims`).
- `std` : la racine de la variance.
- `percentile(x, q, axis=None)` : la méthode de la fiche (🧮 « percentiles et quantiles ») : trie, place le percentile à la position $\frac{q}{100}(n - 1)$, puis interpole entre les deux valeurs voisines. `q` peut être un nombre ou une liste ; refuse un `q` hors de $[0, 100]$.
- `zscore(x, ddof=0, axis=None)` : $(x - \bar{x}) / \sigma$ ; refuse un écart-type nul (données constantes).

Vérifications (la cellule de vérification les calcule avec tes fonctions) :
a) `variance(flipper)`, la variance des nageoires (1 décimale) ;
b) `std(mass, ddof=1)`, l'écart-type des masses avec ddof = 1 (1 décimale) ;
c) `percentile(mass, [25, 50, 75])`, les trois quartiles des masses ;
d) le z-score de la masse du manchot le plus lourd (2 décimales) ;
e) le nombre de manchots dont la masse est à plus de 2 écarts-types de la moyenne ($|z| > 2$) ;
f) le plus grand $|z|$ du tableau `zscore(X_measures, axis=0)` (2 décimales) : une mesure est-elle aberrante, au sens $|z| > 3$ ?

Puis les tests. Dans tes notes : pourquoi faut-il `axis=0` en f ? Que ferait `axis=None` sur ce tableau ?""",
       check=RELOAD + r'''with wb.attempt("2.15"):
    st = mylearn.stats
    wb.check("2.15a", st.variance(flipper))
    wb.check("2.15b", st.std(mass, ddof=1))
    wb.check("2.15c", st.percentile(mass, [25, 50, 75]))
    z_mass = st.zscore(mass)
    wb.check("2.15d", z_mass[np.argmax(mass)])
    wb.check("2.15e", int(np.sum(np.abs(z_mass) > 2)))
    Z_15 = st.zscore(X_measures, axis=0)
    wb.check("2.15f", np.abs(Z_15).max())
    verdict("2.15", np.allclose(Z_15.mean(axis=0), 0) and np.allclose(Z_15.std(axis=0), 1),
            "chaque colonne standardisée a une moyenne 0 et un écart-type 1.",
            "après zscore(X, axis=0), chaque colonne doit avoir une moyenne 0 et un écart-type 1 (ddof = 0).")
    run_stats_tests("test_variance_ or test_std_ or test_percentile_ or test_zscore_")''',
       solution=r'''st = mylearn.stats
z_mass = st.zscore(mass)
Z_15 = st.zscore(X_measures, axis=0)
print("variance of the flippers:", st.variance(flipper), "mm² · std of the masses (ddof=1):", st.std(mass, ddof=1), "g")
print("quartiles of the masses:", st.percentile(mass, [25, 50, 75]))
print("heaviest penguin:", mass.max(), "g, z =", z_mass[np.argmax(mass)], "· |z| > 2:", int(np.sum(np.abs(z_mass) > 2)))
print("largest |z| of each measure:", np.abs(Z_15).max(axis=0).round(2))
print("with axis=None (nonsense: grams and millimetres mixed), mean z of each column:",
      st.zscore(X_measures).mean(axis=0).round(2))
run_stats_tests("test_variance_ or test_std_ or test_percentile_ or test_zscore_", impl="ref")''',
       record=r'''wb.record("2.15a", st.variance(flipper), decimals=1, mistakes={"ddof = 0 par défaut : divise par n, pas par n − 1": st.variance(flipper, ddof=1),
                                                                 "c'est l'écart-type : on demande la variance": st.std(flipper)})
wb.record("2.15b", st.std(mass, ddof=1), decimals=1, mistakes={"ddof = 1 : divise par n − 1, pas par n": st.std(mass),
                                                                "c'est la variance : prends sa racine carrée": st.variance(mass, ddof=1)})
wb.record("2.15c", st.percentile(mass, [25, 50, 75]), decimals=1, mistakes={"q est un percentile, entre 0 et 100 (25, pas 0.25)": st.percentile(mass, [0.25, 0.5, 0.75])})
z_mass = st.zscore(mass)
wb.record("2.15d", z_mass[np.argmax(mass)], decimals=2, mistakes={"c'est l'écart à la moyenne en grammes : divise-le par l'écart-type": mass.max() - st.mean(mass),
                                                                   "divise par l'écart-type, pas par la variance": (mass.max() - st.mean(mass)) / st.variance(mass)})
wb.record("2.15e", int(np.sum(np.abs(z_mass) > 2)))
wb.record("2.15f", np.abs(st.zscore(X_measures, axis=0)).max(), decimals=2,
          mistakes={"standardise chaque COLONNE (axis=0) : avec axis=None, les grammes écrasent les millimètres": np.abs(st.zscore(X_measures)).max()})''',
       note="Les nageoires ont une variance de 197 mm² : son unité (des mm²) la rend peu parlante, d'où l'écart-type "
            "(14 mm). Le manchot le plus lourd (6 300 g) est à 2,62 écarts-types de la moyenne : lourd, pas aberrant ; "
            "aucune des quatre mesures ne dépasse $|z| = 3$. Avec `axis=None`, `zscore` calculerait **une** moyenne "
            "et **un** écart-type pour tout le tableau, grammes et millimètres mélangés : toutes les masses auraient "
            "un z positif et toutes les autres mesures un z négatif. Standardiser colonne par colonne, c'est ce que "
            "fait le `StandardScaler` de scikit-learn (ch. 12)."),

    Ex("2.16", "🔨", 2, 20, "Un histogramme fait maison",
       "écrire un histogramme (comptages ou densité) et lire la forme d'une distribution.",
       "Ex 2.13 · fiche §2.2 (🧮 densité)", thread="Penguins", tracks="M, C",
       body=MYLEARN_SHORT + r"""

Écris `histogram(x, bins=10, bin_range=None, density=False)`. On découpe `[low, high]` (par défaut, du minimum au maximum de `x`) en `bins` intervalles de même largeur : les bords sont `np.linspace(low, high, bins + 1)`. Chaque intervalle contient son bord gauche mais pas son bord droit, **sauf le dernier**, qui garde aussi `high`. Les valeurs hors de `[low, high]` sont ignorées.
- Pour trouver l'intervalle de chaque valeur, `np.searchsorted(edges, x, side="right") - 1` donne le numéro de l'intervalle qui commence au dernier bord inférieur ou égal à la valeur ; ramène les valeurs égales à `high` dans le dernier intervalle ; puis compte avec `np.bincount(..., minlength=bins)`.
- La formule directe $\lfloor (x - \text{low}) / \text{largeur} \rfloor$ marche aussi, mais avec des bords comme 0,1 ou 0,3, qui ne tombent pas juste en binaire, un arrondi peut envoyer une valeur posée pile sur un bord dans l'intervalle voisin. Les tests vérifient les valeurs posées sur un bord.
- Avec `density=True`, divise chaque comptage par (nombre de valeurs comptées × largeur) : l'aire totale des barres vaut alors 1 (fiche, 🧮 densité).

La vérification compare ton histogramme à `np.histogram` sur les nageoires (20 intervalles, puis 7 intervalles de 170 à 240 mm, où beaucoup de nageoires tombent pile sur un bord), vérifie l'aire de la densité, trace l'histogramme des nageoires avec **tes** comptages, puis lance les tests. La cellule d'après compare plusieurs nombres d'intervalles : réponds ensuite dans la cellule 📝.""",
       check=RELOAD + r'''with wb.attempt("2.16"):
    st = mylearn.stats
    counts_16, edges_16 = st.histogram(flipper, bins=20)
    numpy_counts, numpy_edges = np.histogram(flipper, bins=20)
    verdict("2.16", np.array_equal(counts_16, numpy_counts) and np.allclose(edges_16, numpy_edges),
            "mêmes comptages et mêmes bords que np.histogram (20 intervalles).",
            "tes comptages ou tes bords diffèrent de np.histogram(flipper, bins=20) : compare-les.")
    counts_7, _ = st.histogram(flipper, bins=7, bin_range=(170, 240))
    verdict("2.16", np.array_equal(counts_7, np.histogram(flipper, bins=7, range=(170, 240))[0]),
            "bords imposés (170, 180, …, 240) : mêmes comptages que NumPy, bords compris.",
            "avec bin_range=(170, 240), tes comptages diffèrent de NumPy : attention aux valeurs posées pile sur un bord.")
    density_16, edges_d = st.histogram(flipper, bins=20, density=True)
    area_16 = float(np.sum(density_16 * np.diff(edges_d)))
    verdict("2.16", abs(area_16 - 1) < 1e-9, "l'aire des barres de densité vaut 1.",
            f"l'aire des barres de densité vaut {fr(area_16, 4)} : divise par (nombre de valeurs comptées × largeur).")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.stairs(counts_16, edges_16, fill=True, alpha=0.6)
    ax.set(xlabel="flipper_length_mm", ylabel="number of penguins", title="Your histogram of the 342 flippers (20 bins)")
    plt.show()
    run_stats_tests("test_histogram_")''',
       solution=r'''st = mylearn.stats
counts_16, edges_16 = st.histogram(flipper, bins=20)
print(counts_16, edges_16.round(1))
print(st.histogram(flipper, bins=7, bin_range=(170, 240)))
density_16, edges_d = st.histogram(flipper, bins=20, density=True)
print("area of the density bars:", np.sum(density_16 * np.diff(edges_d)))
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.stairs(counts_16, edges_16, fill=True, alpha=0.6)
ax.set(xlabel="flipper_length_mm", ylabel="number of penguins", title="Histogram of the 342 flippers (20 bins)")
plt.show()
run_stats_tests("test_histogram_", impl="ref")''',
       after=[("md", "**Combien d'intervalles ?** La cellule suivante trace l'histogramme des nageoires avec **ton** "
                     "`histogram`, pour 4, 20 et 100 intervalles."),
              ("code", r'''with wb.attempt("2.16"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, n_bins in zip(axes, [4, 20, 100]):
        counts, edges = mylearn.stats.histogram(flipper, bins=n_bins)
        ax.stairs(counts, edges, fill=True, alpha=0.6)
        ax.set(title=f"{n_bins} bins (width {edges[1] - edges[0]:.2f} mm)", xlabel="flipper_length_mm")
    plt.show()'''),
              ("todo_md", "📝 **Mes réponses** (2.16) : avec 4 intervalles, retrouves-tu les deux bosses de la fiche "
                          "(§2.2) ? Avec 100, pourquoi l'histogramme ressemble-t-il à un peigne, avec tant d'intervalles "
                          "vides ? Combien d'intervalles choisirais-tu, et pourquoi ?\n\n…"),
              ("solution_md", "**Réponses (2.16)** : avec 4 intervalles de près de 15 mm, les deux bosses se fondent "
                              "en une seule : le creux entre elles (autour de 205 mm) tombe au milieu d'un intervalle. "
                              "Avec 100 intervalles, chacun ne fait que 0,59 mm, alors que les nageoires sont mesurées "
                              "au millimètre près (des nombres entiers) : beaucoup d'intervalles ne contiennent aucun "
                              "entier, d'où les trous du peigne. Un bon choix se situe entre les deux : de 15 à 25 "
                              "intervalles (2 à 4 mm de large), ou mieux, une largeur multiple de 1 mm, la précision "
                              "des mesures. Le nombre d'intervalles change ce qu'on voit : on en essaie plusieurs "
                              "avant de conclure sur la forme d'une distribution.")],
       note="`np.searchsorted(edges, x, side=\"right\") - 1` est exact sur les bords : il compare aux bords "
            "eux-mêmes au lieu de diviser. C'est aussi un premier exemple d'**interface** : ton `histogram` rend "
            "exactement ce que rend `np.histogram` (comptages, bords), si bien que `ax.stairs(counts, edges)` "
            "trace les deux de la même façon."),

    Ex("2.17", "🎨", 2, 20, "Galerie des lois usuelles",
       "tracer les densités et des tirages des lois uniforme, normale, de Bernoulli et catégorielle.",
       "Ex 2.16 · fiche §2.3.1 à §2.3.4 · livre §2.3 (figures 2.6 à 2.12)", thread="synthétique", tracks="C",
       body=r"""Le livre montre chaque loi par sa courbe, puis par des tirages : sur sa figure 2.8, chaque tirage est un point posé sous la courbe, décalé verticalement au hasard pour qu'on distingue les points. Reproduis cette idée en une figure de quatre panneaux (2 × 2) :
1. **uniforme** : les densités des lois uniformes sur $[0, 1]$ et sur $[-1, 1]$ ;
2. **normale** : les densités de quatre lois inspirées de la figure 2.7 du livre (qui ne chiffre pas leurs écarts-types), données dans `NORMALS_17` sous la forme $(\mu, \sigma)$ : $\mathcal{N}(0, 1)$, $\mathcal{N}(1, 1)$, $\mathcal{N}(-1, 0{,}5^2)$ et $\mathcal{N}(-1, 2^2)$, avec **30 tirages** de chacune, dessinés comme des points sous les courbes (une couleur par loi) ;
3. **Bernoulli** : pour une pièce équilibrée ($p = 0{,}5$) et la pièce truquée du livre ($p = 0{,}3$), des bâtons pour les probabilités de 0 et de 1, et à côté les fréquences observées sur 1 000 lancers (figure 2.12) ;
4. **catégorielle** : les probabilités des cinq types de voitures de 2.2 (`P_CARS`, `CAR_TYPES`) et les fréquences observées sur 1 000 tirages (ici, `rng.choice(5, size=1000, p=P_CARS)` ; tu écriras ton propre tirage en 2.19).

Écris d'abord deux fonctions de densité, qui acceptent un array `x` :
- `uniform_pdf(x, a, b)` : $\frac{1}{b - a}$ sur $[a, b]$, 0 ailleurs (`np.where`) ;
- `normal_pdf(x, mu, sigma)` : la formule de la fiche §2.3.2 (`np.exp`, `np.sqrt`, `np.pi`).

Puis complète `draw_gallery(rng)`, qui trace la figure avec ces deux fonctions et **un seul** générateur `rng` (la vérification lui passe `np.random.default_rng(0)`). La vérification compare tes densités à celles de SciPy (`scipy.stats.uniform`, `scipy.stats.norm`) ; la figure, elle, se juge à l'œil : compare-la avec celle du corrigé. Titres, légendes et axes font partie du travail.""",
       given=r'''NORMALS_17 = [(0, 1), (1, 1), (-1, 0.5), (-1, 2)]      # (mu, sigma): four normal laws in the spirit of the book's figure 2.7''',
       todo=r'''def uniform_pdf(x, a, b):
    """Density of the uniform law on [a, b] at x (x: a number or an array)."""
    raise NotImplementedError("uniform_pdf() is not written yet")


def normal_pdf(x, mu, sigma):
    """Density of the normal law N(mu, sigma**2) at x (x: a number or an array)."""
    raise NotImplementedError("normal_pdf() is not written yet")


def draw_gallery(rng):
    """The 2 x 2 figure: uniform, normal (+ 30 draws of each), Bernoulli, categorical."""
    # fig, axes = plt.subplots(2, 2, figsize=(12, 8))    # axes[0, 1] is the top right panel
    # grid = np.linspace(-5, 5, 1001)
    # TODO 1. axes[0, 0]: uniform_pdf on [0, 1] and on [-1, 1]
    # TODO 2. axes[0, 1]: normal_pdf of the 4 laws of NORMALS_17, and 30 draws of each (rng.normal) as dots
    # TODO 3. axes[1, 0]: Bernoulli p = 0.5 and p = 0.3: bars of the probabilities, bars of the frequencies of 1 000 throws
    # TODO 4. axes[1, 1]: P_CARS: bars of the probabilities, bars of the frequencies of 1 000 draws
    raise NotImplementedError("draw_gallery() is not written yet")''',
       check=r'''with wb.attempt("2.17"):
    grid_17 = np.linspace(-2.9, 2.9, 59) + 0.013            # avoids the bounds a and b (a convention, see the book)
    verdict("2.17", np.allclose(uniform_pdf(grid_17, -1, 1), scipy_stats.uniform(loc=-1, scale=2).pdf(grid_17))
            and np.allclose(uniform_pdf(grid_17, 0, 1), scipy_stats.uniform().pdf(grid_17)),
            "uniform_pdf coïncide avec scipy.stats.uniform.",
            "uniform_pdf diffère de scipy.stats.uniform : la hauteur vaut 1 / (b − a) sur [a, b], et 0 ailleurs.")
    verdict("2.17", all(np.allclose(normal_pdf(grid_17, mu, sigma), scipy_stats.norm(mu, sigma).pdf(grid_17))
                        for mu, sigma in NORMALS_17),
            "normal_pdf coïncide avec scipy.stats.norm pour les quatre lois.",
            "normal_pdf diffère de scipy.stats.norm(mu, sigma).pdf : relis la formule (le carré, 2σ², la racine de 2π).")
    draw_gallery(np.random.default_rng(0))
    plt.show()''',
       solution=r'''def uniform_pdf(x, a, b):
    """Density of the uniform law on [a, b] at x (x: a number or an array)."""
    x = np.asarray(x, dtype=float)
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0.0)


def normal_pdf(x, mu, sigma):
    """Density of the normal law N(mu, sigma**2) at x (x: a number or an array)."""
    x = np.asarray(x, dtype=float)
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def draw_gallery(rng):
    """The 2 x 2 figure: uniform, normal (+ 30 draws of each), Bernoulli, categorical."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    grid = np.linspace(-5, 5, 1001)
    ax = axes[0, 0]
    for a, b in [(0, 1), (-1, 1)]:
        ax.plot(grid, uniform_pdf(grid, a, b), label=f"uniform on [{a}, {b}]")
    ax.set(xlim=(-2, 2), xlabel="x", ylabel="density", title="Uniform: a flat density, total area 1")
    ax.legend()
    ax = axes[0, 1]
    for k, (mu, sigma) in enumerate(NORMALS_17):
        line, = ax.plot(grid, normal_pdf(grid, mu, sigma), label=f"N({mu}, {sigma}²)")
        draws = rng.normal(mu, sigma, size=30)
        heights = -0.1 - 0.12 * k + rng.uniform(-0.04, 0.04, size=30)   # one row of dots per law, jittered
        ax.scatter(draws, heights, s=10, color=line.get_color())
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="x", ylabel="density (dots: 30 draws)", title="Normal: four densities, 30 draws of each")
    ax.legend()
    ax = axes[1, 0]
    for k, p in enumerate([0.5, 0.3]):
        heads = rng.random(1000) < p                                     # 1 000 throws: True with probability p
        positions = np.array([0, 1]) + 3 * k
        ax.bar(positions - 0.2, [1 - p, p], width=0.4, color="C0", label="probability" if k == 0 else None)
        ax.bar(positions + 0.2, [1 - heads.mean(), heads.mean()], width=0.4, color="C1",
               label="frequency, 1 000 throws" if k == 0 else None)
    ax.set_xticks([0, 1, 3, 4], ["0", "1\np = 0.5", "0", "1\np = 0.3"])
    ax.set(ylabel="probability", title="Bernoulli: a fair coin, a biased coin")
    ax.legend()
    ax = axes[1, 1]
    cars = rng.choice(len(P_CARS), size=1000, p=P_CARS)
    positions = np.arange(len(P_CARS))
    ax.bar(positions - 0.2, P_CARS, width=0.4, label="probability")
    ax.bar(positions + 0.2, np.bincount(cars, minlength=len(P_CARS)) / 1000, width=0.4,
           label="frequency, 1 000 draws")
    ax.set_xticks(positions, CAR_TYPES)
    ax.set(ylabel="probability", title="Categorical: the scrapyard of 2.2")
    ax.legend()
    fig.tight_layout()


draw_gallery(np.random.default_rng(0))
plt.show()''',
       note="Les tirages se massent là où la densité est haute (panneau 2) : c'est ce que « densité » veut dire. "
            "$\\mathcal{N}(-1, 0{,}5^2)$ monte jusqu'à 0,8 ; une normale d'écart-type inférieur à 0,4 environ "
            "dépasserait 1, et c'est permis, puisque seule l'aire compte. Les fréquences sur 1 000 tirages s'écartent un peu des "
            "probabilités : de l'ordre de $\\sqrt{p(1 - p)/1\\,000}$, soit 1 à 2 points (0B). Aux bornes $a$ et $b$ "
            "d'une loi uniforme, la valeur de la densité est une convention (le livre le signale) : la "
            "vérification évite ces deux points."),

    Ex("2.18", "🔬", 2, 20, "68-95-99,7 : la théorie face aux tirages et aux manchots",
       "mesurer la part des valeurs à moins de 1, 2 et 3 écarts-types de la moyenne, sur des tirages normaux "
       "puis sur de vraies mesures.",
       "Ex 2.15, Ex 2.3 · fiche §2.3.2 · livre §2.3.2 (figures 2.10 et 2.11)", thread="Penguins", tracks="M, C",
       body=r"""Écris `share_within(x, k)` : la proportion des valeurs de `x` à **strictement moins** de `k` écarts-types de leur moyenne, $|x_i - \bar{x}| < k\,\sigma$, avec l'écart-type de ddof = 0 (tu peux utiliser NumPy ici). Puis :
a) `shares_normal` : la liste des trois proportions, pour $k = 1, 2, 3$, sur 100 000 tirages `np.random.default_rng(0).standard_normal(100_000)` (3 décimales) ;
b) `shares_adelie` : les mêmes proportions pour les nageoires des 151 Adélie (`adelie_flipper`) : le modèle normal de 2.3 tient-il ?
c) `n_above_203` : combien d'Adélie ont une nageoire de **strictement plus** de 203 mm ? Compare avec ta réponse à 2.3 d ;
d) `shares_all` : les trois proportions pour les nageoires de **tous** les manchots (`flipper`).

La cellule suivante trace les deux histogrammes (en densité) avec la courbe normale de même moyenne et de même écart-type. Réponds ensuite dans la cellule 📝 : pour quelles données la règle marche-t-elle ? Pourquoi échoue-t-elle sur les nageoires de tous les manchots, et dans quel sens ?""",
       given=r'''adelie_flipper = measured.loc[measured["species"] == "Adelie", "flipper_length_mm"].to_numpy()   # 151 Adelie


def plot_normal_fit(ax, x, title):
    """Density histogram of x, and the normal density with the same mean and standard deviation (ddof=0)."""
    ax.hist(x, bins=20, density=True, alpha=0.6, label="data")
    grid = np.linspace(x.min() - 10, x.max() + 10, 300)
    ax.plot(grid, scipy_stats.norm(x.mean(), x.std()).pdf(grid), label="normal, same mean and std")
    ax.set(title=title, xlabel="flipper_length_mm", ylabel="density")
    ax.legend()''',
       todo=r'''def share_within(x, k):
    """Proportion of the values of x strictly less than k standard deviations (ddof=0) away from their mean."""
    raise NotImplementedError("share_within() is not written yet")


shares_normal = ...   # a) [k = 1, k = 2, k = 3] for 100 000 standard normal draws (seed 0)
shares_adelie = ...   # b) the same for adelie_flipper
n_above_203 = ...     # c)
shares_all = ...      # d) the same for flipper (all the penguins)''',
       check=r'''wb.check("2.18a", shares_normal)
wb.check("2.18b", shares_adelie)
wb.check("2.18c", n_above_203)
wb.check("2.18d", shares_all)''',
       solution=r'''def share_within(x, k):
    """Proportion of the values of x strictly less than k standard deviations (ddof=0) away from their mean."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.abs(x - x.mean()) < k * x.std()))


z_18 = np.random.default_rng(0).standard_normal(100_000)
shares_normal = [share_within(z_18, k) for k in (1, 2, 3)]
shares_adelie = [share_within(adelie_flipper, k) for k in (1, 2, 3)]
n_above_203 = int(np.sum(adelie_flipper > 203))
shares_all = [share_within(flipper, k) for k in (1, 2, 3)]
print(np.round(shares_normal, 4), np.round(shares_adelie, 4), n_above_203, np.round(shares_all, 4))
print("largest Adelie flippers:", np.sort(adelie_flipper)[-6:])''',
       record=r'''wb.record("2.18a", shares_normal, decimals=3)
adelie_ddof1_18 = [float(np.mean(np.abs(adelie_flipper - adelie_flipper.mean()) < k * adelie_flipper.std(ddof=1))) for k in (1, 2, 3)]
wb.record("2.18b", shares_adelie, decimals=3, mistakes={"l'énoncé impose l'écart-type avec ddof = 0": adelie_ddof1_18})
wb.record("2.18c", n_above_203, mistakes={"« strictement plus de 203 mm » : compte les vraies nageoires avec >, pas avec >= (et pas la prévision de 2.3 d)": int(np.sum(adelie_flipper >= 203))})
all_ddof1_18 = [float(np.mean(np.abs(flipper - flipper.mean()) < k * flipper.std(ddof=1))) for k in (1, 2, 3)]
wb.record("2.18d", shares_all, decimals=3, mistakes={"l'énoncé impose l'écart-type avec ddof = 0": all_ddof1_18})''',
       after=[("code", r'''fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
plot_normal_fit(axes[0], adelie_flipper, "Adelie flippers (151)")
plot_normal_fit(axes[1], flipper, "All the flippers (342)")
plt.show()'''),
              ("todo_md", "📝 **Mes réponses** (2.18) : pour quelles données la règle marche-t-elle ? Pourquoi "
                          "échoue-t-elle sur tous les manchots, et dans quel sens ?\n\n…"),
              ("solution_md", "**Réponses (2.18)** : les tirages normaux redonnent la règle, à l'aléa près "
                              "(68,4 %, 95,5 %, 99,7 %). Les nageoires des Adélie aussi, à peu près (70,9 %, "
                              "95,4 %, 99,3 %) : leur histogramme ressemble à une cloche. Le modèle de 2.3 prévoyait "
                              "environ 4 nageoires au-dessus de 203 mm ; il y en a 3, plus une à 203 mm pile. Sur "
                              "tous les manchots, la règle se trompe : seulement 62,9 % à moins d'un écart-type, et "
                              "100 % à moins de trois. L'histogramme a deux bosses : la moyenne tombe dans le creux "
                              "entre elles, ce qui vide la bande centrale, et l'écart-type, gonflé par la distance "
                              "entre les deux groupes, rend la bande de ± 3 écarts-types si large qu'elle contient "
                              "tout. La règle vaut pour une distribution en cloche : on regarde l'histogramme avant "
                              "de s'en servir.")],
       note="Avec ddof = 1, l'écart-type des Adélie grandit un peu (6,54 mm au lieu de 6,52 mm) et une nageoire "
            "de plus entre dans la bande de ± 2 écarts-types : 96,0 % au lieu de 95,4 %. Sur 151 valeurs, une seule "
            "valeur pèse 0,7 point : ne sur-interprète pas la troisième décimale."),
])

# ---------------------------------------------------------------------------
# Part B: drawing at random (2.19 to 2.21)
# ---------------------------------------------------------------------------
PART_B = Part("B", "Tirer au hasard : roue, dépendance, avec ou sans remise",
              "Fiche §2.3.4, §2.4 et §2.5. Tu programmes la roue de loterie de la fiche (un tirage dans une loi "
              "catégorielle), tu simules une variable qui dépend d'une autre, puis les tirages avec et sans remise, "
              "qui serviront au bootstrap.", exercises=[
    Ex("2.19", "🔨", 2, 25, "La roue de la fortune : tirer dans une distribution discrète",
       "tirer dans une loi catégorielle en inversant les sommes cumulées, et comparer les fréquences obtenues "
       "aux probabilités.",
       "Ex 2.2, Ex 2.16 · fiche §2.3.4 · livre §2.2 (figures 2.2 à 2.4)", thread="synthétique", tracks="M, C",
       body=MYLEARN_SHORT + r"""

Écris `sample_categorical(p, size=None, rng=None)` en suivant **exactement** l'algorithme de sa docstring : c'est la roue de la fiche (§2.3.4) et le calcul de 2.2 f et g.
1. Vérifie `p` : une dimension, aucune valeur négative, une somme égale à 1 à $10^{-8}$ près ; sinon, `ValueError`.
2. Si `rng` vaut `None`, crée `np.random.default_rng()` (sans graine : pas reproductible).
3. Tire `u = rng.random(size)`, puis cherche le segment de chaque `u` : `np.searchsorted(np.cumsum(p), u, side="right")`.
4. Ramène le résultat à `len(p) - 1` au plus (`np.minimum`) : à cause des arrondis, la dernière somme cumulée peut valoir `0.9999999999999999`, et un `u` plus grand donnerait une catégorie qui n'existe pas.
5. Avec `size=None`, renvoie un `int` Python (`int(...)`), sinon un array d'entiers.

La vérification tire 10 000 voitures dans la casse de 2.2 (`P_CARS`) avec la graine 0, compare les fréquences observées aux probabilités, vérifie qu'un tirage unique renvoie un `int`, trace les deux en bâtons, puis lance les tests : ils vérifient aussi qu'à graine égale, tes tirages sont **exactement** ceux de l'algorithme documenté.""",
       check=RELOAD + r'''with wb.attempt("2.19"):
    st = mylearn.stats
    cars_19 = np.asarray(st.sample_categorical(P_CARS, size=10_000, rng=np.random.default_rng(0)))
    freq_19 = np.bincount(cars_19, minlength=len(P_CARS)) / len(cars_19)
    gap_19 = float(np.abs(freq_19 - P_CARS).max())
    verdict("2.19", cars_19.shape == (10_000,) and gap_19 < 0.02,
            f"10 000 tirages, des fréquences proches des probabilités (écart maximal {fr(gap_19, 3)}).",
            f"écart maximal de {fr(gap_19, 3)} entre fréquences et probabilités (ou pas 10 000 tirages) : relis l'algorithme.")
    one_19 = st.sample_categorical(P_CARS, rng=np.random.default_rng(1))
    verdict("2.19", isinstance(one_19, int), "un tirage unique (size=None) renvoie un int.",
            f"avec size=None, renvoie un int Python (reçu : {type(one_19).__name__}).")
    positions_19 = np.arange(len(P_CARS))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(positions_19 - 0.2, P_CARS, width=0.4, label="probability (2.2)")
    ax.bar(positions_19 + 0.2, freq_19, width=0.4, label="frequency, 10 000 draws")
    ax.set_xticks(positions_19, CAR_TYPES)
    ax.legend()
    plt.show()
    run_stats_tests("test_sample_categorical_")''',
       solution=r'''st = mylearn.stats
cars_19 = st.sample_categorical(P_CARS, size=10_000, rng=np.random.default_rng(0))
freq_19 = np.bincount(cars_19, minlength=len(P_CARS)) / len(cars_19)
print("probabilities:", P_CARS.round(3), "· frequencies:", freq_19.round(3), "· first draws:", cars_19[:10])
print("one draw:", st.sample_categorical(P_CARS, rng=np.random.default_rng(1)))
positions_19 = np.arange(len(P_CARS))
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(positions_19 - 0.2, P_CARS, width=0.4, label="probability (2.2)")
ax.bar(positions_19 + 0.2, freq_19, width=0.4, label="frequency, 10 000 draws")
ax.set_xticks(positions_19, CAR_TYPES)
ax.legend()
plt.show()
run_stats_tests("test_sample_categorical_", impl="ref")''',
       note="Sur 10 000 tirages, les fréquences s'écartent des probabilités de quelques dixièmes de point (0,2 "
            "point au plus ici) : l'écart typique vaut $\\sqrt{p(1 - p)/n}$, environ 0,45 point pour $p = 0{,}28$. "
            "`np.searchsorted` trouve le segment de chaque `u` par dichotomie, en $\\log_2 K$ comparaisons pour $K$ "
            "catégories. Un modèle de langage fait lui aussi un tirage dans une loi catégorielle, sur des dizaines de "
            "milliers de tokens (ch. 22 et B3), même si PyTorch utilise une autre astuce de tirage. "
            "`rng.choice(K, p=p)` fait la même chose en NumPy."),

    Ex("2.20", "🔮", 2, 15, "Le pelage des animaux : une variable qui dépend d'une autre",
       "prévoir puis simuler une variable dont la loi dépend d'une autre, et voir ce que change l'indépendance.",
       "Ex 2.19 · fiche §2.3.5, §2.4, §2.4.1 · livre §2.4", thread="synthétique", tracks="C", hypothesis=True,
       body=r"""Le livre (§2.4) tire d'abord un animal, puis la longueur de son pelage dans la loi propre à cet animal. Voici le modèle :

| Animal | Probabilité | Pelage (cm) |
|---|---|---|
| chien | 0,5 | loi normale de moyenne 7 et d'écart-type 0,8 |
| chat | 0,3 | loi normale de moyenne 3 et d'écart-type 0,4 |
| hamster | 0,2 | loi normale de moyenne 1 et d'écart-type 0,2 |

**Sans rien exécuter**, prévois :
a) `pred_mean_fur` : la longueur moyenne du pelage sur un très grand nombre d'animaux tirés (1 décimale) ; pense à l'espérance (fiche §2.3.5) ;
b) `pred_bumps` : le nombre de bosses de l'histogramme de tous les pelages (un entier) ;
c) `pred_above_7` : la probabilité qu'un animal tiré au hasard ait un pelage de plus de 7 cm (2 décimales) ;
d) `pred_above_7_dog` : la même probabilité, **sachant** que l'animal tiré est un chien (2 décimales).

Comparer c et d, c'est voir la dépendance : connaître l'animal change la loi du pelage. Écris ton hypothèse, puis tes quatre prédictions ; ensuite seulement, exécute l'**Expérience** : elle tire 20 000 animaux avec ton `sample_categorical` (2.19), puis leurs pelages.""",
       given=r'''ANIMALS = ["dog", "cat", "hamster"]
P_ANIMALS = np.array([0.5, 0.3, 0.2])        # step 1: which animal?
FUR_MEAN = np.array([7.0, 3.0, 1.0])         # step 2: its fur length (cm), drawn from the normal law of its animal
FUR_STD = np.array([0.8, 0.4, 0.2])
animal_20 = fur_20 = None                    # filled by the experiment below''',
       todo=r'''pred_mean_fur = ...      # a)
pred_bumps = ...         # b) an integer
pred_above_7 = ...       # c)
pred_above_7_dog = ...   # d)''',
       check=r'''wb.check("2.20a", pred_mean_fur)
wb.check("2.20b", pred_bumps)
wb.check("2.20c", pred_above_7)
wb.check("2.20d", pred_above_7_dog)''',
       solution=r'''pred_mean_fur = 0.5 * 7 + 0.3 * 3 + 0.2 * 1     # the expected value: 4.6 cm
pred_bumps = 3
pred_above_7 = 0.5 * 0.5                         # a dog (0.5), whose fur is above its mean (0.5)
pred_above_7_dog = 0.5''',
       record=r'''wb.record("2.20a", pred_mean_fur, decimals=1, mistakes={"pondère chaque moyenne par la probabilité de l'animal (ce n'est pas la moyenne simple de 7, 3 et 1)": 11 / 3})
wb.record("2.20b", pred_bumps, mistakes={"compare les intervalles moyenne ± 3 écarts-types des trois animaux : se chevauchent-ils ?": 1})
wb.record("2.20c", pred_above_7, decimals=2, mistakes={"tu as raisonné comme si l'animal était forcément un chien : ici, il peut être n'importe lequel ; combine la probabilité de tirer chaque animal et celle que son pelage dépasse 7 cm": 0.5})
wb.record("2.20d", pred_above_7_dog, decimals=2, mistakes={"sachant que c'est un chien, ne multiplie plus par la probabilité de tirer un chien : seule compte la loi de son pelage": 0.25})''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", guarded(r'''with wb.attempt("2.20"):
    rng_20 = np.random.default_rng(0)
    animal_20 = mylearn.stats.sample_categorical(P_ANIMALS, size=20_000, rng=rng_20)   # step 1: the animals
    fur_20 = rng_20.normal(FUR_MEAN[animal_20], FUR_STD[animal_20])                   # step 2: each fur, given its animal
    print(f"mean fur: {fur_20.mean():.2f} cm · P(fur > 7 cm) ≈ {np.mean(fur_20 > 7):.3f} · "
          f"P(fur > 7 cm | dog) ≈ {np.mean(fur_20[animal_20 == 0] > 7):.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].hist(fur_20, bins=80, color="0.5")
    axes[0].set_title("All the animals")
    for k, name in enumerate(ANIMALS):
        axes[1].hist(fur_20[animal_20 == k], bins=40, alpha=0.6, label=name)
    axes[1].set_title("By animal")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("fur length (cm)")
    plt.show()''', ['pred_mean_fur', 'pred_bumps', 'pred_above_7', 'pred_above_7_dog'],
                               "⏳ Ex 2.20 : écris d'abord tes quatre prédictions, puis relance cette cellule.")),
              ("md", "e) On mélange maintenant les pelages au hasard, sans toucher aux animaux (`rng.permutation(fur_20)`) : "
                     "chaque animal reçoit le pelage d'un animal quelconque, et les deux colonnes deviennent "
                     "**indépendantes**. Prévois, **avant** d'exécuter la cellule d'après, `pred_hamster_shuffled` : la "
                     "longueur moyenne du pelage des « hamsters » après ce mélange (1 décimale)."),
              ("todo", "pred_hamster_shuffled = ...   # e)"),
              ("check", 'wb.check("2.20e", pred_hamster_shuffled)'),
              ("solution", "pred_hamster_shuffled = 4.6   # after shuffling, a hamster gets the fur of any animal: the overall mean"),
              ("record", 'wb.record("2.20e", pred_hamster_shuffled, decimals=1, mistakes={"le mélange sépare chaque pelage de son '
                         'animal : un « hamster » reçoit maintenant le pelage d\'un animal quelconque": 1.0})'),
              ("code", guarded(r'''if fur_20 is None:
    print("⏳ Ex 2.20 : exécute d'abord l'expérience de a) à d) (elle utilise ton sample_categorical de 2.19).")
else:
    shuffled_fur_20 = np.random.default_rng(1).permutation(fur_20)     # the furs, in a random order
    print(f"mean fur of the hamsters: {fur_20[animal_20 == 2].mean():.2f} cm before shuffling, "
          f"{shuffled_fur_20[animal_20 == 2].mean():.2f} cm after")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for k, name in enumerate(ANIMALS):
        ax.hist(shuffled_fur_20[animal_20 == k], bins=40, alpha=0.5, density=True, label=name)
    ax.set(title="After shuffling: the same law for every animal", xlabel="fur length (cm)", ylabel="density")
    ax.legend()
    plt.show()''', ['pred_hamster_shuffled'],
                               "⏳ Ex 2.20 : écris d'abord ta prédiction e), puis relance cette cellule."))],
       note="$\\mathbb{E}[\\text{pelage}] = 0{,}5 \\times 7 + 0{,}3 \\times 3 + 0{,}2 \\times 1 = 4{,}6$ cm : "
            "l'espérance pondère la moyenne de chaque animal par sa probabilité. Les trois bosses ne se "
            "chevauchent pas (hamster : de 0,4 à 1,6 cm ; chat : de 1,8 à 4,2 cm ; chien : de 4,6 à 9,4 cm, à ± 3 "
            "écarts-types). $P(\\text{pelage} > 7) = 0{,}25$ mais $P(\\text{pelage} > 7 \\mid \\text{chien}) = 0{,}5$ : "
            "connaître l'animal change la loi du pelage, les deux variables sont **dépendantes**. Les couples "
            "(animal, pelage) successifs sont pourtant **i.i.d.** : chaque tirage recommence à zéro, avec la même "
            "loi (fiche §2.4.1). Après le mélange, chaque « hamster » porte le pelage d'un animal quelconque : sa "
            "moyenne devient 4,6 cm, et les trois histogrammes se superposent. C'est exactement ce que veut dire "
            "l'indépendance : connaître l'animal n'apprend plus rien sur le pelage. Les probabilités "
            "« sachant que » sont le sujet des ch. 3 et 4."),

    Ex("2.21", "🔨", 2, 20, "Tirer avec ou sans remise",
       "écrire les tirages avec et sans remise, et les reconnaître dans les usages du ML (mini-batches, bootstrap).",
       "Quiz 2.Q9 · 0A (mini-batches) · fiche §2.5 · livre §2.5 (figures 2.13 et 2.14)", thread="synthétique",
       tracks="R, M, C",
       body=MYLEARN_SHORT + r"""

Écris `sample(population, size, replace=True, rng=None)` en suivant l'algorithme de sa docstring : convertis la population en array (`np.asarray`), puis choisis des **indices** de lignes :
- **avec remise** : `rng.integers(0, n, size)` (chaque tirage choisit parmi les $n$ éléments) ;
- **sans remise** : les `size` premiers éléments d'une permutation, `rng.permutation(n)[:size]` ;

et renvoie `population[indices]`. Refuse une taille négative, une population vide, ou plus de tirages que d'éléments sans remise (`ValueError`). Si `rng` vaut `None`, crée `np.random.default_rng()`.

Vérifications, avec les huit lettres `LETTERS` (les huit objets des figures 2.13 et 2.14 du livre) ; la cellule de vérification les calcule avec ta fonction :
a) `"".join(sample(LETTERS, 8, rng=np.random.default_rng(1)))` : huit tirages **avec** remise ;
b) le nombre de lettres **distinctes** dans ce tirage ;
c) `"".join(sample(LETTERS, 8, replace=False, rng=np.random.default_rng(1)))` : la même chose **sans** remise ;
d) le nom de l'exception levée par `sample(LETTERS, 10, replace=False)`.

Puis deux usages du ML, à écrire avec **ta** fonction `sample` :
- `epoch_minibatches(n_examples, batch_size, rng)` : les mini-batches d'une epoch (0A), sous forme de **liste** d'arrays : un ordre au hasard des indices `0, …, n_examples − 1`, tiré **sans** remise, puis découpé en morceaux consécutifs de `batch_size` (le dernier peut être plus court). e) La vérification l'appelle pour 10 exemples, des mini-batches de 4 et `rng = np.random.default_rng(0)`, et vérifie le **deuxième** mini-batch.
- `mean_share_distinct(n_resamples, n, rng)` : la part moyenne d'éléments **distincts** dans `n_resamples` rééchantillons de `n` tirages **avec** remise parmi `np.arange(n)`, tous tirés avec le même générateur `rng`. f) La vérification l'appelle pour 200 rééchantillons de 1 000, avec `np.random.default_rng(0)` (3 décimales) : compare avec 2.6 h.""",
       given=r'''LETTERS = list("ABCDEFGH")   # the pool of eight objects of the book's figures 2.13 and 2.14''',
       todo=r'''def epoch_minibatches(n_examples, batch_size, rng):
    """The mini-batches of one epoch: a random order of the indices 0, ..., n_examples - 1, drawn WITHOUT
    replacement with YOUR mylearn.stats.sample, cut into consecutive pieces of batch_size (a list of arrays)."""
    raise NotImplementedError("epoch_minibatches() is not written yet")


def mean_share_distinct(n_resamples, n, rng):
    """Mean share of distinct elements in n_resamples resamples of n draws WITH replacement among np.arange(n)
    (YOUR mylearn.stats.sample, the same generator rng for all the resamples)."""
    raise NotImplementedError("mean_share_distinct() is not written yet")''',
       check=RELOAD + r'''with wb.attempt("2.21"):
    st = mylearn.stats
    drawn_21 = st.sample(LETTERS, 8, rng=np.random.default_rng(1))
    wb.check("2.21a", "".join(drawn_21))
    wb.check("2.21b", len(set(drawn_21)))
    wb.check("2.21c", "".join(st.sample(LETTERS, 8, replace=False, rng=np.random.default_rng(1))))
    wb.check("2.21d", error_name(st.sample, LETTERS, 10, replace=False))
    run_stats_tests("test_sample_ and not categorical")
with wb.attempt("2.21"):
    batches_21 = [np.asarray(batch) for batch in epoch_minibatches(10, 4, np.random.default_rng(0))]
    verdict("2.21", sorted(np.concatenate(batches_21).tolist()) == list(range(10))
            and [len(batch) for batch in batches_21] == [4, 4, 2],
            "une epoch : chaque exemple une fois, en mini-batches de 4, 4 et 2.",
            "une epoch doit contenir chaque exemple exactement une fois, en mini-batches de 4, 4 et 2.")
    wb.check("2.21e", batches_21[1])
    wb.check("2.21f", mean_share_distinct(200, 1000, np.random.default_rng(0)))''',
       solution=r'''def epoch_minibatches(n_examples, batch_size, rng):
    """The mini-batches of one epoch: a random order of the indices 0, ..., n_examples - 1, drawn WITHOUT
    replacement with YOUR mylearn.stats.sample, cut into consecutive pieces of batch_size (a list of arrays)."""
    order = mylearn.stats.sample(np.arange(n_examples), n_examples, replace=False, rng=rng)
    return [order[start:start + batch_size] for start in range(0, n_examples, batch_size)]


def mean_share_distinct(n_resamples, n, rng):
    """Mean share of distinct elements in n_resamples resamples of n draws WITH replacement among np.arange(n)
    (YOUR mylearn.stats.sample, the same generator rng for all the resamples)."""
    population = np.arange(n)
    shares = [len(np.unique(mylearn.stats.sample(population, n, rng=rng))) / n for _ in range(n_resamples)]
    return float(np.mean(shares))


st = mylearn.stats
drawn_21 = st.sample(LETTERS, 8, rng=np.random.default_rng(1))
print("with replacement:", "".join(drawn_21), "·", len(set(drawn_21)), "distinct letters")
print("without replacement:", "".join(st.sample(LETTERS, 8, replace=False, rng=np.random.default_rng(1))))
print("10 draws without replacement among 8:", error_name(st.sample, LETTERS, 10, replace=False))
batches_21 = epoch_minibatches(10, 4, np.random.default_rng(0))
share_21 = mean_share_distinct(200, 1000, np.random.default_rng(0))
print("mini-batches:", [batch.tolist() for batch in batches_21], "· mean share of distinct elements:", share_21)
run_stats_tests("test_sample_ and not categorical", impl="ref")''',
       record=r'''wb.record("2.21a", "".join(drawn_21))
wb.record("2.21b", len(set(drawn_21)), mistakes={"compte les lettres DIFFÉRENTES, pas les tirages": 8})
wb.record("2.21c", "".join(st.sample(LETTERS, 8, replace=False, rng=np.random.default_rng(1))),
          mistakes={"sans remise, l'algorithme documenté prend le début de rng.permutation(n) ; rng.choice(…, replace=False) ne tire pas de la même façon": "".join(np.random.default_rng(1).choice(LETTERS, 8, replace=False))})
wb.record("2.21d", error_name(st.sample, LETTERS, 10, replace=False),
          mistakes={"sans remise, on ne peut pas tirer plus d'éléments qu'il n'y en a : ta fonction doit le refuser": "no error"})
wb.record("2.21e", batches_21[1])
wb.record("2.21f", share_21, decimals=3, mistakes={"c'est la part des éléments ABSENTS : on demande celle des éléments présents au moins une fois": 1 - share_21,
                                                   "c'est la valeur théorique de 2.6 h : on demande la moyenne de TES 200 rééchantillons": 1 - (1 - 1 / 1000) ** 1000})''',
       note="Avec remise, huit tirages parmi huit lettres n'en donnent que 6 différentes : G et H sortent deux fois, "
            "C et F sont absentes. Sans remise, on obtient une **permutation** : chaque lettre une fois. Une epoch est "
            "un tirage sans remise (chaque exemple est vu une fois), découpé en mini-batches ; un rééchantillon "
            "bootstrap est un tirage avec remise, qui contient en moyenne $1 - (1 - 1/n)^n \\approx 63{,}2$ % "
            "d'éléments distincts (2.6 h) ; ta simulation de f donne 0,633, à l'aléa près. Remarque : `rng.choice(LETTERS, 8)` redonne exactement a (NumPy tire "
            "aussi des indices avec `integers`), mais pas c : sans remise, `rng.choice` n'utilise pas "
            "`permutation(n)[:size]`. Deux méthodes correctes peuvent donc donner des tirages différents à graine "
            "égale : c'est pourquoi la docstring impose l'algorithme."),
])

# ---------------------------------------------------------------------------
# Part C: the bootstrap (2.22 to 2.24)
# ---------------------------------------------------------------------------
PART_C_GIVEN = r'''chinstrap_mass = measured.loc[measured["species"] == "Chinstrap", "body_mass_g"].to_numpy()   # 68 penguins


def plot_bootstrap(boot, estimate, ci, title, xlabel):
    """Histogram of bootstrap values, with the estimate (dashed line) and the interval (shaded)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(boot, bins=30, color="0.6")
    ax.axvspan(ci[0], ci[1], alpha=0.2, label=f"interval [{ci[0]:.1f}, {ci[1]:.1f}]")
    ax.axvline(estimate, ls="--", color="black", label=f"estimate {estimate:.1f}")
    ax.set(title=title, xlabel=xlabel, ylabel="number of resamples")
    ax.legend()
    plt.show()


def check_widths_23(widths):
    """Verdict of 2.23 (widths for SIZES_23)."""
    widths = np.asarray(widths, dtype=float)
    ok = widths.shape == (len(SIZES_23),) and bool(np.all(np.diff(widths) < 0)) and 25 < widths[-1] < 45
    verdict("2.23", ok, "tes largeurs sont mesurées : lis le graphique, puis réponds dans la cellule 📝.",
            "tes largeurs ne sont pas celles attendues : ci_width doit renvoyer la largeur de l'intervalle à 80 %, "
            "avec des rééchantillons de la taille demandée (vérifie confidence et sample_size).")


def check_coverage_23(cover_20, cover_500):
    """Verdict of 2.23 (coverage of the two recipes)."""
    verdict("2.23", cover_20 > 0.95 and 0.65 <= cover_500 <= 0.93,
            "tes deux couvertures sont mesurées : compare-les à la promesse de 80 % (cellule 📝).",
            "couvertures inattendues : relis coverage (à chaque enquête, un NOUVEL échantillon de 500 valeurs tiré "
            "sans remise, puis son intervalle à 80 %).")


def check_n_boot_23(widths):
    """Verdict of 2.23 (widths for 100, 1 000 and 10 000 resamples)."""
    widths = np.asarray(widths, dtype=float)
    ok = len(set(np.round(widths, 9))) == 3 and widths.max() / widths.min() < 1.3
    verdict("2.23", ok, "tu as les trois largeurs : réponds dans la cellule 📝.",
            "ci_width doit utiliser son argument n_boot (et transmettre le même générateur) : relis-la.")'''

PART_C = Part("C", "Le bootstrap et son intervalle de confiance",
              "Fiche §2.6. Tu programmes le bootstrap et tu l'appliques à la masse des manchots Chinstrap ; puis tu "
              "mets à l'épreuve la recette du livre (des rééchantillons de 20) et tu compares tes intervalles avec "
              "l'outil professionnel, `scipy.stats.bootstrap`.", given=PART_C_GIVEN, exercises=[
    Ex("2.22", "🔨", 2, 30, "Bootstrap : distribution et intervalle de confiance",
       "programmer le bootstrap et en tirer un intervalle de confiance percentile.",
       "Ex 2.21, Ex 2.15 · 0A (`Callable`, arguments après `*`) · fiche §2.6 (🧮 intervalle de confiance) · "
       "livre §2.6 (figures 2.17 et 2.18)", thread="Penguins", tracks="R, M, C",
       body=MYLEARN_SHORT + r"""

Écris les deux fonctions du bootstrap.
- `bootstrap_distribution(x, statistic=np.mean, *, n_boot=1000, sample_size=None, rng=None)` : pour chaque rééchantillon, **dans l'ordre**, tire les indices `idx = rng.integers(0, n, size=sample_size)` (avec remise), puis calcule `statistic(x[idx])` ; renvoie les `n_boot` valeurs dans un array. `sample_size=None` veut dire $n$, la taille de `x`. L'étoile `*` de la signature oblige à nommer les arguments qui la suivent (`n_boot=500`, pas `500`) : on ne peut plus les confondre (0A.40). `statistic` est une fonction : `np.mean`, `np.median`, ou la tienne.
- `bootstrap_ci(x, statistic=np.mean, *, confidence=0.95, n_boot=1000, sample_size=None, rng=None)` : réutilise `bootstrap_distribution` avec les mêmes arguments, puis coupe $\frac{1 - c}{2}$ des valeurs de chaque côté, avec $c$ = `confidence` : les bornes sont les percentiles $50\,(1 - c)$ et $50\,(1 + c)$ de la distribution (ta fonction `percentile`). Renvoie un tuple de deux `float` ; refuse `confidence` hors de $]0, 1[$.

Vérifications, sur la masse des 68 manchots Chinstrap (`chinstrap_mass`), chaque appel avec **son** générateur `rng=np.random.default_rng(0)` (la cellule de vérification les calcule avec tes fonctions) :
a) la masse moyenne de l'échantillon (1 décimale) ;
b) l'intervalle de confiance à 95 % de la moyenne, `[bas, haut]` (1 décimale) ;
c) l'intervalle à 80 %, comme dans le livre (figure 2.18) ;
d) l'intervalle à 95 % de la **médiane** (`statistic=np.median`) ;
e) l'écart-type (ddof = 0) de la distribution bootstrap de la moyenne (1 décimale) : c'est l'**erreur type** (*standard error*), la dispersion de la moyenne d'un échantillon à l'autre.

La vérification trace ensuite l'histogramme des 1 000 moyennes, avec la moyenne de l'échantillon et l'intervalle à 95 % (l'équivalent des figures 2.17 et 2.18 du livre), puis lance les tests.""",
       check=RELOAD + r'''with wb.attempt("2.22"):
    st = mylearn.stats
    wb.check("2.22a", st.mean(chinstrap_mass))
    ci_22 = st.bootstrap_ci(chinstrap_mass, rng=np.random.default_rng(0))
    wb.check("2.22b", ci_22)
    wb.check("2.22c", st.bootstrap_ci(chinstrap_mass, confidence=0.8, rng=np.random.default_rng(0)))
    wb.check("2.22d", st.bootstrap_ci(chinstrap_mass, np.median, rng=np.random.default_rng(0)))
    boot_22 = st.bootstrap_distribution(chinstrap_mass, rng=np.random.default_rng(0))
    wb.check("2.22e", st.std(boot_22))
    plot_bootstrap(boot_22, st.mean(chinstrap_mass), ci_22, "1 000 bootstrap means of the Chinstrap mass", "mean mass (g)")
    run_stats_tests("test_bootstrap_")''',
       solution=r'''st = mylearn.stats
ci_22 = st.bootstrap_ci(chinstrap_mass, rng=np.random.default_rng(0))
boot_22 = st.bootstrap_distribution(chinstrap_mass, rng=np.random.default_rng(0))
print("sample mean:", st.mean(chinstrap_mass), "· 95 %:", ci_22,
      "· 80 %:", st.bootstrap_ci(chinstrap_mass, confidence=0.8, rng=np.random.default_rng(0)))
print("95 % interval of the median:", st.bootstrap_ci(chinstrap_mass, np.median, rng=np.random.default_rng(0)))
print("standard error:", st.std(boot_22), "· classical formula s / sqrt(n):",
      st.std(chinstrap_mass, ddof=1) / np.sqrt(len(chinstrap_mass)))
plot_bootstrap(boot_22, st.mean(chinstrap_mass), ci_22, "1 000 bootstrap means of the Chinstrap mass", "mean mass (g)")
run_stats_tests("test_bootstrap_", impl="ref")''',
       record=r'''wb.record("2.22a", st.mean(chinstrap_mass), decimals=1)
wb.record("2.22b", ci_22, decimals=1, mistakes={"c'est l'intervalle à 80 % : confidence vaut 0,95 par défaut": st.bootstrap_ci(chinstrap_mass, confidence=0.8, rng=np.random.default_rng(0)),
                                                "pour 95 %, coupe 2,5 % de CHAQUE côté : les percentiles 2,5 et 97,5 (pas 5 et 95)": st.bootstrap_ci(chinstrap_mass, confidence=0.9, rng=np.random.default_rng(0))})
wb.record("2.22c", st.bootstrap_ci(chinstrap_mass, confidence=0.8, rng=np.random.default_rng(0)), decimals=1,
          mistakes={"pour 80 %, coupe 10 % de CHAQUE côté : les percentiles 10 et 90 (pas 20 et 80)": st.bootstrap_ci(chinstrap_mass, confidence=0.6, rng=np.random.default_rng(0))})
wb.record("2.22d", st.bootstrap_ci(chinstrap_mass, np.median, rng=np.random.default_rng(0)), decimals=1)
wb.record("2.22e", st.std(boot_22), decimals=1, mistakes={"c'est l'écart-type des 68 MASSES : on demande celui des 1 000 moyennes bootstrap": st.std(chinstrap_mass)})''',
       note="À 95 %, la masse moyenne des Chinstrap est comprise entre 3 647 et 3 822 g environ : la moyenne de "
            "l'échantillon (3 733 g) est connue à ± 90 g près. L'intervalle à 80 % est plus étroit : il promet moins. "
            "Celui de la médiane va de 3 650 à 3 800 g, des valeurs « rondes », parce que la médiane de masses "
            "arrondies à 25 g est elle-même arrondie. L'erreur type (45,4 g) est proche de la formule classique "
            "$s / \\sqrt{n}$ (46,6 g, avec ddof = 1), que tu croiseras dans tout cours de statistique ; avec une "
            "infinité de rééchantillons, le bootstrap donnerait exactement l'écart-type **à ddof = 0** divisé par "
            "$\\sqrt{n}$, soit 46,3 g. Il retrouve ce résultat sans formule, et marche aussi pour une médiane ou une "
            "corrélation."),

    Ex("2.23", "🔬", 2, 30, "Bootstraps de 20 (livre) ou de n (aujourd'hui) ?",
       "mesurer l'effet de la taille des rééchantillons sur l'intervalle, et vérifier lequel tient sa promesse.",
       "Ex 2.22 · fiche §2.6 (🕰️ taille des rééchantillons, 🧮 intervalle de confiance) · livre §2.6",
       thread="synthétique", tracks="M, C",
       body=r"""Le livre (§2.6) part d'une population de 5 000 entiers de 0 à 1 000, en tire un échantillon de 500, puis fait 1 000 rééchantillons de **20** éléments seulement, et lit un intervalle à 80 %. La pratique actuelle prend des rééchantillons de la taille de l'échantillon, ici 500. Refais l'expérience avec la `population` et l'échantillon `sample_23` fournis : on connaît donc la vraie moyenne, `population.mean()`, ce qui n'arrive jamais en vrai.

Écris, avec tes fonctions de `mylearn.stats` :
1. `ci_width(sample_size, rng, n_boot=1000)` : la largeur (haut − bas) de l'intervalle à 80 % de la moyenne de `sample_23`, avec `n_boot` rééchantillons de taille `sample_size` ;
2. `widths_by_size()` : la liste des largeurs pour chaque taille de `SIZES_23` (de 5 à 500), chaque fois avec `np.random.default_rng(1)` ;
3. `coverage(sample_size, n_repeats, rng)` : la proportion d'intervalles à 80 % qui contiennent la vraie moyenne quand on refait **toute** l'enquête `n_repeats` fois. À chaque fois : un nouvel échantillon de 500 valeurs tiré **sans** remise dans `population` (ton `sample`, 2.21), puis son intervalle bootstrap à 80 %, avec 500 rééchantillons de taille `sample_size` ; le même générateur `rng` sert à tout. Un intervalle « à 80 % » honnête doit contenir la vraie valeur environ 80 fois sur 100 (fiche, 🧮) : cette proportion s'appelle la **couverture** (*coverage*) de la méthode.

La vérification affiche les largeurs et le rapport des largeurs (20 contre 500), trace la largeur selon la taille en échelle logarithmique, mesure la couverture des deux recettes, puis compare 100, 1 000 et 10 000 rééchantillons de taille 500. Réponds dans la cellule 📝 :
- De combien l'intervalle du livre est-il trop large ? Comment la largeur varie-t-elle avec la taille des rééchantillons (quand la taille est multipliée par 4) ?
- Laquelle des deux recettes tient sa promesse de 80 % ?
- Augmenter le **nombre** de rééchantillons rétrécit-il l'intervalle ? Que change-t-il ?""",
       given=r'''rng_23 = np.random.default_rng(2023)
population = rng_23.integers(0, 1001, size=5000)            # 5 000 integers from 0 to 1 000 (book, §2.6)
sample_23 = population[rng_23.permutation(5000)[:500]]      # a sample of 500 values, drawn without replacement
SIZES_23 = [5, 10, 20, 50, 100, 200, 500]
N_REPEATS_23 = 100 if FAST_MODE else 400                    # number of surveys redone for the coverage
print(f"true mean of the population: {population.mean():.1f} · mean of the sample: {sample_23.mean():.1f}")''',
       todo=r'''def ci_width(sample_size, rng, n_boot=1000):
    """Width (high - low) of the 80 % bootstrap interval of the mean of sample_23 (n_boot resamples of sample_size)."""
    raise NotImplementedError("ci_width() is not written yet")


def widths_by_size():
    """The list of ci_width(size, np.random.default_rng(1)) for every size of SIZES_23."""
    raise NotImplementedError("widths_by_size() is not written yet")


def coverage(sample_size, n_repeats, rng):
    """Share of n_repeats new surveys (500 values drawn WITHOUT replacement from population) whose 80 % bootstrap
    interval (500 resamples of size sample_size) contains population.mean(). One generator rng for everything."""
    raise NotImplementedError("coverage() is not written yet")''',
       check=r'''with wb.attempt("2.23"):
    widths_23 = widths_by_size()
    for size, width in zip(SIZES_23, widths_23):
        print(f"resamples of {size:>3}: width of the 80 % interval = {width:6.1f}")
    ratio_23 = widths_23[SIZES_23.index(20)] / widths_23[SIZES_23.index(500)]
    print(f"ratio of the widths, 20 against 500: {ratio_23:.2f}")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.loglog(SIZES_23, widths_23, "o-")
    ax.set(xlabel="size of each resample (log scale)", ylabel="width of the 80 % interval (log scale)")
    plt.show()
    check_widths_23(widths_23)
    cover_20 = coverage(20, N_REPEATS_23, np.random.default_rng(7))
    cover_500 = coverage(500, N_REPEATS_23, np.random.default_rng(7))
    print(f"share of the 80 % intervals that contain the true mean: size 20 → {cover_20:.2f} · size 500 → {cover_500:.2f}")
    check_coverage_23(cover_20, cover_500)
    by_n_boot = [ci_width(500, np.random.default_rng(1), n_boot=b) for b in (100, 1000, 10_000)]
    print("width for 100, 1 000 and 10 000 resamples of size 500:", np.round(by_n_boot, 1))
    check_n_boot_23(by_n_boot)''',
       solution=r'''def ci_width(sample_size, rng, n_boot=1000):
    """Width (high - low) of the 80 % bootstrap interval of the mean of sample_23 (n_boot resamples of sample_size)."""
    low, high = mylearn.stats.bootstrap_ci(sample_23, confidence=0.8, n_boot=n_boot, sample_size=sample_size, rng=rng)
    return high - low


def widths_by_size():
    """The list of ci_width(size, np.random.default_rng(1)) for every size of SIZES_23."""
    return [ci_width(size, np.random.default_rng(1)) for size in SIZES_23]


def coverage(sample_size, n_repeats, rng):
    """Share of n_repeats new surveys (500 values drawn WITHOUT replacement from population) whose 80 % bootstrap
    interval (500 resamples of size sample_size) contains population.mean(). One generator rng for everything."""
    true_mean = population.mean()
    hits = 0
    for _ in range(n_repeats):
        new_sample = mylearn.stats.sample(population, 500, replace=False, rng=rng)
        low, high = mylearn.stats.bootstrap_ci(new_sample, confidence=0.8, n_boot=500, sample_size=sample_size, rng=rng)
        hits += low <= true_mean <= high
    return hits / n_repeats


widths_23 = widths_by_size()
ratio_23 = widths_23[SIZES_23.index(20)] / widths_23[SIZES_23.index(500)]
print(dict(zip(SIZES_23, np.round(widths_23, 1))), "· ratio 20 / 500:", round(ratio_23, 2))
print("width × √size:", np.round(np.array(widths_23) * np.sqrt(SIZES_23)))
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.loglog(SIZES_23, widths_23, "o-")
ax.set(xlabel="size of each resample (log scale)", ylabel="width of the 80 % interval (log scale)")
plt.show()
cover_20 = coverage(20, N_REPEATS_23, np.random.default_rng(7))
cover_500 = coverage(500, N_REPEATS_23, np.random.default_rng(7))
print(f"coverage: size 20 → {cover_20:.2f} · size 500 → {cover_500:.2f}")
print("width for 100, 1 000 and 10 000 resamples:", [round(ci_width(500, np.random.default_rng(1), n_boot=b), 1) for b in (100, 1000, 10_000)])''',
       after=[("todo_md", "📝 **Mes réponses** (2.23) :\n\n1. …\n2. …\n3. …"),
              ("solution_md", "**Réponses (2.23)** :\n\n"
               "1. Avec des rééchantillons de 20, l'intervalle à 80 % est environ **4,7 fois** plus large qu'avec des "
               "rééchantillons de 500, à peu près $\\sqrt{500 / 20} = 5$. Sur le graphique en échelle logarithmique, "
               "les points sont presque alignés sur une droite de pente $-\\frac{1}{2}$ : quand la taille est "
               "multipliée par 4, la largeur est divisée par 2. La largeur varie comme $\\frac{1}{\\sqrt{\\text{taille}}}$ "
               "(le produit largeur × $\\sqrt{\\text{taille}}$ reste vers 700-800). Un rééchantillon de 20 imite un "
               "échantillon de 20 : il mesure l'incertitude qu'on aurait avec 20 valeurs, pas avec 500.\n"
               "2. L'intervalle du livre contient la vraie moyenne **à chaque fois** (100 %) : il promet 80 % et "
               "donne beaucoup plus, parce qu'il est bien trop large. Avec des rééchantillons de 500, la couverture "
               "est proche de 80 % (84 % ici, sur 100 enquêtes, 80 % en mode complet sur 400 ; l'aléa est de ± 4 "
               "points sur 100 enquêtes) : la promesse est tenue.\n"
               "3. Non : 100, 1 000 et 10 000 rééchantillons donnent des largeurs voisines. Le **nombre** de "
               "rééchantillons rend les bornes plus **stables** d'une exécution à l'autre (moins de bruit dans les "
               "percentiles) ; c'est la **taille** de l'échantillon, et donc des rééchantillons, qui fixe la largeur.")],
       note="Le livre présente la petite taille des rééchantillons comme un avantage (plus rapides à calculer). "
            "L'expérience montre le prix : un intervalle quelque 5 fois trop large, qui annonce une incertitude "
            "qu'on n'a pas. Aujourd'hui : des rééchantillons de taille $n$, de 1 000 à 10 000 fois (fiche, 🕰️)."),

    Ex("2.24", "📦", 2, 15, "Comparer avec `scipy.stats.bootstrap`",
       "utiliser l'outil professionnel du bootstrap en lisant sa documentation, et comparer ses intervalles aux tiens.",
       "Ex 2.22 · fiche §2.6 (🕰️) · documentation de `scipy.stats.bootstrap`", thread="Penguins", tracks="R, C",
       body=r"""SciPy fait le bootstrap en une ligne. Lis d'abord sa documentation (`help(scipy_stats.bootstrap)`, ou [en ligne](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)) : le premier argument, `data`, est une **séquence d'échantillons**, même quand il n'y en a qu'un. On écrit donc `(chinstrap_mass,)`, un tuple d'un seul élément : sans la virgule, SciPy croit recevoir 68 échantillons d'une seule valeur et lève une erreur d'axe (essaie).

Avec `confidence_level=0.95`, `n_resamples=9999` (la valeur par défaut) et, à chaque appel, `rng=np.random.default_rng(0)` :
a) `ci_percentile` : l'intervalle **percentile** (`method="percentile"`) de la moyenne de `chinstrap_mass`, `[bas, haut]`, arrondi au gramme ;
b) `ci_bca` : l'intervalle de la méthode par défaut, `"BCa"` (arrondi au gramme) ;
c) `standard_error` : l'erreur type calculée par SciPy (l'attribut `standard_error` du résultat, 1 décimale) ; compare-la à 2.22 e.

La vérification compare aussi ton intervalle percentile de 2.22 b à celui de SciPy. Dans tes notes : pourquoi ne sont-ils pas identiques, alors que la méthode est la même (compare aussi les 1 000 premières valeurs de `bootstrap_distribution`, l'attribut du résultat de SciPy, avec ta distribution de 2.22) ?""",
       todo=r'''ci_percentile = ...    # a) [low, high]
ci_bca = ...           # b) [low, high]
standard_error = ...   # c)''',
       check=r'''ok_24a = wb.check("2.24a", ci_percentile)
wb.check("2.24b", ci_bca)
wb.check("2.24c", standard_error)
with wb.attempt("2.24"):
    mine_24 = mylearn.stats.bootstrap_ci(chinstrap_mass, rng=np.random.default_rng(0))
    print(f"yours (1 000 resamples): [{mine_24[0]:.1f}, {mine_24[1]:.1f}]")
    if ok_24a:
        gap_24 = max(abs(mine_24[0] - ci_percentile[0]), abs(mine_24[1] - ci_percentile[1]))
        verdict("2.24", gap_24 < 15, f"ton intervalle et celui de SciPy diffèrent d'au plus {fr(gap_24, 1)} g : même méthode, "
                "mais pas le même nombre de rééchantillons.",
                f"ton intervalle et celui de SciPy diffèrent de {fr(gap_24, 1)} g : c'est beaucoup pour la même méthode.")''',
       solution=r'''result_pct = scipy_stats.bootstrap((chinstrap_mass,), np.mean, confidence_level=0.95, n_resamples=9999,
                                  method="percentile", rng=np.random.default_rng(0))
result_bca = scipy_stats.bootstrap((chinstrap_mass,), np.mean, confidence_level=0.95, n_resamples=9999,
                                  rng=np.random.default_rng(0))
ci_percentile = [result_pct.confidence_interval.low, result_pct.confidence_interval.high]
ci_bca = [result_bca.confidence_interval.low, result_bca.confidence_interval.high]
standard_error = result_pct.standard_error
mine_24 = mylearn.stats.bootstrap_ci(chinstrap_mass, rng=np.random.default_rng(0))
print("percentile:", np.round(ci_percentile, 1), "· BCa:", np.round(ci_bca, 1), "· standard error:", round(standard_error, 2))
print("yours (1 000 resamples):", np.round(mine_24, 1))
try:
    scipy_stats.bootstrap(chinstrap_mass, np.mean, rng=np.random.default_rng(0))   # the classic mistake: no tuple
except Exception as error:  # noqa: BLE001
    print("without the tuple:", type(error).__name__, "-", error)''',
       record=r'''legacy_24 = scipy_stats.bootstrap((chinstrap_mass,), np.mean, confidence_level=0.95, n_resamples=9999,
                                  method="percentile", random_state=0).confidence_interval
shared_rng_24 = np.random.default_rng(0)                       # one generator used for both calls: a classic slip
scipy_stats.bootstrap((chinstrap_mass,), np.mean, n_resamples=9999, method="percentile", rng=shared_rng_24)
shared_24 = scipy_stats.bootstrap((chinstrap_mass,), np.mean, n_resamples=9999, rng=shared_rng_24).confidence_interval
wb.record("2.24a", ci_percentile, decimals=0, mistakes={"c'est l'intervalle BCa, la méthode par défaut : précise method=\"percentile\"": ci_bca,
                                                        "un entier comme random_state crée l'ANCIEN générateur de NumPy : passe rng=np.random.default_rng(0)": list(legacy_24)})
wb.record("2.24b", ci_bca, decimals=0, mistakes={"c'est l'intervalle percentile : ici, on veut la méthode par défaut (BCa)": ci_percentile,
                                                 "crée un NOUVEAU générateur de graine 0 pour cet appel : celui de a) a déjà servi": list(shared_24)})
wb.record("2.24c", standard_error, decimals=1)''',
       note="Les deux intervalles percentile utilisent la même méthode, et même les mêmes premiers rééchantillons : "
            "SciPy tire tous ses indices d'un coup (`rng.integers(0, n, (9999, n))`), ce qui redonne exactement la "
            "suite de tes tirages ; les 1 000 premières valeurs de `result_pct.bootstrap_distribution` sont les "
            "tiennes. La seule différence est le nombre de rééchantillons, 1 000 contre 9 999 : l'écart de quelques "
            "grammes est le **bruit de Monte-Carlo** de 1 000 rééchantillons, et `bootstrap_ci(..., n_boot=9999)` "
            "redonne l'intervalle de SciPy à l'identique. L'intervalle BCa corrige le "
            "biais et l'asymétrie de la distribution bootstrap ; ici, sur des moyennes presque symétriques, il ne "
            "diffère que d'un ou deux grammes. En pratique : `scipy.stats.bootstrap`, avec la méthode par défaut, "
            "et `rng` pour la reproductibilité."),
])

# ---------------------------------------------------------------------------
# Part D: high dimension, covariance and correlation, ddof, Anscombe (2.25 to 2.32)
# ---------------------------------------------------------------------------
PART_D_GIVEN = r'''bill_length = measured["bill_length_mm"].to_numpy()     # in mm
bill_depth = measured["bill_depth_mm"].to_numpy()       # in mm'''

CLOUDS_27 = r'''def make_clouds_27():
    """Six clouds of 60 points (x, y), named A to F in a shuffled order: guess by eye, do not read the recipes."""
    rng = np.random.default_rng(97)
    x = rng.normal(0, 1, (6, 60))
    e = rng.normal(0, 1, (6, 60))
    y = np.array([x[0] ** 2 + 0.3 * e[0], -x[1] + 0.45 * e[1], e[2], x[3] + 0.15 * e[3],
                  -0.6 * x[4] + e[4], 0.35 * x[5] + e[5]])
    x[2, 0], y[2, 0] = 9.0, 9.0
    order = rng.permutation(6)
    return {letter: (x[k], y[k]) for letter, k in zip("ABCDEF", order)}


clouds_27 = make_clouds_27()
R_VALUES_27 = sorted(round(float(np.corrcoef(x, y)[0, 1]), 2) for x, y in clouds_27.values())
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (letter, (x, y)) in zip(axes.ravel(), clouds_27.items()):
    ax.scatter(x, y, s=12)
    ax.set_title(letter, fontsize=14)
plt.show()
print("the six correlations, in increasing order (not in the order of the clouds):", R_VALUES_27)'''

ANSCOMBE_DATA = r'''x_123 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
ANSCOMBE = {   # F. J. Anscombe, "Graphs in Statistical Analysis", 1973 (public domain)
    "I": (x_123, np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])),
    "II": (x_123, np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])),
    "III": (x_123, np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])),
    "IV": (np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float),
           np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])),
}'''

ZSCORE_VERSIONS = r'''def zscore_ok(x, ddof=0, axis=None):
    """A correct version, with the contract of mylearn.stats.zscore."""
    x = np.asarray(x, dtype=float)
    std = x.std(axis=axis, ddof=ddof, keepdims=True)
    if x.size == 0 or np.any(std == 0):
        raise ValueError("zscore() is undefined for constant data (standard deviation 0)")
    return (x - x.mean(axis=axis, keepdims=True)) / std


def zscore_bug_1(x, ddof=0, axis=None):
    x = np.asarray(x, dtype=float)
    var = x.var(axis=axis, ddof=ddof, keepdims=True)
    if x.size == 0 or np.any(var == 0):
        raise ValueError("constant data")
    return (x - x.mean(axis=axis, keepdims=True)) / var


def zscore_bug_2(x, ddof=0, axis=None):
    x = np.asarray(x, dtype=float)
    std = x.std(axis=axis, ddof=ddof, keepdims=True)
    return (x - x.mean(axis=axis, keepdims=True)) / np.where(std == 0, 1.0, std)


def zscore_bug_3(x, ddof=0, axis=None):
    x = np.asarray(x, dtype=float)
    std = x.std(ddof=ddof)
    if x.size == 0 or std == 0:
        raise ValueError("constant data")
    return (x - x.mean()) / std'''

HEART = r'''def heart_points(n=40):
    """n points (hx, hy) along a heart-shaped curve (the classic parametric heart), in the order of the curve."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    hx = 16 * np.sin(t) ** 3
    hy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return hx, hy


def plot_quartet_and(x_new, y_new, name):
    """The four sets of ANSCOMBE and (x_new, y_new), each with its least-squares line."""
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharex=True, sharey=True)
    for ax, (label, (x, y)) in zip(axes, [*ANSCOMBE.items(), (name, (x_new, y_new))]):
        slope, intercept = np.polyfit(x, y, 1)
        ax.scatter(x, y, s=14)
        ax.plot([0, 22], [intercept, intercept + 22 * slope], color="0.3")
        ax.set_title(f"{label}: y = {intercept:.2f} + {slope:.2f} x")
    plt.show()'''

PART_D = Part("D", "Grande dimension, covariance et corrélation, ddof, Anscombe",
              "Fiche §2.7 à §2.9. Tu mesures des distances entre images de MNIST, tu écris covariance et "
              "corrélation et tu les appliques aux manchots, d'abord tous ensemble puis espèce par espèce, tu traques "
              "les pièges de ddof, "
              "puis tu redessines le quartet d'Anscombe et tu fabriques le tien.", given=PART_D_GIVEN, exercises=[
    Ex("2.25", "📦", 2, 25, "Distances entre chiffres dans l'espace à 784 dimensions",
       "calculer des distances entre images vues comme des points à 784 dimensions, et constater qu'en grande "
       "dimension, des points au hasard sont tous à peu près à la même distance.",
       "0B (distance, 101.3.2) · Ex 2.11 · fiche §2.7 · livre §2.7", thread="MNIST", tracks="M, C",
       body=r"""`X_digits` contient 2 000 images de MNIST aplaties en vecteurs de 784 pixels, entre 0 et 1, et `y_digits` leurs labels. La distance entre deux images est la distance euclidienne de 0B, avec 784 termes : `np.linalg.norm(a - b)`.

a) `dist_01` : la distance entre les images 0 et 1 (2 décimales).
b) `nearest_index` : l'indice (le numéro de ligne dans `X_digits`) de l'image la plus proche de l'image 0, parmi les 1 999 autres. Calcule toutes les distances d'un coup, `np.linalg.norm(X_digits - X_digits[0], axis=1)` (broadcasting, 0A), puis écarte l'image 0 elle-même (sa distance vaut 0 : remplace-la par `np.inf`). Est-ce le même chiffre ? (la vérification affiche les deux labels)
c) `nn_accuracy` : pour chacune des **500 premières** images, cherche la plus proche parmi les 1 999 autres (une boucle sur les 500 images) ; quelle proportion a le même label que sa plus proche voisine (3 décimales) ? C'est déjà un classifieur, celui des plus proches voisins (ch. 13).
d) `same_vs_other` : parmi les 500 premières images, le rapport (distance moyenne entre deux images de chiffres **différents**) / (distance moyenne entre deux images du **même** chiffre), sur toutes les paires d'images distinctes (2 décimales).
e) `contrast_random` : le **contraste** d'un nuage de points est $\frac{d_{\max} - d_{\min}}{d_{\min}}$, où $d_{\min}$ et $d_{\max}$ sont la plus petite et la plus grande distance entre le point 0 et les autres points. Calcule-le pour 1 000 points tirés uniformément dans $[0, 1]^{784}$, `np.random.default_rng(0).random((1000, 784))` (4 décimales).
f) `contrast_mnist` : le contraste des 1 000 premières images de `X_digits` (2 décimales).

La cellule suivante trace le contraste de points tirés au hasard selon la dimension. Dans tes notes : que devient le contraste quand la dimension grandit ? Pourquoi les images de MNIST se comportent-elles autrement que des points tirés au hasard (pense à 2.11, question 6) ?""",
       given=r'''X_digits, y_digits = wb.datasets.load_mnist("train", n=2000, flatten=True, normalize=True, seed=0)   # 2 000 images
print(X_digits.shape, X_digits.dtype)''',
       todo=r'''dist_01 = ...           # a)
nearest_index = ...     # b)
nn_accuracy = ...       # c)
same_vs_other = ...     # d)
contrast_random = ...   # e)
contrast_mnist = ...    # f)''',
       check=r'''wb.check("2.25a", dist_01)
if wb.check("2.25b", nearest_index):
    print(f"   label of image 0: {y_digits[0]} · label of its nearest image: {y_digits[nearest_index]}")
wb.check("2.25c", nn_accuracy)
wb.check("2.25d", same_vs_other)
wb.check("2.25e", contrast_random)
wb.check("2.25f", contrast_mnist)''',
       solution=r'''def distances_to(points, i):
    """Distances from point i to all the points (point i itself: infinity)."""
    d = np.linalg.norm(points - points[i], axis=1)
    d[i] = np.inf
    return d


def contrast(points):
    """(d_max - d_min) / d_min for the distances between point 0 and the other points."""
    d = distances_to(points, 0)[1:]
    return (d.max() - d.min()) / d.min()


dist_01 = float(np.linalg.norm(X_digits[0] - X_digits[1]))
nearest_index = int(np.argmin(distances_to(X_digits, 0)))
nn_accuracy = float(np.mean([y_digits[np.argmin(distances_to(X_digits, i))] == y_digits[i] for i in range(500)]))
X500, y500 = X_digits[:500], y_digits[:500]
D500 = np.array([np.linalg.norm(X500 - X500[i], axis=1) for i in range(500)])
upper = np.triu_indices(500, k=1)                                  # every pair (i, j) with i < j, once
same_digit = (y500[:, None] == y500[None, :])[upper]
same_vs_other = float(D500[upper][~same_digit].mean() / D500[upper][same_digit].mean())
contrast_random = float(contrast(np.random.default_rng(0).random((1000, 784))))
contrast_mnist = float(contrast(X_digits[:1000]))
print(dist_01, nearest_index, y_digits[0], y_digits[nearest_index], nn_accuracy, same_vs_other, contrast_random, contrast_mnist)''',
       record=r'''wb.record("2.25a", dist_01, decimals=2, mistakes={"c'est le CARRÉ de la distance : prends la racine": dist_01 ** 2})
wb.record("2.25b", nearest_index, mistakes={"l'image 0 est à distance 0 d'elle-même : écarte-la (np.inf) avant de chercher la plus proche": 0})
wb.record("2.25c", nn_accuracy, decimals=3, mistakes={"une image est à distance 0 d'elle-même : écarte-la (np.inf) avant de chercher sa plus proche voisine": 1.0})
with_diagonal = (y500[:, None] == y500[None, :])
wb.record("2.25d", same_vs_other, decimals=2, mistakes={"ne compte pas une image avec elle-même (distance 0) parmi les paires du même chiffre": D500[~with_diagonal].mean() / D500[with_diagonal].mean(),
                                                        "c'est le rapport inverse : chiffres DIFFÉRENTS divisé par MÊME chiffre": 1 / same_vs_other})
wb.record("2.25e", contrast_random, decimals=4)
wb.record("2.25f", contrast_mnist, decimals=2)''',
       after=[("code", r'''dims_25 = [1, 2, 3, 10, 30, 100, 300, 784, 3072]
contrasts_25 = []
for dim in dims_25:
    points = np.random.default_rng(1).random((1000, dim))          # 1 000 uniform points in [0, 1]^dim
    d = np.linalg.norm(points[1:] - points[0], axis=1)
    contrasts_25.append((d.max() - d.min()) / d.min())
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.loglog(dims_25, contrasts_25, "o-")
ax.set(xlabel="dimension (log scale)", ylabel="contrast (d_max - d_min) / d_min (log scale)",
       title="Random points: the nearest and the farthest become alike")
plt.show()''')],
       note="En moyenne, deux images de chiffres différents ne sont qu'environ 16 % plus éloignées que deux images "
            "du même chiffre : en 784 dimensions, les distances se ressemblent beaucoup. Pour des points tirés "
            "au hasard, c'est pire : le point le plus lointain n'est qu'à 14 % de plus que le plus proche "
            "(contraste 0,14), contre un rapport de plus de 100 en dimension 2. C'est un visage du **fléau de la "
            "dimension** : « le plus proche » perd son sens. Les images de MNIST gardent un contraste d'environ "
            "1,85 : elles ne remplissent pas l'espace au hasard, elles vivent près d'une structure de bien plus "
            "petite dimension (2.11, question 6), et la recherche du plus proche voisin marche (90 % de bons "
            "labels ici)."),

    Ex("2.26", "🔨", 2, 25, "Covariance et corrélation",
       "écrire la covariance et la corrélation, et les interpréter sur les mesures des manchots.",
       "Ex 2.7, Ex 2.15 · fiche §2.8.1, §2.8.2", thread="Penguins", tracks="R, M, C",
       body=MYLEARN_SHORT + r"""

Écris `covariance(x, y, ddof=0)` et `correlation(x, y)`.
- `covariance` : la somme des produits des écarts à la moyenne, $(x_i - \bar{x})(y_i - \bar{y})$, divisée par $n - \mathrm{ddof}$ ; `x` et `y` doivent être deux tableaux à une dimension de même longueur (sinon, `ValueError`).
- `correlation` : la covariance divisée par les deux écarts-types, **avec le même ddof** (fiche §2.8.2) ; refuse une variable constante, dont l'écart-type est nul. Les arrondis peuvent donner `1.0000000000000002` : ramène le résultat dans $[-1, 1]$ (`np.clip`).

Vérifications (la cellule de vérification les calcule avec tes fonctions) :
a) la covariance (ddof = 0) de la nageoire (mm) et de la masse (g) (1 décimale) ;
b) leur corrélation (3 décimales) ;
c) la covariance de la nageoire en **cm** et de la masse en **kg** (3 décimales) ; la vérification calcule aussi la corrélation en cm et en kg (2.8, question 6) ;
d) la corrélation de la **longueur** et de l'**épaisseur** du bec, sur tous les manchots (3 décimales). Un bec plus long serait-il plus fin ? Garde ta réponse : tu y reviens en 2.28.

La vérification trace les deux nuages de points de b et de d, puis lance les tests.""",
       check=RELOAD + r'''with wb.attempt("2.26"):
    st = mylearn.stats
    wb.check("2.26a", st.covariance(flipper, mass))
    wb.check("2.26b", st.correlation(flipper, mass))
    wb.check("2.26c", st.covariance(flipper / 10, mass / 1000))
    verdict("2.26", abs(st.correlation(flipper / 10, mass / 1000) - st.correlation(flipper, mass)) < 1e-12,
            "en cm et en kg, la corrélation ne change pas.",
            "la corrélation doit rester la même en cm et en kg : utilise le même ddof pour la covariance et les écarts-types.")
    wb.check("2.26d", st.correlation(bill_length, bill_depth))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(flipper, mass, s=10)
    axes[0].set(xlabel="flipper_length_mm", ylabel="body_mass_g", title=f"r = {st.correlation(flipper, mass):.3f}")
    axes[1].scatter(bill_length, bill_depth, s=10)
    axes[1].set(xlabel="bill_length_mm", ylabel="bill_depth_mm", title=f"r = {st.correlation(bill_length, bill_depth):.3f}")
    plt.show()
    run_stats_tests("(test_covariance_ or test_correlation_) and not matrix")''',
       solution=r'''st = mylearn.stats
print("covariance:", st.covariance(flipper, mass), "mm·g · correlation:", st.correlation(flipper, mass))
print("in cm and kg: covariance", st.covariance(flipper / 10, mass / 1000), "· correlation",
      st.correlation(flipper / 10, mass / 1000))
print("bill length and bill depth:", st.correlation(bill_length, bill_depth))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(flipper, mass, s=10)
axes[0].set(xlabel="flipper_length_mm", ylabel="body_mass_g", title=f"r = {st.correlation(flipper, mass):.3f}")
axes[1].scatter(bill_length, bill_depth, s=10)
axes[1].set(xlabel="bill_length_mm", ylabel="bill_depth_mm", title=f"r = {st.correlation(bill_length, bill_depth):.3f}")
plt.show()
run_stats_tests("(test_covariance_ or test_correlation_) and not matrix", impl="ref")''',
       record=r'''wb.record("2.26a", st.covariance(flipper, mass), decimals=1, mistakes={"ddof = 0 par défaut : divise par n, pas par n − 1": st.covariance(flipper, mass, ddof=1)})
wb.record("2.26b", st.correlation(flipper, mass), decimals=3, mistakes={"garde le même ddof pour la covariance et pour les écarts-types": st.covariance(flipper, mass, ddof=1) / (st.std(flipper) * st.std(mass))})
wb.record("2.26c", st.covariance(flipper / 10, mass / 1000), decimals=3,
          mistakes={"les deux unités changent : la nageoire est divisée par 10 ET la masse par 1 000": st.covariance(flipper / 10, mass)})
wb.record("2.26d", st.correlation(bill_length, bill_depth), decimals=3, mistakes={"le signe compte : regarde le sens du nuage de droite": -st.correlation(bill_length, bill_depth)})''',
       note="La nageoire et la masse sont très liées ($r = 0{,}871$) : on peut prédire l'une à partir de l'autre avec "
            "une droite (la régression, ch. 9). Leur covariance, 9 796 mm·g, n'est pas interprétable seule : en cm et "
            "en kg, elle est divisée par $10 \\times 1\\,000$ et vaut 0,980, alors que la corrélation ne bouge pas. "
            "La corrélation longueur–épaisseur du bec est faiblement **négative** ($-0{,}235$) : sur le nuage, les "
            "points forment plusieurs paquets… la suite en 2.28."),

    Ex("2.27", "📈", 2, 20, "Deviner la corrélation d'un nuage de points",
       "estimer une corrélation à l'œil, et repérer les nuages où elle trompe.",
       "Ex 2.26 · fiche §2.8.2 (⚠️ deux contresens) · livre §2.8.2 (figures 2.25 à 2.29)", thread="synthétique",
       tracks="R, M",
       body=r"""Les six nuages A à F ci-dessous ont chacun 60 points, et leurs corrélations sont affichées sous la figure, **dans l'ordre croissant**, pas dans l'ordre des nuages. **Sans rien calculer**, attribue à chaque nuage sa corrélation.

a) `r_guess` : la liste des six corrélations, dans l'ordre des nuages A, B, C, D, E, F (recopie les valeurs affichées) ;
b) `r_without_outlier` : un des nuages doit presque toute sa corrélation à un seul point. Retire ce point (le plus éloigné du centre du nuage) et calcule la corrélation des 59 autres (2 décimales), avec `np.corrcoef` ou ta fonction `correlation`. Les nuages sont dans `clouds_27`, un dictionnaire `{"A": (x, y), …}`.

La cellule **Réponse** affiche ensuite la vraie corrélation de chaque nuage. Puis, dans la cellule 📝 : quel nuage a une corrélation proche de 0 alors que $y$ dépend fortement de $x$ ? Pourquoi la corrélation ne le voit-elle pas ?""",
       given=CLOUDS_27,
       todo=r'''r_guess = ...             # a) six values, in the order of the clouds A to F
r_without_outlier = ...   # b)''',
       check=r'''wb.check("2.27a", r_guess)
wb.check("2.27b", r_without_outlier)''',
       solution=r'''r_by_cloud = {letter: round(float(np.corrcoef(x, y)[0, 1]), 2) for letter, (x, y) in clouds_27.items()}
r_guess = list(r_by_cloud.values())
outlier_letter = next(letter for letter, (x, y) in clouds_27.items() if np.max(np.hypot(x, y)) > 8)
x_27, y_27 = clouds_27[outlier_letter]
far = np.argmax(np.hypot(x_27 - x_27.mean(), y_27 - y_27.mean()))       # the point farthest from the centre
r_without_outlier = float(np.corrcoef(np.delete(x_27, far), np.delete(y_27, far))[0, 1])
print(r_by_cloud, "· cloud with the isolated point:", outlier_letter, "· without it:", round(r_without_outlier, 3))''',
       record=r'''wb.record("2.27a", r_guess, decimals=2)
wb.record("2.27b", r_without_outlier, decimals=2, mistakes={"c'est la corrélation AVEC le point isolé : retire-le d'abord": r_by_cloud[outlier_letter]})''',
       after=[("md", "**Réponse** : exécute la cellule pour voir la vraie corrélation de chaque nuage."),
              ("code", guarded(r'''for letter, (x, y) in clouds_27.items():
    print(f"{letter}: r = {np.corrcoef(x, y)[0, 1]:+.2f}")''', ['r_guess'],
                               "⏳ Ex 2.27 : donne d'abord ta réponse a), puis relance cette cellule.")),
              ("todo_md", "📝 **Mes réponses** (2.27) : quel nuage a une corrélation proche de 0 alors que $y$ "
                          "dépend fortement de $x$ ? Pourquoi la corrélation ne le voit-elle pas ?\n\n…"),
              ("solution_md", "**Réponses (2.27)** : le nuage F, en forme de U (une parabole) : $y$ se déduit "
                              "presque de $x$, mais il monte à droite et à gauche du centre. Les produits des écarts "
                              "$(x_i - \\bar{x})(y_i - \\bar{y})$ sont positifs d'un côté et négatifs de l'autre, et se "
                              "compensent : la corrélation mesure seulement la part **linéaire** du lien (fiche, "
                              "⚠️ « Deux contresens »). À l'inverse, le nuage D n'a aucun lien, mais un seul point "
                              "très éloigné lui donne une corrélation de 0,58 : sans lui, elle tombe à 0,01. Deux "
                              "raisons de toujours regarder le nuage avant de croire un $r$.")],
       note="Les nuages A (0,99) et E (−0,91) se reconnaissent vite : des points serrés autour d'une droite. B "
            "(0,31) et C (−0,45) sont plus difficiles : un nuage large mais penché. Les deux pièges sont F (une "
            "parabole, $r \\approx 0$ malgré un lien très fort) et D (un point isolé qui fabrique une corrélation). "
            "Tu retrouveras ce point **influent** dans le jeu IV d'Anscombe (2.12, 2.30)."),

    Ex("2.28", "🔨", 2, 25, "Matrices de covariance et de corrélation des manchots",
       "calculer les matrices de covariance et de corrélation d'un tableau, et comparer la corrélation de "
       "l'ensemble à celle de chaque groupe.",
       "Ex 2.26 · 0B (produit matriciel) · fiche §2.8 (🧮 matrice de covariance)", thread="Penguins", tracks="M, C",
       body=MYLEARN_SHORT + r"""

Écris `covariance_matrix(X, ddof=0)` et `correlation_matrix(X)`.
- `covariance_matrix` : `X` a une ligne par manchot et une colonne par mesure. Centre chaque colonne (soustrais la moyenne de la colonne), puis la case $(j, k)$ est la somme des produits des colonnes centrées $j$ et $k$, divisée par $n - \mathrm{ddof}$. Deux boucles sur $j$ et $k$ suffisent ; en une ligne, c'est `Xc.T @ Xc / (n - ddof)` (0B : vérifie les formes, $(p, n) \times (n, p)$ donne $(p, p)$).
- `correlation_matrix` : divise chaque case $(j, k)$ de la matrice de covariance par $\sigma_j\,\sigma_k$ ; la diagonale vaut 1.

Vérifications, sur `X_measures` (342 manchots × 4 mesures) :
a) la matrice de corrélation (4 × 4, 2 décimales) ; la vérification l'affiche avec le nom des mesures ;
b) `most_negative` : d'après cette matrice, la paire de mesures la plus **négativement** corrélée (une liste de deux noms de `MEASURES`).

Ensuite, une question sur les espèces.""",
       todo=r'''most_negative = ...   # b) a list of two names of MEASURES''',
       check=RELOAD + r'''with wb.attempt("2.28"):
    st = mylearn.stats
    C_28 = st.covariance_matrix(X_measures)
    verdict("2.28", np.shape(C_28) == (4, 4) and np.allclose(np.diag(C_28), X_measures.var(axis=0)),
            "ta matrice de covariance est 4 × 4, avec les variances des colonnes sur la diagonale.",
            "la matrice de covariance doit être 4 × 4, avec les variances des colonnes (ddof = 0) sur la diagonale.")
    R_28 = st.correlation_matrix(X_measures)
    print(pd.DataFrame(R_28, index=MEASURES, columns=MEASURES).round(2))
    wb.check("2.28a", R_28)
    run_stats_tests("test_covariance_matrix_ or test_correlation_matrix_")
wb.check("2.28b", most_negative.replace(" ", "").split(",") if isinstance(most_negative, str) else most_negative)''',
       solution=r'''st = mylearn.stats
C_28 = st.covariance_matrix(X_measures)
R_28 = st.correlation_matrix(X_measures)
print(pd.DataFrame(C_28, index=MEASURES, columns=MEASURES).round(2))
print(pd.DataFrame(R_28, index=MEASURES, columns=MEASURES).round(3))
j, k = np.unravel_index(np.argmin(R_28), R_28.shape)
most_negative = [MEASURES[j], MEASURES[k]]
print("most negative pair:", most_negative)
run_stats_tests("test_covariance_matrix_ or test_correlation_matrix_", impl="ref")''',
       record=r'''wb.record("2.28a", R_28, decimals=2)
wb.record("2.28b", set(most_negative), mistakes={"c'est la paire la plus liée en valeur absolue ; on demande la plus NÉGATIVE (le coefficient le plus bas)": {"flipper_length_mm", "body_mass_g"}})''',
       after=[("md", "**Et dans chaque espèce ?** En 2.26 d, la corrélation entre la longueur et l'épaisseur du bec est "
                     "**négative** sur l'ensemble des manchots : un bec plus long serait plus fin.\n\n"
                     "c) `prediction_gentoo` : **avant de calculer**, prédis le signe de cette corrélation chez les seuls "
                     "Gentoo : `+1` si elle est positive, `-1` si elle est négative."),
              ("todo", 'prediction_gentoo = ...   # c) +1 (positive) or -1 (negative), BEFORE computing d'),
              ("check", 'wb.check("2.28c", prediction_gentoo)'),
              ("solution", 'prediction_gentoo = +1'),
              ("record", 'wb.record("2.28c", prediction_gentoo, mistakes={"chez des manchots d\'une même espèce, un gros oiseau a-t-il un '
                         'bec plus fin, ou un bec plus grand dans tous les sens ? Le signe global vient peut-être d\'ailleurs": -1})'),
              ("md", "d) Écris `bill_r_by_species()` : la corrélation longueur–épaisseur du bec pour chaque espèce, dans "
                     "l'ordre Adélie, Chinstrap, Gentoo (une liste, 2 décimales), avec ta `correlation` ou ta "
                     "`correlation_matrix`. La vérification trace ensuite le nuage coloré par espèce, avec une droite "
                     "par espèce et une pour l'ensemble."),
              ("todo", r'''def bill_r_by_species():
    """[r Adelie, r Chinstrap, r Gentoo]: correlation of bill length and bill depth within each species."""
    raise NotImplementedError("bill_r_by_species() is not written yet")'''),
              ("check", r'''with wb.attempt("2.28"):
    wb.check("2.28d", bill_r_by_species())'''),
              ("solution", r'''def bill_r_by_species():
    """[r Adelie, r Chinstrap, r Gentoo]: correlation of bill length and bill depth within each species."""
    return [mylearn.stats.correlation(group["bill_length_mm"], group["bill_depth_mm"])
            for _, group in measured.groupby("species")]      # groupby sorts: Adelie, Chinstrap, Gentoo


print(np.round(bill_r_by_species(), 3), "· all the penguins:", round(mylearn.stats.correlation(bill_length, bill_depth), 3))'''),
              ("record", 'wb.record("2.28d", bill_r_by_species(), decimals=2)'),
              ("code", guarded(r'''fig, ax = plt.subplots(figsize=(7.5, 4.8))
for species, group in measured.groupby("species"):
    ax.scatter(group["bill_length_mm"], group["bill_depth_mm"], s=12, label=species)
    slope, intercept = np.polyfit(group["bill_length_mm"], group["bill_depth_mm"], 1)
    ends = np.array([group["bill_length_mm"].min(), group["bill_length_mm"].max()])
    ax.plot(ends, intercept + slope * ends)
slope, intercept = np.polyfit(bill_length, bill_depth, 1)
ends = np.array([bill_length.min(), bill_length.max()])
ax.plot(ends, intercept + slope * ends, "k--", label="all the penguins")
ax.set(xlabel="bill_length_mm", ylabel="bill_depth_mm", title="All the penguins, and each species")
ax.legend()
plt.show()''', ['prediction_gentoo'],
                               "⏳ Ex 2.28 : écris d'abord ta prédiction c), puis relance cette cellule.")),
              ("todo_md", "📝 **Mes réponses** (2.28) : compare le signe de la corrélation sur l'ensemble des "
                          "manchots et dans chaque espèce. Comment l'expliques-tu ?\n\n…"),
              ("solution_md", "**Réponses (2.28)** : dans chaque espèce, la corrélation est **positive** (0,39 chez "
                              "les Adélie, 0,65 chez les Chinstrap, 0,64 chez les Gentoo) : un manchot plus grand a un "
                              "bec plus long **et** plus épais. Mais les espèces diffèrent : les Adélie (151) ont des "
                              "becs courts et épais, les Gentoo (123) des becs longs et fins. Ces deux gros paquets "
                              "occupent deux coins opposés du nuage et tirent la corrélation globale vers le négatif ; "
                              "les Chinstrap, aux becs longs et épais, l'atténuent seulement (sans eux, $r = -0{,}55$). "
                              "C'est le **paradoxe de Simpson** (*Simpson's paradox*) : un lien peut changer de sens "
                              "quand on réunit des groupes différents. L'espèce est ici une **variable de confusion** "
                              "(fiche, ⚠️) ; le réflexe : colorer le nuage par groupe avant de conclure.")],
       note="La paire la plus négative est épaisseur du bec – nageoire ($-0{,}58$) : là encore, c'est l'effet des "
            "espèces (les Gentoo ont de longues nageoires et des becs fins). La matrice de covariance, elle, se lit "
            "mal : ses cases vont de −745 à 641 251, parce que les unités diffèrent (g², mm·g…) ; la matrice de "
            "corrélation les met toutes sur la même échelle. Tu repartiras de la matrice de covariance pour "
            "l'analyse en composantes principales (ch. 12)."),

    Ex("2.29", "🐛", 2, 20, "Le piège de ddof : NumPy, pandas et toi",
       "diagnostiquer trois bugs silencieux dus aux conventions de ddof et de `np.cov`, puis les corriger.",
       "Ex 2.28, Ex 2.7 · fiche §2.8 (🕰️ conventions ddof) · pandas (0A)", thread="Penguins", tracks="C",
       body=r"""Un collègue a écrit les trois fonctions ci-dessous. Elles tournent sans erreur… mais elles sont fausses. Mesure d'abord les dégâts :
a) `ratio_29` : `colleague_correlation(flipper, mass)` divisée par la vraie corrélation, `np.corrcoef(flipper, mass)[0, 1]` (4 décimales). Reconnais-tu ce nombre (indice : $n = 342$) ? Essaie aussi la fonction sur les cinq étudiants de 2.7 (`hours_2_7`, `grades_2_7`) : que vaut le résultat ?
b) `shape_29` : la forme (`.shape`) du résultat de `colleague_cov_matrix(measured[MEASURES])` ;
c) `std_after_29` : l'écart-type avec ddof = 0 (`np.std`, la convention du `StandardScaler` de scikit-learn) de la colonne `body_mass_g` du tableau standardisé par `colleague_standardize(measured[MEASURES])` (4 décimales).

Puis corrige : écris `correlation_fixed(x, y)`, `cov_matrix_fixed(df)` (la matrice 4 × 4 des covariances des colonnes, ddof = 0) et `standardize_fixed(df)` (chaque colonne de moyenne 0 et d'écart-type 1 avec ddof = 0), avec NumPy ou pandas, sans `mylearn`. Dans tes notes, pour chaque bug : quelle convention a été mélangée, et comment l'aurais-tu repéré ?""",
       given=r'''def colleague_correlation(x, y):
    return np.cov(x, y)[0, 1] / (np.std(x) * np.std(y))


def colleague_cov_matrix(df):
    return np.cov(df.to_numpy())


def colleague_standardize(df):
    return (df - df.mean()) / df.std()


hours_2_7 = np.array([1, 2, 3, 4, 5])     # the five students of 2.7: hours of revision
grades_2_7 = np.array([2, 3, 5, 4, 6])    # and grades''',
       todo=r'''ratio_29 = ...       # a)
shape_29 = ...       # b)
std_after_29 = ...   # c)


def correlation_fixed(x, y):
    """Pearson correlation of x and y."""
    raise NotImplementedError("correlation_fixed() is not written yet")


def cov_matrix_fixed(df):
    """Covariance matrix (ddof=0) of the COLUMNS of df: shape (number of columns, number of columns)."""
    raise NotImplementedError("cov_matrix_fixed() is not written yet")


def standardize_fixed(df):
    """Every column minus its mean, divided by its standard deviation with ddof=0."""
    raise NotImplementedError("standardize_fixed() is not written yet")''',
       check=r'''wb.check("2.29a", ratio_29)
wb.check("2.29b", tuple(int(v) for v in re.findall(r"\d+", shape_29)) if isinstance(shape_29, str) else shape_29)
wb.check("2.29c", std_after_29)
with wb.attempt("2.29"):
    verdict("2.29", abs(correlation_fixed(flipper, mass) - np.corrcoef(flipper, mass)[0, 1]) < 1e-12
            and abs(correlation_fixed(hours_2_7, grades_2_7) - np.corrcoef(hours_2_7, grades_2_7)[0, 1]) < 1e-12,
            "correlation_fixed donne la vraie corrélation, aussi pour les cinq étudiants de 2.7.",
            "correlation_fixed doit redonner np.corrcoef, aussi sur les cinq points de 2.7 (une corrélation ne dépasse "
            "jamais 1) : le même ddof en haut et en bas.")
    C_29 = np.asarray(cov_matrix_fixed(measured[MEASURES]), dtype=float)
    verdict("2.29", C_29.shape == (4, 4) and np.allclose(C_29, np.cov(X_measures, rowvar=False, ddof=0)),
            "cov_matrix_fixed : une matrice 4 × 4, celle des colonnes, avec ddof = 0.",
            "cov_matrix_fixed doit renvoyer la matrice 4 × 4 des covariances des COLONNES, avec ddof = 0.")
    Z_29 = np.asarray(standardize_fixed(measured[MEASURES]), dtype=float)
    verdict("2.29", Z_29.shape == (342, 4) and np.allclose(Z_29.mean(axis=0), 0) and np.allclose(Z_29.std(axis=0), 1),
            "standardize_fixed : chaque colonne a une moyenne 0 et un écart-type 1 (ddof = 0).",
            "après standardize_fixed, chaque colonne doit avoir une moyenne 0 et un écart-type 1 avec ddof = 0.")''',
       solution=r'''ratio_29 = colleague_correlation(flipper, mass) / np.corrcoef(flipper, mass)[0, 1]
shape_29 = colleague_cov_matrix(measured[MEASURES]).shape
std_after_29 = float(np.std(colleague_standardize(measured[MEASURES])["body_mass_g"].to_numpy()))
print(ratio_29, 342 / 341, "· 2.7:", colleague_correlation(hours_2_7, grades_2_7), "·", shape_29, "·", std_after_29)


def correlation_fixed(x, y):
    """Pearson correlation of x and y."""
    return float(np.cov(x, y, ddof=0)[0, 1] / (np.std(x) * np.std(y)))      # the same ddof everywhere


def cov_matrix_fixed(df):
    """Covariance matrix (ddof=0) of the COLUMNS of df: shape (number of columns, number of columns)."""
    return np.cov(df.to_numpy(), rowvar=False, ddof=0)                       # rows = samples, columns = variables


def standardize_fixed(df):
    """Every column minus its mean, divided by its standard deviation with ddof=0."""
    return (df - df.mean()) / df.std(ddof=0)                                 # pandas: ddof=1 by default


print(correlation_fixed(flipper, mass), correlation_fixed(hours_2_7, grades_2_7), cov_matrix_fixed(measured[MEASURES]).shape,
      standardize_fixed(measured[MEASURES]).to_numpy().std(axis=0))''',
       record=r'''wb.record("2.29a", ratio_29, decimals=4, mistakes={"divise la corrélation du collègue par la vraie, pas l'inverse": 1 / ratio_29})
wb.record("2.29b", shape_29, mistakes={"c'est la forme attendue ; mesure ce que le code du collègue renvoie VRAIMENT (exécute-le)": (4, 4)})
wb.record("2.29c", std_after_29, decimals=4, mistakes={"mesure avec ddof = 0 (np.std), pas avec la méthode .std() de pandas, qui divise par n − 1": 1.0})''',
       note="Bug 1 : `np.cov` divise par $n - 1$ par défaut, `np.std` par $n$ : la corrélation est multipliée par "
            "$\\frac{n}{n - 1}$, soit 1,0029 pour 342 manchots (invisible) et 1,25 pour 5 points, où elle devient "
            "1,125, impossible pour une corrélation (c'est l'erreur classique de 2.7 f). Bug 2 : `np.cov` prend "
            "chaque **ligne** pour une variable : sans `rowvar=False`, on obtient 342 × 342 covariances entre "
            "manchots. Bug 3 : `.std()` de pandas divise par $n - 1$ ; les colonnes standardisées ont un écart-type "
            "de 0,9985 au sens de NumPy et de scikit-learn, et ne coïncident pas avec un `StandardScaler`. "
            "Réflexe : **toujours écrire `ddof` explicitement**, et vérifier les formes (`.shape`) après chaque "
            "étape."),

    Ex("2.30", "🎨", 2, 25, "Le quartet d'Anscombe",
       "redessiner le quartet d'Anscombe avec ses droites de régression, et voir que ses statistiques coïncident… "
       "pas toutes.",
       "Ex 2.26, Ex 2.12 · fiche §2.9 · livre §2.9 (figures 2.31 et 2.32)", thread="—", tracks="M, C",
       body=r"""Les quatre jeux d'Anscombe sont dans `ANSCOMBE`, un dictionnaire `{"I": (x, y), …}`. Écris :
- `fit_line(x, y)` : la **droite de régression** de $y$ en $x$, sous la forme `(intercept, slope)`. Sa pente est $a = \frac{\mathrm{Cov}(x, y)}{\mathrm{Var}(x)}$ et son ordonnée à l'origine $b = \bar{y} - a\,\bar{x}$ (formules admises ici, démontrées au ch. 9), avec tes fonctions de `mylearn.stats` ou avec NumPy ;
- `draw_quartet()` : la figure du livre (figure 2.31) : quatre panneaux (2 × 2), chaque nuage avec sa droite, les **mêmes axes** pour les quatre (`sharex=True, sharey=True`), et un titre par panneau.

La vérification contrôle que les quatre jeux ont bien, à 0,01 près, la même moyenne et la même variance de $x$ et de $y$, la même corrélation et la même droite ; puis :
a) `medians_y` : la médiane de $y$ dans chacun des quatre jeux (liste, 2 décimales). Toutes les statistiques sont-elles égales ?
b) `max_residuals` : pour chaque jeu, le plus grand écart $|y_i - (b + a\,x_i)|$ entre un point et **sa** droite (liste, 2 décimales) ;
c) `slope_iii_clean` : la pente de la droite de régression du jeu III privé de son point isolé, celui du plus grand écart (3 décimales). Que vaut alors sa corrélation ?""",
       given=ANSCOMBE_DATA,
       todo=r'''def fit_line(x, y):
    """(intercept, slope) of the least-squares line of y on x."""
    raise NotImplementedError("fit_line() is not written yet")


def draw_quartet():
    """Figure 2 x 2: the four sets of ANSCOMBE, each with its line, the same axes everywhere."""
    raise NotImplementedError("draw_quartet() is not written yet")


medians_y = ...         # a)
max_residuals = ...     # b)
slope_iii_clean = ...   # c)''',
       check=r'''with wb.attempt("2.30"):
    table_30 = np.array([[np.mean(x), np.mean(y), np.var(x, ddof=1), np.var(y, ddof=1), np.corrcoef(x, y)[0, 1],
                          *fit_line(x, y)] for x, y in ANSCOMBE.values()])
    x_i, y_i = ANSCOMBE["I"]
    verdict("2.30", np.allclose(fit_line(x_i, y_i), np.polyfit(x_i, y_i, 1)[::-1]),
            "fit_line donne la droite des moindres carrés (comme np.polyfit), dans l'ordre (intercept, slope).",
            "fit_line doit renvoyer (intercept, slope), la droite des moindres carrés : vérifie l'ordre et le ddof "
            "(le même pour la covariance et la variance).")
    verdict("2.30", bool(np.all(np.ptp(table_30, axis=0) < 0.01)),
            "les quatre jeux ont les mêmes moyennes, variances, corrélation et droite, à 0,01 près.",
            "les droites (intercept, slope) des quatre jeux devraient coïncider à 0,01 près : vérifie fit_line.")
    draw_quartet()
    plt.show()
wb.check("2.30a", medians_y)
wb.check("2.30b", max_residuals)
wb.check("2.30c", slope_iii_clean)''',
       solution=r'''def fit_line(x, y):
    """(intercept, slope) of the least-squares line of y on x."""
    slope = mylearn.stats.covariance(x, y) / mylearn.stats.variance(x)
    return mylearn.stats.mean(y) - slope * mylearn.stats.mean(x), slope


def draw_quartet():
    """Figure 2 x 2: the four sets of ANSCOMBE, each with its line, the same axes everywhere."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    ends = np.array([2, 20])
    for ax, (name, (x, y)) in zip(axes.ravel(), ANSCOMBE.items()):
        intercept, slope = fit_line(x, y)
        ax.scatter(x, y, s=25)
        ax.plot(ends, intercept + slope * ends, color="0.3")
        ax.set_title(f"{name}: y = {intercept:.2f} + {slope:.2f} x")
    fig.tight_layout()


def largest_residual(x, y):
    """Index and size of the largest |y_i - line(x_i)|."""
    intercept, slope = fit_line(x, y)
    residuals = np.abs(y - (intercept + slope * x))
    return int(np.argmax(residuals)), float(residuals.max())


medians_y = [float(np.median(y)) for _, y in ANSCOMBE.values()]
max_residuals = [largest_residual(x, y)[1] for x, y in ANSCOMBE.values()]
x_iii, y_iii = ANSCOMBE["III"]
worst = largest_residual(x_iii, y_iii)[0]
slope_iii_clean = fit_line(np.delete(x_iii, worst), np.delete(y_iii, worst))[1]
r_iii_clean = float(np.corrcoef(np.delete(x_iii, worst), np.delete(y_iii, worst))[0, 1])
print(medians_y, np.round(max_residuals, 3), slope_iii_clean, r_iii_clean)
draw_quartet()
plt.show()''',
       record=r'''wb.record("2.30a", medians_y, decimals=2, mistakes={"c'est la moyenne de y, la même dans les quatre jeux : on demande la médiane": [7.5, 7.5, 7.5, 7.5]})
signed_30 = [float(np.max(y - (fit_line(x, y)[0] + fit_line(x, y)[1] * x))) for x, y in ANSCOMBE.values()]
wb.record("2.30b", max_residuals, decimals=2, mistakes={"prends la valeur absolue des écarts : un point peut être loin EN DESSOUS de sa droite": signed_30})
wb.record("2.30c", slope_iii_clean, decimals=3, mistakes={"c'est la pente AVEC le point isolé : retire-le d'abord": fit_line(x_iii, y_iii)[1]})''',
       note="Moyennes, variances, corrélation et droite coïncident, mais pas les médianes de $y$ (de 7,04 à 8,14) : "
            "les statistiques « identiques » d'Anscombe sont celles qu'il a choisies. Dans le jeu III, un seul point "
            "est à 3,24 de la droite ; sans lui, les dix autres sont alignés aux arrondis près ($r = 0{,}999997$), "
            "sur une autre droite, de pente 0,345. Le jeu IV (2.12) est le cas extrême : un seul point fabrique toute la "
            "pente. Un résumé chiffré ne remplace jamais le graphique."),

    Ex("2.31", "🛠️", 2, 30, "Docstring et test pytest pour `zscore`",
       "documenter une fonction au format NumPy et écrire des tests pytest qui attrapent les cas limites d'une "
       "standardisation.",
       "Ex 2.15 · 0A.61, 0A.62 · fiche §2.3.2", thread="—", tracks="C",
       body=r"""Deux gestes de développeur, sur la standardisation de 2.15.

**1. Une docstring.** Écris `flag_outliers(x, threshold=3.0)` : elle renvoie les **indices** (un array d'entiers, dans l'ordre croissant) des valeurs dont le z-score (ddof = 0) dépasse `threshold` en valeur absolue ; calcule les z-scores avec `mylearn.stats.zscore`, et utilise `np.flatnonzero`. Donne-lui une docstring complète au format NumPy (comme en 0A.61) : un résumé d'une ligne, puis les sections `Parameters`, `Returns`, `Raises` (que se passe-t-il si les données sont constantes ?) et `Examples`, avec au moins **deux** exemples `>>>` que doctest vérifie. Pour un exemple, prends des valeurs dont tu peux calculer le résultat à la main : pour `[1, 1, 1, 1, 1, 1, 1, 1, 1, 10]`, calcule d'abord le z-score de 10 (moyenne, écart-type), puis choisis `threshold=2.5`. Pourquoi `threshold=3.0` serait-il un mauvais exemple ?

**2. Des tests.** Écris au moins **trois** tests pytest pour une fonction `zscore(x, ddof=0, axis=None)` qui a le contrat de `mylearn.stats.zscore` :
- une **propriété** : sur des données quelconques, le résultat a une moyenne 0 et un écart-type 1 (`pytest.approx`) ;
- un **cas limite** : des données constantes lèvent une `ValueError` (`pytest.raises`) ;
- l'**axe** : avec `axis=0`, **chaque colonne** d'un tableau 2-D a une moyenne 0 et un écart-type 1.

Range-les dans la liste `my_zscore_tests`. Comme en 0A.62, la vérification lance vraiment pytest : tes tests doivent **passer** sur une version juste (`zscore_ok`) et **échouer** sur chacune des trois versions buggées fournies (lis-les : où est l'erreur ?). Tes tests ne peuvent utiliser que `zscore`, `np`, `math` et `pytest`.""",
       given=ZSCORE_VERSIONS,
       todo=r'''def flag_outliers(x, threshold=3.0):
    """TODO: write the NumPy-style docstring here."""
    raise NotImplementedError("flag_outliers() is not written yet")


# Write your test functions here; they call zscore. For example:
# def test_something():
#     assert ...

my_zscore_tests = ...   # the list of your test functions''',
       check=r'''with wb.attempt("2.31"):
    flagged_31 = flag_outliers([1, 1, 1, 1, 1, 1, 1, 1, 1, 10], threshold=2.5)
    verdict("2.31", list(np.asarray(flagged_31)) == [9], "flag_outliers trouve la valeur isolée (indice 9).",
            "flag_outliers([1, 1, 1, 1, 1, 1, 1, 1, 1, 10], threshold=2.5) doit renvoyer l'array [9].")
    verdict("2.31", error_name(flag_outliers, [4, 4, 4]) == "ValueError",
            "des données constantes lèvent une ValueError.",
            "des données constantes doivent lever une ValueError (celle de zscore).")
    doc_31 = inspect.getdoc(flag_outliers) or ""
    missing_31 = [s for s in ("Parameters", "Returns", "Raises", "Examples")
                  if not re.search(rf"^{s}\n-{{3,}}$", doc_31, flags=re.MULTILINE)]
    verdict("2.31", not missing_31, "les quatre sections de la docstring sont là.", f"sections manquantes : {missing_31}")
    runner_31 = doctest.DocTestRunner(verbose=False)
    for test in doctest.DocTestFinder().find(flag_outliers, "flag_outliers",
                                             globs={"np": np, "flag_outliers": flag_outliers, "mylearn": mylearn}):
        runner_31.run(test)
    failed_31, attempted_31 = runner_31.summarize(verbose=False)
    verdict("2.31", attempted_31 >= 2 and failed_31 == 0, f"doctest : {attempted_31} exemples, tous justes.",
            f"doctest : {failed_31} exemple(s) faux sur {attempted_31} (détail au-dessus), ou moins de 2 exemples.")
if my_zscore_tests is ...:
    print("⏳ Ex 2.31 : tests pas encore écrits.")
else:
    good_31 = wb.run_pytest(my_zscore_tests, subject=zscore_ok, name="zscore")
    if good_31.ok and good_31.passed < len(my_zscore_tests):
        print("❌ Ex 2.31 : pytest n'a pas trouvé tous tes tests : chaque fonction de test doit s'appeler test_…")
    verdict("2.31", good_31.ok and good_31.passed >= 3, f"tes tests passent sur la version juste ({good_31.passed} cas).",
            "tes tests doivent tous passer sur la version juste, et être au moins 3 (voir le détail au-dessus).")
    if good_31.ok:
        for bug in [zscore_bug_1, zscore_bug_2, zscore_bug_3]:
            result = wb.run_pytest(my_zscore_tests, subject=bug, name="zscore", quiet=True)
            verdict("2.31", result.failed + result.errors > 0, f"{bug.__name__} est attrapé ({result.failed} test(s) en échec).",
                    f"tes tests passent sur {bug.__name__}, qui est faux : ajoute un test qui le fait échouer.")''',
       solution=r'''def flag_outliers(x, threshold=3.0):
    """Find the values that are far from the mean, in standard deviations.

    Parameters
    ----------
    x : array-like of shape (n,)
        Numbers (no NaN), not all equal.
    threshold : float, default=3.0
        A value is flagged when the absolute value of its z-score (ddof=0) is
        strictly greater than ``threshold``.

    Returns
    -------
    np.ndarray of int
        The indices of the flagged values, in increasing order (possibly empty).

    Raises
    ------
    ValueError
        If the data are constant (standard deviation 0), empty or contain NaN
        (raised by ``mylearn.stats.zscore``).

    Examples
    --------
    >>> flag_outliers([1, 1, 1, 1, 1, 1, 1, 1, 1, 10], threshold=2.5)
    array([9])
    >>> flag_outliers([1, 2, 3, 4])
    array([], dtype=int64)
    >>> flag_outliers([4, 4, 4])
    Traceback (most recent call last):
        ...
    ValueError: standard deviation is 0 (constant data): the z-score is undefined
    """
    z = mylearn.stats.zscore(x)
    return np.flatnonzero(np.abs(z) > threshold)


def test_mean_0_and_std_1():
    z = zscore(np.array([3.0, -1.0, 4.0, 1.5, 9.0, 2.6]))
    assert z.mean() == pytest.approx(0, abs=1e-12)
    assert z.std() == pytest.approx(1)


def test_constant_data_raises():
    with pytest.raises(ValueError):
        zscore([5.0, 5.0, 5.0])


def test_axis_0_standardizes_every_column():
    X = np.array([[1.0, 100.0], [2.0, 300.0], [4.0, 200.0], [7.0, 900.0]])
    Z = zscore(X, axis=0)
    assert Z.mean(axis=0) == pytest.approx([0, 0], abs=1e-12)
    assert Z.std(axis=0) == pytest.approx([1, 1])


my_zscore_tests = [test_mean_0_and_std_1, test_constant_data_raises, test_axis_0_standardizes_every_column]

runner_31 = doctest.DocTestRunner(verbose=False)
for test in doctest.DocTestFinder().find(flag_outliers, "flag_outliers",
                                         globs={"np": np, "flag_outliers": flag_outliers, "mylearn": mylearn}):
    runner_31.run(test)
print(runner_31.summarize(verbose=False))
good_31 = wb.run_pytest(my_zscore_tests, subject=zscore_ok, name="zscore")
for bug in [zscore_bug_1, zscore_bug_2, zscore_bug_3]:
    result = wb.run_pytest(my_zscore_tests, subject=bug, name="zscore", quiet=True)
    print(bug.__name__, "caught:", result.failed + result.errors > 0, f"({result.failed} failed)")''',
       note="Pour `[1, …, 1, 10]` (neuf fois 1), la moyenne vaut 1,9 et l'écart-type 2,7 : le z-score de 10 vaut "
            "exactement 3. Avec `threshold=3.0`, le test « strictement plus grand » tomberait pile sur la limite, "
            "et le résultat dépendrait d'un arrondi de la dernière décimale : un exemple de documentation doit être "
            "net. Chaque test vise un bug : la propriété attrape la division par la variance (bug 1), le cas "
            "limite attrape la version qui renvoie des zéros au lieu de refuser (bug 2), et le test de l'axe "
            "attrape celle qui ignore `axis` (bug 3), à condition que les colonnes n'aient pas toutes la même "
            "moyenne et le même écart-type."),

    Ex("2.32", "🏆", 3, 60, "Mêmes statistiques, autre dessin : fabrique ton quartet",
       "fabriquer un nuage de forme imposée qui a exactement les statistiques d'Anscombe.",
       "Ex 2.30, Ex 2.15, Ex 2.8 · fiche §2.9 (🕰️ Datasaurus)", thread="synthétique", tracks="M, C",
       body=r"""**Défi.** Ajoute un cinquième jeu au quartet, en forme de **cœur**, avec les statistiques d'Anscombe à 0,01 près :
- moyenne de $x$ : 9 ; moyenne de $y$ : 7,50 ;
- variance de $x$ : 11 et variance de $y$ : 4,125, **avec ddof = 1** (les chiffres des tableaux habituels, 2.12) ;
- corrélation : 0,816 (et donc la même droite, $y \approx 3 + 0{,}5\,x$).

`heart_points(40)` fournit 40 points $(h_x, h_y)$ le long d'un cœur. Écris `make_heart_dataset(n=40)`, qui renvoie `(x, y)`, deux arrays de $n$ nombres obtenus à partir de $h_x$ et $h_y$ par des **opérations affines** seulement (multiplier par un nombre, ajouter un nombre, additionner $h_x$ et $h_y$ multipliés par des nombres), pour que le cœur reste reconnaissable, étiré ou penché.

Une méthode, à suivre pas à pas (sans algèbre linéaire) :
1. standardise $h_x$ (ddof = 0) : $z_x$ a une moyenne 0 et un écart-type 1 ;
2. retire de $h_y$ sa partie « en ligne droite » avec $z_x$ : $e = h_y - \bar{h}_y - c\,z_x$, avec $c = \mathrm{Cov}(h_y, z_x)$ ; alors $\mathrm{Cov}(e, z_x) = 0$ (vérifie-le, à partir de la définition de la covariance ou numériquement), puis standardise $e$ en $z_e$. Ici, le cœur est symétrique : tu trouveras $c \approx 0$, mais l'étape compte pour une forme penchée ;
3. pour la corrélation voulue $r$, pose $z_y = r\,z_x + \sqrt{1 - r^2}\,z_e$ : alors $z_y$ a un écart-type 1 et $\mathrm{Corr}(z_x, z_y) = r$ (le démontrer est un bon exercice, en développant la définition de la covariance) ;
4. remets les centres et les échelles : $x = 9 + s_x\,z_x$ et $y = 7{,}5 + s_y\,z_y$, où $s_x$ et $s_y$ sont les écarts-types voulus… avec le bon ddof : $z_x$ a été standardisé avec ddof = 0, les variances visées le sont avec ddof = 1.

La vérification contrôle les cinq statistiques et que ton nuage est bien une image affine du cœur, puis trace les quatre jeux d'Anscombe et le tien, chacun avec sa droite : cinq dessins, une seule droite. Pour aller plus loin : une autre forme (un cercle, une étoile, tes initiales), ou 400 points au lieu de 40.""",
       given=ANSCOMBE_DATA + "\n\n\n" + HEART,
       todo=r'''def make_heart_dataset(n=40):
    """(x, y): the heart of heart_points(n), transformed (affine operations only) to have Anscombe's statistics."""
    raise NotImplementedError("make_heart_dataset() is not written yet")''',
       check=r'''with wb.attempt("2.32"):
    x_32, y_32 = (np.asarray(v, dtype=float) for v in make_heart_dataset(40))
    targets_32 = {"moyenne de x": (np.mean(x_32), 9.0), "moyenne de y": (np.mean(y_32), 7.5),
                  "variance de x (ddof = 1)": (np.var(x_32, ddof=1), 11.0),
                  "variance de y (ddof = 1)": (np.var(y_32, ddof=1), 4.125),
                  "corrélation": (np.corrcoef(x_32, y_32)[0, 1], 0.816)}
    for name, (value, target) in targets_32.items():
        verdict("2.32", abs(value - target) <= 0.01, f"{name} = {fr(value, 4)} (cible : {fr(target, 3)})",
                f"{name} = {fr(value, 4)}, loin de la cible {fr(target, 3)} (à 0,01 près)")
    hx_32, hy_32 = heart_points(40)
    design_32 = np.column_stack([np.ones(40), hx_32, hy_32])
    fits_32 = [np.linalg.lstsq(design_32, v, rcond=None)[0] for v in (x_32, y_32)]
    affine_32 = all(np.allclose(design_32 @ coef, v, atol=1e-6) for coef, v in zip(fits_32, (x_32, y_32)))
    determinant_32 = fits_32[0][1] * fits_32[1][2] - fits_32[0][2] * fits_32[1][1]
    verdict("2.32", len(x_32) == 40 and affine_32 and abs(determinant_32) > 1e-6,
            "ton nuage est une image affine du cœur : il reste un cœur, étiré ou penché.",
            "ton nuage n'est pas une image affine (non aplatie) du cœur, ou n'a pas 40 points : "
            "n'utilise que des multiplications, des additions et des combinaisons de h_x et h_y.")
    plot_quartet_and(x_32, y_32, "heart")''',
       solution=r'''def make_heart_dataset(n=40):
    """(x, y): the heart of heart_points(n), transformed (affine operations only) to have Anscombe's statistics."""
    hx, hy = heart_points(n)
    zx = (hx - hx.mean()) / hx.std()                        # 1. mean 0, standard deviation 1 (ddof=0)
    e = hy - hy.mean() - np.mean((hy - hy.mean()) * zx) * zx  # 2. remove the straight-line part: Cov(e, zx) = 0
    ze = e / e.std()
    r = 0.816
    zy = r * zx + np.sqrt(1 - r ** 2) * ze                  # 3. standard deviation 1, correlation r with zx
    s_x = np.sqrt(11 * (n - 1) / n)                         # 4. ddof=1 variance 11 <=> ddof=0 std sqrt(11 (n-1)/n)
    s_y = np.sqrt(4.125 * (n - 1) / n)
    return 9 + s_x * zx, 7.5 + s_y * zy


x_32, y_32 = make_heart_dataset(40)
print(x_32.mean(), y_32.mean(), np.var(x_32, ddof=1), np.var(y_32, ddof=1), np.corrcoef(x_32, y_32)[0, 1])
plot_quartet_and(x_32, y_32, "heart")''',
       note="Pourquoi ça marche : $\\mathrm{Var}(z_y) = r^2\\,\\mathrm{Var}(z_x) + (1 - r^2)\\,\\mathrm{Var}(z_e) "
            "+ 2r\\sqrt{1 - r^2}\\,\\mathrm{Cov}(z_x, z_e) = r^2 + 1 - r^2 + 0 = 1$, et $\\mathrm{Cov}(z_x, z_y) = "
            "r\\,\\mathrm{Var}(z_x) + \\sqrt{1 - r^2}\\,\\mathrm{Cov}(z_x, z_e) = r$. Le piège est l'étape 4 : "
            "$x = 9 + \\sqrt{11}\\,z_x$ donnerait une variance (ddof = 1) de $11 \\times \\frac{40}{39} \\approx "
            "11{,}28$. Matejka et Fitzmaurice (2017) vont plus loin : ils déplacent les points un par un en gardant "
            "les statistiques, ce qui permet d'atteindre n'importe quelle forme, pas seulement une image affine "
            "(fiche, 🕰️)."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 2.1 à 2.7 | vérifier tes exercices papier ✏️ | ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {ex.title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if kind == "exercise":
        title = "# 2 · Hasard et statistiques de base — notebook d'exercices"
        how = ("La partie 0 vérifie tes exercices papier. Chaque exercice de code : un énoncé, une cellule à "
               "compléter (les `...` et les `raise NotImplementedError`), puis une cellule de vérification "
               "(`wb.check`, ou les tests de ta librairie `mylearn`). « Exécuter tout » va jusqu'au bout même si rien "
               "n'est rempli : ce qui n'est pas fait affiche ⏳. Les questions « pourquoi ? » qui n'ont pas de "
               "cellule 📝 se notent dans la section « Notes sur le notebook » de ta copie de `06_mes_reponses.md`. "
               "Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch02_stats/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 2`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 2 · Hasard et statistiques de base — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Résumer des données par un centre (moyenne, médiane, mode) et une dispersion (variance, écart-type, "
               "percentiles), et savoir lequel choisir.\n"
               "- Tirer au hasard de façon reproductible, dans une loi discrète, avec ou sans remise.\n"
               "- Mesurer l'incertitude d'une statistique par le bootstrap.\n"
               "- Calculer et interpréter covariance et corrélation, et toujours regarder les données.\n\n"
               "**Rappel express.** $\\bar{x} = \\frac{1}{n}\\sum_i x_i$ ; "
               "$\\mathrm{Var}(x) = \\frac{1}{n - \\mathrm{ddof}}\\sum_i (x_i - \\bar{x})^2$ et $\\sigma = \\sqrt{\\mathrm{Var}(x)}$ ; "
               "$z_i = \\frac{x_i - \\bar{x}}{\\sigma}$ ; loi normale : 68 %, 95 %, 99,7 % des tirages à moins de 1, 2, 3 "
               "écarts-types de la moyenne ; $\\mathbb{E}[X] = \\sum_k x_k\\,p_k$ ; "
               "$\\mathrm{Cov}(x, y) = \\frac{1}{n - \\mathrm{ddof}}\\sum_i (x_i - \\bar{x})(y_i - \\bar{y})$ et "
               "$r = \\frac{\\mathrm{Cov}(x, y)}{\\sigma_x\\,\\sigma_y}$. En NumPy : `rng = np.random.default_rng(seed)`, "
               "`x.var(ddof=...)`, `np.percentile`, `np.cov(x, y)`, `np.corrcoef(x, y)`.")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu choisir entre moyenne et médiane, et calculer variance et écart-type avec le bon diviseur ?\n"
               "2. Sais-tu utiliser la règle 68-95-99,7 et dire quand elle ne s'applique pas ?\n"
               "3. Sais-tu calculer une covariance et une corrélation, et dire ce qu'une corrélation ne prouve pas ?\n\n"
               "**Pour aller plus loin** : *Seeing Theory* (Brown University) et le cours *Statistiques et "
               "probabilités* de Khan Academy, cités dans la fiche. La suite : le ch. 3 (probabilités conditionnelles, "
               "mesures de la qualité d'un classifieur), puis le bootstrap revient au ch. 14 (forêts aléatoires) et "
               "la matrice de covariance au ch. 12 (analyse en composantes principales).")]


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
