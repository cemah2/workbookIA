#!/usr/bin/env python
"""Build the two notebooks of chapter 2 from a single source (used by Claude).

    python tools/chapters/build_ch02.py
    python tools/run_all_notebooks.py chapitres/ch02_stats/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch02_stats/03_notebook.ipynb

Part 0 checks the ✏️ paper exercises (2.1 to 2.7). The notebook exercises 2.13 to
2.32 (parts A to D, mylearn.stats) are added in the second generation session (PARTS).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Paper, badge, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

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
# Notebook parts (2.13 to 2.32, mylearn.stats): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "2.13 à 2.18", "résumer des données : tendances centrales, dispersion, histogramme, lois, 68-95-99,7"),
    ("B", "2.19 à 2.21", "tirer au hasard : roue de la fortune, dépendance, avec ou sans remise"),
    ("C", "2.22 à 2.24", "le bootstrap et son intervalle de confiance"),
    ("D", "2.25 à 2.32", "grande dimension, covariance et corrélation, ddof, Anscombe"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 2.1 à 2.7 | vérifier tes exercices papier ✏️ | ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {ex.title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |" for key, ids, title in NEXT_SESSION]
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
    later = ("" if PARTS else
             "\n\n*Les exercices de code 2.13 à 2.32 (parties A à D, ta librairie `mylearn.stats`) seront ajoutés à la "
             "prochaine session de génération ; relance alors `python tools/start_chapter.py 2` pour obtenir le "
             "notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu choisir entre moyenne et médiane, et calculer variance et écart-type avec le bon diviseur ?\n"
               "2. Sais-tu utiliser la règle 68-95-99,7 et dire quand elle ne s'applique pas ?\n"
               "3. Sais-tu calculer une covariance et une corrélation, et dire ce qu'une corrélation ne prouve pas ?\n\n"
               "**Pour aller plus loin** : *Seeing Theory* (Brown University) et le cours *Statistiques et "
               "probabilités* de Khan Academy, cités dans la fiche. La suite : le ch. 3 (probabilités conditionnelles, "
               "mesures de la qualité d'un classifieur), puis le bootstrap revient au ch. 14 (forêts aléatoires) et "
               "la matrice de covariance au ch. 12 (analyse en composantes principales)." + later)]


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
