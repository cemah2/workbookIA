#!/usr/bin/env python
"""Build the two notebooks of chapter 15 from a single source (used by Claude).

    python tools/chapters/build_ch15.py
    python tools/run_all_notebooks.py chapitres/ch15_sklearn/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch15_sklearn/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper
exercises 15.1 to 15.4 and of the 🧮 estimate 15.5 (02_exercices.md). The reflection (🗣️ 15.6, ⚖️ 15.7,
📄 15.8) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 15.9 to 15.31) come with the second generation session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Paper, badge, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "15"
FOLDER = "chapitres/ch15_sklearn"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md),
# and the defaults of the installed scikit-learn 1.6 where a question asks for one
import inspect
import math
from math import comb

import numpy as np
from scipy.special import expit, softmax
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

# Q6, Q10, Q12, R3: defaults and sizes, read from scikit-learn itself
KMEANS_DEFAULT_K = inspect.signature(KMeans).parameters["n_clusters"].default
RANDOM_DEFAULT_ITER = inspect.signature(RandomizedSearchCV).parameters["n_iter"].default
_, TEST_Q12 = train_test_split(np.arange(125), test_size=0.3, random_state=0)
_, TEST_DEFAULT = train_test_split(np.arange(1000), random_state=0)
LARGEST_FOLD_R3 = max(len(val) for _, val in KFold(5).split(np.arange(1003)))


def fits(n_candidates, k, refit=True):
    """Trainings of a search: each candidate is cross-validated on k folds, plus the final refit."""
    return n_candidates * k + (1 if refit else 0)


# 15.1: the grids of the statement
GRID_151 = {"alpha": [0.01, 0.1, 1, 10], "solver": ["svd", "cholesky", "lsqr"]}
N_CAND_151 = math.prod(len(v) for v in GRID_151.values())
POLY_151 = 5 * 2 * 3                          # degree x interaction_only x alpha
LIST_151 = 3 * 1 + 3 * 2 * 1                  # two dictionaries, each with one kernel

# 15.2: scores of a binary logistic regression, then of a 3-class multinomial one
S_152 = np.array([2.0, -1.0, 0.0, 0.5, -3.2])
P_152 = expit(S_152)


def logit(p):
    return math.log(p / (1 - p))


PRED_152 = [int(s > 0) for s in S_152]        # scikit-learn: class 1 when the score is strictly positive
SOFT_152 = softmax(np.array([2.0, 1.0, -1.0]))

# 15.3: a MinMaxScaler learned on x in [-3, 5] and y in [10, 30], then the line y' = 0.6 x' + 0.25
X_MIN, X_MAX, Y_MIN, Y_MAX = -3.0, 5.0, 10.0, 30.0
A_153, B_153 = 0.6, 0.25


def x_scaled(x):
    return (x - X_MIN) / (X_MAX - X_MIN)


def y_back(y_scaled):
    return Y_MIN + (Y_MAX - Y_MIN) * y_scaled


def line_153(x):
    """The fitted line, in the original units."""
    return y_back(A_153 * x_scaled(x) + B_153)


SLOPE_153 = (line_153(X_MAX) - line_153(X_MIN)) / (X_MAX - X_MIN)

# 15.4: monomials of degree at most d in p variables


def n_poly(p, d, bias=True):
    return comb(p + d, d) - (0 if bias else 1)


SMALLEST_D_154 = next(d for d in range(1, 20) if n_poly(10, d) > 1000)

# 15.5: a 4-minute training, 5 hyperparameters, 5 folds
FITS_155 = 5 ** 5 * 5
DAYS_155 = FITS_155 * 4 / 60 / 24
HALVING_155 = sum(n * r for n, r in zip((81, 27, 9, 3, 1), (1 / 81, 3 / 81, 9 / 81, 27 / 81, 1)))
N_99_155 = math.ceil(math.log(0.01) / math.log(0.95))'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("15.Q1", "Coder soi-même ou utiliser une bibliothèque ?", [
        ("a", TF, "False",
         r'''mistakes={"relis le §15.1 : quel chemin le livre choisit-il pour ce chapitre ?": True}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"une bibliothèque sait-elle quelles données tu vas lui donner ?": "B",
          "peut-on choisir et régler un modèle sans savoir ce qu'il fait ?": "C",
          "les modèles de scikit-learn ont-ils des arguments à régler ? Relis le §15.4.1": "D"}'''),
        ("c", TF, "True",
         r'''mistakes={"relis le §15.2 : sur quelles bibliothèques s'appuie scikit-learn ?": False}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"relis le §15.2 : à quoi scikit-learn n'est-il pas destiné ?": "A",
          "SciPy fournit-il des réseaux de neurones et le calcul sur GPU ?": "C",
          "seaborn sert-il à entraîner des modèles ?": "D"}'''),
    ]),
    Paper("15.Q2", "Importer proprement et lire une dépréciation", [
        ("a", LETTER, '"B"',
         r'''mistakes={"on a importé le module, pas la classe : quel préfixe faut-il écrire ?": "A",
          "le nom sklearn a-t-il été importé, et Ridge est-il rangé directement sous sklearn ?": "C",
          "linear_model est un module : peut-on l'appeler comme une fonction ?": "D"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"un avertissement arrête-t-il le programme ?": "A",
          "pourrait-on déprécier un argument qui n'a jamais existé ?": "C",
          "relis le §15.3 : que t'annonce un avertissement de dépréciation ?": "D"}'''),
        ("c", TF, "False",
         r'''mistakes={"relis le paragraphe « Suivre les dépréciations » du §15.3 : que devient un argument déprécié deux versions plus tard ?": True}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"masquer l'avertissement empêchera-t-il le code de casser à la version suivante ?": "A",
          "rester sur une vieille version, est-ce une solution durable ?": "C",
          "que se passera-t-il le jour où l'argument sera retiré ?": "D"}'''),
        ("e", TF, "False",
         r'''mistakes={"relis le §15.3 : quelle version décrit l'adresse « stable » ?": True}'''),
    ]),
    Paper("15.Q3", "Hyperparamètres au constructeur, attributs appris avec « _ »", [
        ("a", LETTER, '"B"',
         r'''mistakes={"qui fixe la valeur de alpha : toi, ou l'apprentissage ?": "A",
          "une feature est une colonne des données : alpha en est-il une ?": "C",
          "un label est la réponse attendue d'un exemple : alpha en est-il un ?": "D"}'''),
        ("b", TF, "False",
         r'''mistakes={"relis le §15.4.2 : que signale le tiret bas à la fin d'un nom d'attribut ?": True}'''),
        ("c", TF, "True",
         r'''mistakes={"relis le §15.4.1 : que renvoie get_params ?": False}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"relis l'encadré sur clone (§15.9.3) : la copie garde-t-elle ce que l'objet a appris ?": "A",
          "clone reçoit-il des données ?": "C",
          "quel objet renvoie clone, et quelle méthode renvoie un dictionnaire ?": "D"}'''),
        ("e", TF, "False",
         r'''mistakes={"relis le §15.4.1 : le constructeur reçoit-il des données ?": True}'''),
    ]),
    Paper("15.Q4", "Ce que renvoient fit et predict", [
        ("a", LETTER, '"C"',
         r'''mistakes={"relis le §15.4.2 : comment peut-on enchaîner Ridge().fit(X, y).predict(X_new) ?": "A",
          "fit reçoit-il de nouvelles données à prédire ?": "B",
          "quelle méthode calcule le score ?": "D"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"relis le §15.4.2 : combien de dimensions doit avoir X, même avec une seule feature ?": "A",
          "une ligne par exemple ou une ligne par feature ?": "C"}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"predict renvoie-t-il un tableau en colonnes, ou une valeur par exemple ?": "B",
          "combien d'exemples X_new contient-il, et combien de valeurs predict en renvoie-t-il ?": "C",
          "X_new contient combien d'exemples ?": "D"}'''),
        ("d", TF, "False",
         r'''mistakes={"relis l'encadré « des données toujours en deux dimensions » (§15.4.3)": True}'''),
        ("e", TF, "False",
         r'''mistakes={"relis le §15.4.2 : que fait un second appel à fit ?": True}'''),
    ]),
    Paper("15.Q5", "decision_function ou predict_proba ?", [
        ("a", LETTER, '"A"',
         r'''mistakes={"combien de scores un classifieur binaire renvoie-t-il par exemple ? Relis le tableau du §15.4.4": "B",
          "à deux classes, decision_function renvoie-t-il un score par classe ? Relis l'encadré ⚠️ du §15.4.4": "C",
          "une ligne par exemple, ou une ligne par classe ?": "D"}'''),
        ("b", LETTER, '"C"',
         r'''mistakes={"predict_proba donne-t-il une seule probabilité par exemple, même à deux classes ?": "A",
          "combien de colonnes faut-il pour que chaque ligne somme à 1 sur toutes les classes ?": "B",
          "une ligne par exemple, ou une ligne par classe ?": "D"}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"au-delà de deux classes, combien de scores par exemple ?": "A",
          "combien de classes a ce problème ?": "B",
          "une ligne par exemple, ou une ligne par classe ?": "D"}'''),
        ("d", "a number", "float(expit(0.0))",
         r'''decimals=2, mistakes={"un score nul est-il en faveur d'une classe ou à égalité ? Relis l'encadré sur la sigmoïde": 0.0,
          "que vaut e⁰ ?": 1.0}'''),
        ("e", TF, "False",
         r'''mistakes={"relis le §15.4.4 : quels classifieurs n'ont pas de predict_proba ?": True}'''),
        ("f", TF, "True",
         r'''mistakes={"relis le §15.4.4 : comment SVC obtient-il ses probabilités, et que dit la documentation de leur accord avec predict ?": False}'''),
    ]),
    Paper("15.Q6", "KMeans : n_clusters, labels_ et choix de k", [
        ("a", "a whole number", "KMEANS_DEFAULT_K",
         r'''mistakes={"relis le §15.5 : quel est le défaut de n_clusters ?": 2,
          "relis le §15.5 : on demande le défaut de n_clusters de KMeans": 10}'''),
        ("b", TF, "False",
         r'''mistakes={"relis le §15.5 : où KMeans lit-il le nombre de features ?": True}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"un clustering voit-il les vraies classes ?": "A",
          "quel attribut donne les centres ?": "C",
          "quel attribut donne l'inertie ?": "D"}'''),
        ("d", TF, "False",
         r'''mistakes={"relis l'encadré sur l'inertie et la silhouette : quelle valeur de k donne la plus petite inertie possible ?": True}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"KMeans affecte-t-il chaque point à un centre, sans exception ?": "A",
          "à qui donne-t-on le nombre de groupes, dans AgglomerativeClustering(n_clusters=3) ?": "C"}'''),
    ]),
    Paper("15.Q7", "fit, transform, fit_transform, inverse_transform", [
        ("a", LETTER, '"B"',
         r'''mistakes={"fit sur le test : quelle transformation apprendrait-on ? Relis la règle d'or du §15.6": "A",
          "fit_transform sur le test réapprend-il la transformation ? Relis le §15.6": "C",
          "inverse_transform transforme-t-il, ou revient-il en arrière ?": "D"}'''),
        ("b", "a number", "(70 - 20) / (60 - 20)",
         r'''decimals=2, mistakes={"un scaler appris sur l'entraînement ramène-t-il toujours le test dans [0, 1] ?": 1.0,
          "x' = (x − min) / (max − min) : as-tu divisé par l'étendue ?": 50.0,
          "divise par l'étendue max − min, pas par le maximum": 0.83,
          "as-tu retiré le minimum de l'entraînement avant de diviser ?": 1.75,
          "retire le minimum, puis divise par l'étendue max − min": 1.17}'''),
        ("c", "a number", "(10 - 20) / (60 - 20)",
         r'''decimals=2, mistakes={"un scaler appris sur l'entraînement ramène-t-il toujours le test dans [0, 1] ?": 0.0,
          "attention au signe : la valeur est sous le minimum de l'entraînement": 0.25,
          "x' = (x − min) / (max − min) : as-tu divisé par l'étendue ?": -10.0,
          "divise par l'étendue max − min, pas par le maximum": -0.17}'''),
        ("d", "a number", "20 + (60 - 20) * 0.75",
         r'''decimals=1, mistakes={"x = min + (max − min) × x' : as-tu ajouté le minimum ?": 30.0,
          "multiplie par l'étendue, pas par le maximum": 65.0}'''),
        ("e", TF, "False",
         r'''mistakes={"relis l'encadré ⚠️ du §15.6.1 : quels transformateurs n'ont pas d'inverse ?": True}'''),
    ]),
    Paper("15.Q8", "Ce que la PCA fait vraiment aux features", [
        ("a", LETTER, '"B"',
         r'''mistakes={"la PCA garde-t-elle des features d'origine ? Relis l'encadré ⚠️ du §15.7": "A",
          "la PCA est-elle linéaire ?": "C",
          "diviser une feature par son écart-type, est-ce une PCA ou une standardisation ?": "D"}'''),
        ("b", TF, "False",
         r'''mistakes={"relis l'encadré ⚠️ du §15.7 : la PCA supprime-t-elle des features d'origine ?": True}'''),
        ("c", TF, "True",
         r'''mistakes={"relis ce que fait whiten=True dans l'encadré ⚠️ du §15.7": False}'''),
        ("d", TF, "False",
         r'''mistakes={"le blanchiment divise chaque composante par un nombre : change-t-il sa direction ?": True}'''),
        ("e", TF, "False",
         r'''mistakes={"fit d'une PCA se sert-il de y ? Relis la PCA au ch. 12": True}'''),
    ]),
    Paper("15.Q9", "Un ensemble est un estimateur comme les autres", [
        ("a", TF, "True",
         r'''mistakes={"relis le §15.8 : comment s'utilise un ensemble dans scikit-learn ?": False}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"relis l'encadré 🕰️ du §15.8 : que devient base_estimator en 1.6 ?": "A",
          "relis l'encadré 🕰️ du §15.8 : le nom de l'argument du modèle de base": "C",
          "algorithm choisissait la variante de boosting : relis l'encadré 🕰️ du §15.8": "D"}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"relis le code du §15.8 : quel est le modèle par défaut ?": "A",
          "AdaBoost de scikit-learn part-il d'un modèle linéaire par défaut ?": "C",
          "le livre passe un RidgeClassifier, mais quel est le défaut ?": "D"}'''),
        ("d", TF, "False",
         r'''mistakes={"relis l'encadré ⚠️ du §15.8 : n_estimators est-il un nombre exact ?": True}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"relis l'encadré ⚠️ du §15.8 : AdaBoost donne-t-il le même poids à tous ses membres ?": "A",
          "moyenner des probabilités à poids égaux, est-ce le vote d'AdaBoost ? Relis le ch. 14": "C"}'''),
    ]),
    Paper("15.Q10", "Grille ou recherche aléatoire ?", [
        ("a", LETTER, '"B"',
         r'''mistakes={"qui choisit le nombre de clusters : toi ou l'algorithme ? Relis le §15.9": "A"}'''),
        ("b", TF, "False",
         r'''mistakes={"relis le §15.9.2 : la recherche accélère-t-elle chaque entraînement ?": True}'''),
        ("c", "a whole number", "3 * 4 * 2",
         r'''mistakes={"une grille essaie toutes les combinaisons : multiplie les nombres de valeurs, ne les additionne pas": 9}'''),
        ("d", "a whole number", "RANDOM_DEFAULT_ITER",
         r'''mistakes={"relis le §15.9.4 : la valeur par défaut de n_iter": 8,
          "relis le §15.9.4 : le défaut porte sur les combinaisons, pas sur les folds": 5}'''),
        ("e", TF, "True",
         r'''mistakes={"relis l'argument de Bergstra et Bengio (§15.9.2) et la figure : combien de valeurs de h₁ essaie chaque méthode ?": False}'''),
        ("f", TF, "True",
         r'''mistakes={"relis l'encadré sur les distributions (§15.9.4) : où tombent presque tous les tirages d'une uniforme entre 0,001 et 1 000 ?": False}'''),
    ]),
    Paper("15.Q11", "Pipeline : step__param et absence de fuite", [
        ("a", LETTER, '"D"',
         r'''mistakes={"le pipeline a deux étapes : comment la grille saurait-elle de quelle étape tu parles ?": "A",
          "compte les tirets bas entre le nom de l'étape et l'hyperparamètre": "B",
          "le nom de l'étape est celui que tu lui as donné dans le pipeline : relis-le": "C"}'''),
        ("b", TF, "True",
         r'''mistakes={"relis le §15.9.7 : que fait cross_val_score du pipeline à chaque tour ?": False}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"un fit_transform sur le fold de validation : sur quelles données le scaler apprendrait-il ?": "A",
          "le modèle a appris sur des données standardisées : peut-il prédire sur des données brutes ?": "C",
          "un scaler appris sur toutes les lignes, c'est la situation (a) de la figure du §15.9.7": "D"}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"make_pipeline garde-t-il les majuscules du nom de la classe ? Relis le §15.9.5": "A",
          "clf est un nom choisi à la main : quel nom make_pipeline choisit-il seul ?": "C",
          "relis le §15.9.5 : d'après quoi make_pipeline nomme-t-il ses étapes ?": "D"}'''),
        ("e", TF, "False",
         r'''mistakes={"relis le §15.9.5 : quel objet termine le pipeline du livre ?": True}'''),
    ]),
    Paper("15.Q12", "Datasets, train_test_split et documentation", [
        ("a", "a number", "len(TEST_DEFAULT) / 1000",
         r'''decimals=2, mistakes={"relis le §15.11 : la part par défaut du test": 0.2,
          "c'est la part de l'entraînement : on demande celle du test": 0.75,
          "une proportion entre 0 et 1, pas un pourcentage": 25.0,
          "relis la question : on demande la valeur par défaut, quand on ne donne pas test_size": 0.3}'''),
        ("b", TF, "True",
         r'''mistakes={"relis le §15.11 : que montre l'exemple des cercles générés dans l'ordre ?": False}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"ne pas mélanger garde-t-il les proportions des classes ?": "A",
          "random_state rend le découpage reproductible : garde-t-il les proportions ?": "C",
          "changer la taille du test garde-t-il les proportions ?": "D"}'''),
        ("d", TF, "False",
         r'''mistakes={"relis l'encadré 🕰️ du §15.10 : que devient load_boston en 1.2 ?": True}'''),
        ("e", "a whole number", "len(TEST_Q12)",
         r'''mistakes={"arrondis la taille du test au-dessus, comme scikit-learn": 37,
          "relis la question : quelle partie du découpage compte-t-on ?": 87},
          fractional="0,3 × 125 n'est pas un entier : relis la règle d'arrondi de l'énoncé"'''),
        ("f", LETTER, '"B"',
         r'''mistakes={"make_moons fait deux demi-lunes": "A", "make_blobs fait des nuages gaussiens": "C",
          "make_classification fait des problèmes réglables, pas des cercles": "D"}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("15.R1", "Forêt aléatoire ou gradient boosting ?", [
        ("a", LETTER, '"A"',
         r'''mistakes={"chaque arbre d'un gradient boosting corrige-t-il les erreurs des précédents ?": "B"}'''),
        ("b", TF, "False",
         r'''mistakes={"relis le bilan du ch. 14 : que fait une forêt quand on lui ajoute des arbres ?": True}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"le boosting combine des apprenants faibles : que leur manque-t-il, biais ou variance ?": "B"}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"max_features règle le tirage des features des forêts": "A",
          "n_jobs règle le nombre de cœurs": "C", "bootstrap règle le tirage des exemples du bagging": "D"}'''),
        ("e", TF, "False",
         r'''mistakes={"relis le bilan du ch. 14 : quelle méthode donne un score out-of-bag ?": True}'''),
    ]),
    Paper("15.R2", "Où ajuster le scaler dans une validation croisée ?", [
        ("a", TF, "True",
         r'''mistakes={"relis le §12.10 : les folds de validation ont-ils servi au calcul de la moyenne et de l'écart-type ?": False}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"tout le dataset contient le fold de validation : est-ce permis ?": "A",
          "apprendre le scaler sur le fold qu'on va noter, est-ce honnête ?": "C",
          "le test final sert-il pendant la validation croisée ?": "D"}'''),
        ("c", "a number", "(65 - 50) / 10",
         r'''decimals=2, mistakes={"un z-score se calcule en retirant la moyenne : as-tu retiré 50 ?": 6.5,
          "retire la moyenne, puis divise par l'écart-type": 15.0,
          "divise par l'écart-type, pas par la variance": 0.15}'''),
        ("d", TF, "True",
         r'''mistakes={"relis la fin du §15.9.7 : qu'a montré l'expérience du ch. 8 sur du bruit pur ?": False}'''),
        ("e", LETTER, '"A"',
         r'''mistakes={"le logarithme d'une valeur dépend-il des autres lignes ?": "B"}'''),
    ]),
    Paper("15.R3", "Combien de modèles pour une validation croisée à 5 folds ?", [
        ("a", "a whole number", "5",
         r'''mistakes={"relis la question : on ne compte pas de réentraînement final": 6,
          "combien de tours fait une validation croisée à 5 folds ?": 1}'''),
        ("b", "a whole number", "1",
         r'''mistakes={"chaque exemple appartient à un seul fold : combien de fois ce fold sert-il à la validation ?": 5,
          "un exemple sert à la validation dans les tours où son fold est mis de côté : combien en a-t-il ?": 4,
          "chaque exemple appartient-il à un fold ?": 0,
          "combien de fois son fold est-il mis de côté ?": 2}'''),
        ("c", "a whole number", "4",
         r'''mistakes={"un exemple ne sert pas à l'entraînement quand son fold est mis de côté": 5,
          "relis la question : on demande combien de fois il sert à l'entraînement": 1}'''),
        ("d", "a whole number", "LARGEST_FOLD_R3",
         r'''mistakes={"1 003 n'est pas divisible par 5 : que fait KFold du reste ?": 200,
          "c'est la taille de l'entraînement d'un tour, pas d'un fold": 802,
          "KFold met-il tout le reste dans un seul fold ?": 203},
          fractional="un fold a un nombre entier d'exemples : relis comment KFold répartit le reste"'''),
        ("e", "a whole number", "40",
         r'''mistakes={"un leave-one-out met-il de côté chaque exemple, une fois ?": 39,
          "combien de tours fait un leave-one-out ?": 1}'''),
        ("f", TF, "True",
         r'''mistakes={"relis le §15.9.1 : à quoi sert la stratification ?": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("15.1", "Compter les modèles entraînés par une recherche en grille", [
        ("a", "a whole number", "N_CAND_151",
         r'''mistakes={"une grille essaie toutes les combinaisons : multiplie les nombres de valeurs, ne les additionne pas": 7}'''),
        ("b", "a whole number", "fits(N_CAND_151, 5, refit=False)",
         r'''mistakes={"chaque candidat est noté par une validation croisée à 5 folds : combien d'entraînements par candidat ?": 12,
          "relis la question : on demande les entraînements des validations croisées, sans le réentraînement final": 61}'''),
        ("c", "a whole number", "fits(N_CAND_151, 5)",
         r'''mistakes={"as-tu compté le réentraînement final du meilleur réglage (refit=True) ?": 60,
          "un seul réentraînement final, celui du meilleur candidat, pas un par candidat": 72,
          "chaque candidat est noté par une validation croisée à 5 folds": 13}'''),
        ("d", "a whole number", "fits(N_CAND_151, 3)",
         r'''mistakes={"as-tu compté le réentraînement final ?": 36,
          "relis la question : avec combien de folds ?": 61}'''),
        ("e", "a whole number", "fits(POLY_151, 4)",
         r'''mistakes={"as-tu compté le réentraînement final ?": 120,
          "interaction_only est aussi un hyperparamètre de la grille : combien de valeurs a-t-il ?": 61,
          "relis cv : combien de folds ?": 151,
          "on compte les entraînements du pipeline entier, pas de chacune de ses étapes": 242}'''),
        ("f", "a whole number", "fits(LIST_151, 5, refit=False)",
         r'''mistakes={"refit=False : pas de réentraînement final": 46,
          "les combinaisons des deux dictionnaires s'additionnent, elles ne se multiplient pas": 90,
          "chaque dictionnaire est une grille à part : linear n'a pas de gamma": 60,
          "as-tu compté les deux valeurs de gamma dans la seconde grille ?": 30}'''),
        ("g", "a whole number", "fits(20, 5)",
         r'''mistakes={"as-tu compté le réentraînement final ?": 100, "chaque combinaison est-elle notée une seule fois ?": 21}'''),
        ("h", "a whole number", "3 * fits(N_CAND_151, 5)",
         r'''mistakes={"chaque fold extérieur relance une recherche complète, réentraînement final compris": 180,
          "combien de recherches complètes la validation croisée extérieure lance-t-elle ?": 61,
          "cross_val_score n'ajoute pas de réentraînement final après ses 3 folds": 184}'''),
    ]),
    Paper("15.2", "De decision_function à predict_proba", [
        ("a", "a number with 4 decimals", "float(P_152[0])",
         r'''decimals=4, mistakes={"σ(s) = 1 / (1 + e^(−s)) : attention au signe de l'exposant": 0.1192,
          "une probabilité peut-elle dépasser 1 ?": 7.3891}'''),
        ("b", "a number with 4 decimals", "float(P_152[1])",
         r'''decimals=4, mistakes={"σ(s) = 1 / (1 + e^(−s)) : attention au signe de l'exposant": 0.7311,
          "e^s seul n'est pas la sigmoïde : relis sa formule": 0.3679}'''),
        ("c", "a number", "float(P_152[2])",
         r'''decimals=2, mistakes={"que vaut e⁰ ?": 1.0, "un score nul penche-t-il d'un côté ?": 0.0}'''),
        ("d", "a list of 5 classes, e.g. [1, 1, 1, 1, 1]", "PRED_152",
         r'''mistakes={"relis le §15.4.4 : la classe 1 exige-t-elle un score positif ou nul, ou strictement positif ?": [1, 0, 1, 1, 0],
          "le signe du score donne la classe : la classe 1 pour un score positif": [0, 1, 1, 0, 1],
          "le seuil de predict porte sur le score : quel score correspond à la probabilité 0,5 ?": [1, 0, 0, 0, 0]}'''),
        ("e", "a number with 4 decimals", "logit(0.9)",
         r'''decimals=4, mistakes={"le logit est ln(p / (1 − p)), pas ln(p)": -0.1054,
          "le logit utilise le logarithme népérien, pas log₁₀": 0.9542,
          "c'est ln(p / (1 − p)) : as-tu inversé la fraction ?": -2.1972}'''),
        ("f", "a number with 4 decimals", "logit(0.8)",
         r'''decimals=4, mistakes={"le logit est ln(p / (1 − p)), pas ln(p)": -0.2231,
          "le logit utilise le logarithme népérien, pas log₁₀": 0.6021,
          "c'est ln(p / (1 − p)) : as-tu inversé la fraction ?": -1.3863}'''),
        ("g", "a whole number", "int(np.sum(S_152 >= logit(0.8)))",
         r'''mistakes={"compare chaque score au seuil de f), pas à 0": 2, "au moins un score dépasse le seuil de f) : lequel ?": 0}'''),
        ("h", "a number with 4 decimals", "float(SOFT_152[0])",
         r'''decimals=4, mistakes={"le softmax divise e^(s₀) par la somme des e^(sⱼ) de TOUTES les classes": 0.7311,
          "le softmax travaille sur les exponentielles, pas sur les scores eux-mêmes": 1.0,
          "le softmax travaille sur les exponentielles de toutes les classes : as-tu divisé les scores bruts des deux premières ?": 0.6667,
          "as-tu gardé le signe du troisième score (−1) dans son exponentielle ?": 0.5761}'''),
        ("i", TF, "False",
         r'''mistakes={"relis l'encadré sur le softmax : que devient e^(s+5) / Σ e^(sⱼ+5) ?": True}'''),
        ("j", TF, "False",
         r'''mistakes={"relis le §15.4.4 : pourquoi RidgeClassifier n'a-t-il pas de predict_proba ?": True}'''),
    ]),
    Paper("15.3", "Ramener une droite ajustée dans l'espace d'origine", [
        ("a", "a number", "x_scaled(1.0)",
         r'''decimals=3, mistakes={"x' = (x − min) / (max − min) : as-tu retiré le minimum, −3 ?": 0.125,
          "divise par l'étendue max − min, pas par le maximum": 0.8,
          "retirer un minimum négatif, c'est ajouter : attention au signe de −3": -0.25}'''),
        ("b", "a number", "A_153 * x_scaled(1.0) + B_153",
         r'''decimals=3, mistakes={"la droite s'applique à x', pas à x : as-tu d'abord transformé x ?": 0.85,
          "as-tu ajouté l'ordonnée à l'origine 0,25 ?": 0.3}'''),
        ("c", "a number", "line_153(1.0)",
         r'''decimals=2, mistakes={"y = min + (max − min) × y' : as-tu ajouté le minimum de y ?": 11.0,
          "ramène y' avec l'étendue de y, pas celle de x": 14.4,
          "la droite s'applique à x', pas à x : as-tu d'abord transformé x ?": 27.0,
          "relis la question : on demande la prédiction dans l'unité d'origine de y": 0.55}'''),
        ("d", "a number", "line_153(X_MIN)",
         r'''decimals=2, mistakes={"y = min + (max − min) × y' : as-tu ajouté le minimum de y ?": 5.0,
          "en x' = 0, y' vaut l'ordonnée à l'origine de la droite transformée : as-tu pris 0 ?": 10.0,
          "c'est encore une valeur transformée : ramène-la dans l'unité de y": 0.25}'''),
        ("e", "a number", "line_153(X_MAX)",
         r'''decimals=2, mistakes={"y = min + (max − min) × y' : as-tu ajouté le minimum de y ?": 17.0,
          "en x' = 1, y' = 0,6 + 0,25 : as-tu oublié l'ordonnée à l'origine ?": 22.0,
          "c'est encore une valeur transformée : ramène-la dans l'unité de y": 0.85}'''),
        ("f", "a number", "SLOPE_153",
         r'''decimals=3, mistakes={"la pente transformée est-elle dans les unités d'origine ?": 0.6,
          "une pente se mesure en unités de y par unité de x : as-tu inversé le rapport ?": 0.24,
          "as-tu divisé par l'étendue de x ?": 12.0,
          "as-tu oublié la pente 0,6 de la droite transformée ?": 2.5}'''),
        ("g", "a number", "line_153(0.0)",
         r'''decimals=2, mistakes={"x = 0 n'est pas x' = 0 : transforme d'abord x = 0": 15.0,
          "ramène y' dans l'unité d'origine avec y = min + (max − min) × y'": 0.475,
          "c'est l'ordonnée à l'origine de la droite transformée : ramène-la dans l'espace d'origine, en x = 0": 0.25}'''),
        ("h", "a number", "x_scaled(11.0)",
         r'''decimals=3, mistakes={"x' = (x − min) / (max − min) : as-tu retiré le minimum, −3 ?": 1.375,
          "un point hors de l'étendue de l'entraînement sort de [0, 1] : faut-il le ramener à 1 ?": 1.0}'''),
        ("i", "a number", "line_153(11.0)",
         r'''decimals=2, mistakes={"un point hors de l'étendue sort de [0, 1] : faut-il le ramener à 1 avant la droite ?": 27.0,
          "y = min + (max − min) × y' : as-tu ajouté le minimum de y ?": 26.0}'''),
        ("j", TF, "False",
         r'''mistakes={"fais-le : que donne inverse_transform sur le point (0,6 ; 0,25), et est-ce la pente trouvée en f) ?": True}'''),
    ]),
    Paper("15.4", "Combien de features polynomiales ?", [
        ("a", "a whole number", "n_poly(2, 2)",
         r'''mistakes={"as-tu compté la constante 1 (include_bias=True) ?": 5,
          "x·y et y·x sont le même monôme : l'ordre des facteurs ne compte pas": 7,
          "compte aussi les monômes de degré 0 et 1, pas seulement ceux de degré 2": 3}'''),
        ("b", "a whole number", "n_poly(2, 5)",
         r'''mistakes={"as-tu compté la constante 1 ?": 20,
          "l'ordre des facteurs ne compte pas : ce n'est pas un nombre de suites": 63,
          "compte tous les degrés de 0 à 5, pas seulement le degré 5": 6}'''),
        ("c", "a whole number", "comb(2 + 4 - 1, 4)",
         r'''mistakes={"on demande le degré EXACTEMENT 4 : retire les monômes de degré plus petit": 15,
          "énumère-les : x⁴, x³y, …": 4,
          "l'ordre des facteurs ne compte pas : x³y et yx³ sont le même monôme": 16}'''),
        ("d", "a whole number", "n_poly(8, 2)",
         r'''mistakes={"as-tu compté la constante 1 ?": 44, "compte aussi les carrés x₁², …, x₈²": 37,
          "l'ordre des facteurs ne compte pas : x₁x₂ et x₂x₁ sont le même monôme": 73}'''),
        ("e", "a whole number", "n_poly(8, 3)",
         r'''mistakes={"as-tu compté la constante 1 ?": 164,
          "dans C(p + d, d), le nombre du haut est p + d : relis l'encadré": 120}'''),
        ("f", "a whole number", "n_poly(20, 3, bias=False)",
         r'''mistakes={"include_bias=False retire la constante": 1771,
          "dans C(p + d, d), le nombre du haut est p + d : relis l'encadré": 1539}'''),
        ("g", "a whole number", "1 + 5 + comb(5, 2)",
         r'''mistakes={"interaction_only=True retire les carrés comme x₁², mais garde la constante et les x_j": 21,
          "as-tu compté la constante 1 ?": 15, "as-tu gardé les cinq features de degré 1 ?": 11}'''),
        ("h", "a number with 2 decimals", "100_000 * n_poly(20, 3) * 8 / 1e9",
         r'''decimals=2, mistakes={"compte 8 octets par nombre": 0.18,
          "l'énoncé compte en Go de 10⁹ octets, pas de 2³⁰": 1.32,
          "un Go vaut 10⁹ octets : convertis les octets en Go": 1416800000.0}'''),
        ("i", "a whole number", "SMALLEST_D_154",
         r'''mistakes={"as-tu vérifié que C(10 + d, d) dépasse 1 000 pour ta valeur de d ?": 3,
          "teste les degrés un par un, en partant de 1": 5}'''),
    ]),
    # ---------------------------------------------------------------- reflection with numbers
    Paper("15.5", "Budget d'une recherche d'hyperparamètres", [
        ("a", "a whole number", "FITS_155",
         r'''mistakes={"chaque combinaison est notée sur 5 folds": 3125,
          "les nombres de valeurs des 5 hyperparamètres se multiplient entre eux": 125}'''),
        ("b", "a number with 1 decimal", "DAYS_155",
         r'''decimals=1, mistakes={"c'est une durée en minutes : convertis-la en jours": 62500.0,
          "c'est une durée en heures : convertis-la en jours": 1041.7,
          "as-tu compté les 5 folds ?": 8.7,
          "62 500 est en minutes : par quoi as-tu divisé pour obtenir des jours ?": 2604.2}'''),
        ("c", "a number with 1 decimal", "DAYS_155 / 8",
         r'''decimals=1, mistakes={"le parallélisme divise la durée, il ne la multiplie pas": 347.2,
          "combien de cœurs travaillent en même temps ?": 21.7}'''),
        ("d", "a number with 1 decimal", "60 * 5 * 4 / 8 / 60",
         r'''decimals=1, mistakes={"as-tu divisé par les 8 cœurs ?": 20.0, "as-tu compté les 5 folds ?": 0.5,
          "convertis les minutes en heures": 150.0}'''),
        ("e", "a number with 4 decimals", "1 - 0.95 ** 60",
         r'''decimals=4, mistakes={"c'est la probabilité qu'AUCUN essai ne tombe dans les 5 % : on demande le complément": 0.0461,
          "les essais ne s'additionnent pas : 60 × 0,05 dépasse 1": 3.0,
          "quelle est la probabilité qu'un seul essai RATE la bonne zone ?": 1.0}'''),
        ("f", "a whole number", "N_99_155",
         r'''mistakes={"le nombre d'essais est un entier : arrondis au-dessus pour atteindre au moins 0,99": 89,
          "dans le dénominateur, quelle probabilité as-tu mise : celle de réussir ou celle de rater un essai ?": 2,
          "c'est le nombre pour 0,95, pas pour 0,99": 59},
          fractional="un nombre d'essais est entier : faut-il arrondir au plus proche, ou au-dessus pour atteindre 0,99 ?"''',
         ),
        ("g", "a number", "HALVING_155",
         r'''decimals=1, mistakes={"as-tu compté tous les tours ?": 4.0,
          "additionne le coût de TOUS les tours, pas seulement le dernier": 1.0,
          "chaque candidat d'un tour ne reçoit qu'une part des données : en as-tu tenu compte ?": 121.0}'''),
        ("h", "a number with 1 decimal", "81 / HALVING_155",
         r'''decimals=1, mistakes={"on demande le coût de la grille divisé par celui du successive halving, pas l'inverse": 0.1,
          "relis g) : le successive halving coûte-t-il un seul entraînement ?": 81.0,
          "as-tu repris le coût de g) avec ses 5 tours ?": 20.3,
          "relis g) : combien de tours as-tu comptés ?": 20.2}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 15.1 à 15.4 et 🧮 15.5)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "un nom : une chaîne (`\"standardscaler\"`) ; plusieurs nombres : une liste (`[4, 7]`). Arrondis "
               "comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore remplies "
               "affichent ⏳. Les questions « dans ta copie », la réflexion (🗣️ 15.6, ⚖️ 15.7, 📄 15.8) et "
               "l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 15.9 to 15.31): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "15.9 à 15.12", "l'interface d'un estimateur, ses trois sorties, les datasets et le découpage, le code de "
                          "2018 à faire tourner avec la version 1.6"),
    ("B", "15.13 à 15.16", "clustering (KMeans, DBSCAN, hiérarchique), transformateurs et inverse_transform, PCA et "
                           "blanchiment, ensembles sur cinq spirales"),
    ("C", "15.17 à 15.24", "validation croisée, recherches en grille et au hasard, frontières de décision, pipeline "
                           "polynomial, lecture de cv_results_, la fuite du ch. 8 refaite avec un pipeline"),
    ("D", "15.25 à 15.31", "seuil de décision, classes déséquilibrées, six familles de modèles, ColumnTransformer sur "
                           "les manchots bruts, ta librairie face à scikit-learn, sauvegarde, le défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 15.Q1 à 15.Q12, 15.R1 à 15.R3, 15.1 à 15.5 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 | "
            "★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# 15 · scikit-learn — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`). « Exécuter tout » va "
               "jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche ⏳. Les questions « dans "
               "tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes sur le notebook » de ta "
               "copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch15_sklearn/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 15`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 15 · scikit-learn — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Utiliser l'interface commune de scikit-learn : hyperparamètres au constructeur, `fit` qui renvoie "
               "l'objet, attributs appris suffixés par `_`, `predict`, `decision_function`, `predict_proba`, "
               "`transform` et `inverse_transform`.\n"
               "- Évaluer et régler un modèle : `cross_val_score`, `cross_validate`, `KFold`, `StratifiedKFold`, "
               "`GridSearchCV`, `RandomizedSearchCV`, et lire `cv_results_`.\n"
               "- Enchaîner sans fuite : `Pipeline`, `make_pipeline`, `ColumnTransformer`, noms `étape__paramètre`.\n"
               "- Moderniser du code ancien, choisir un seuil de décision, traiter des classes déséquilibrées, "
               "sauvegarder un pipeline avec ses versions.\n\n"
               "**Rappel express.** $\\hat p = \\sigma(s) = 1/(1 + e^{-s})$ à deux classes, le softmax à $K$ "
               "classes ; nombre d'entraînements d'une grille : $\\big(\\prod_h n_h\\big) \\times k + 1$ ; "
               "probabilité qu'au moins un de $n_{\\text{iter}}$ essais au hasard tombe dans le meilleur $q$ : $1 - (1 - q)^{n_{\\text{iter}}}$ ; "
               "nombre de features polynomiales : $\\binom{p + d}{d}$ ; min-max : $x' = (x - m)/(M - m)$. En "
               "scikit-learn : `sklearn.model_selection` (validation croisée, recherches, découpage), "
               "`sklearn.pipeline`, `sklearn.compose`, `sklearn.preprocessing`, `sklearn.inspection`, "
               "`sklearn.calibration` ; `joblib` pour la sauvegarde.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 15.9 à 15.31 (parties A à D) seront ajoutés à la prochaine session de "
             "génération ; relance alors `python tools/start_chapter.py 15` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dire ce que renvoient `fit`, `predict`, `decision_function`, `predict_proba` et "
               "`transform`, et sous quelle forme ?\n"
               "2. Sais-tu construire un pipeline qui se règle par une recherche en grille ou au hasard sans fuite, "
               "et estimer le coût de cette recherche ?\n"
               "3. Sais-tu moderniser un appel déprécié en lisant la documentation de ta version, et sauvegarder "
               "un modèle avec ses versions ?\n\n"
               "**Pour aller plus loin** : les articles de Pedregosa et coll. (2011), de Buitinck et coll. (2013) "
               "et de Bergstra et Bengio (2012), la page « Common pitfalls » de la documentation, cités dans la "
               "fiche. La suite : le checkpoint de la partie III (ML classique), puis les réseaux de neurones "
               "(ch. 16)." + later)]


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
