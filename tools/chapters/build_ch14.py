#!/usr/bin/env python
"""Build the two notebooks of chapter 14 from a single source (used by Claude).

    python tools/chapters/build_ch14.py
    python tools/run_all_notebooks.py chapitres/ch14_ensembles/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch14_ensembles/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q11, of the recalls R1 to R3, of the ✏️ paper
exercises 14.1, 14.2, 14.5, 14.6 and 14.7 and of the numerical checks of the proof ∂ 14.3
(02_exercices.md). The proof ∂ 14.4, the reflection (🗣️ 14.8, ⚖️ 14.9, 📄 14.10) and the interview are
corrected with 05_solutions.md.
Parts A to D: exercises 14.11 to 14.28 (the learner's mylearn.ensemble, scikit-learn, XGBoost, LightGBM).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "14"
FOLDER = "chapitres/ch14_ensembles"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np
from scipy.stats import binom


def gini(weights_per_class):
    p = np.asarray(weights_per_class, dtype=float) / np.sum(weights_per_class)
    return float(1 - np.sum(p ** 2))


def majority_right(M, p):
    """P(more than M/2 of M independent voters are right), each right with probability p."""
    return float(binom.sf(M // 2, M, p))


# Q3: seven voters, their classes and weights
VOTES_Q3 = np.array(list("ABCABAC"))
W_Q3 = np.array([1, 2, 1, 1, 2, 1, -1])
SCORE_Q3 = {c: float(W_Q3[VOTES_Q3 == c].sum()) for c in "ABC"}

# 14.1: five voters (rows) and four samples (columns)
P_141 = np.array([[0, 1, 2, 0], [1, 1, 0, 2], [1, 0, 2, 2], [0, 1, 1, 0], [2, 0, 0, 1]])
W_141 = np.array([2, 1, 1.5, -1, 0.5])


def scores_141(weights):
    """(4 samples, 3 classes) table of weighted votes."""
    return np.array([[weights[P_141[:, j] == k].sum() for k in range(3)] for j in range(4)])


def winners_141(weights):
    return [int(k) for k in np.argmax(scores_141(weights), axis=1)]      # argmax: the smallest of tied classes


# 14.2: three weak learners A (x1 >= 1), B (x2 >= 1), C (x1 + x2 <= 3)
def score_142(x1, x2, wA=2, wB=2, wC=-3):
    return wA * (x1 >= 1) + wB * (x2 >= 1) + wC * (x1 + x2 <= 3)


REGIONS_142 = {"ABC": (1.5, 1.2), "AB": (2.5, 2.5), "AC": (2, 0.5), "A": (3, 0.5), "BC": (0.5, 1.5),
               "B": (0.5, 2.8), "C": (0.5, 0.5)}                      # one point inside each region
POINTS_142 = [(3, 0.5), (0.5, 2.8), (2, 0.5), (0.2, 1.5), (2.5, 2.5)]
TRUE_142 = [1, 1, 0, 0, 1]

# 14.6: AdaBoost by hand, eight points, stumps "x <= s -> g, else the other class"
X_146 = np.arange(1, 9)
Y_146 = np.array([0, 1, 0, 0, 1, 1, 1, 1])


def stump_146(s, g):
    return np.where(X_146 <= s, g, 1 - g)


def best_stump_146(w):
    candidates = [(float(w[stump_146(s, g) != Y_146].sum()), s, g) for s in np.arange(1.5, 8.5) for g in (0, 1)]
    return min(candidates)                                             # unique minimum here


def adaboost_146(rounds, stumps=None):
    """Errors, alphas, stumps and the weights before each round (learning_rate = 1, two classes)."""
    w = np.full(8, 1 / 8)
    out = []
    for m in range(rounds):
        if stumps and m < len(stumps):
            s, g = stumps[m]
            err = float(w[stump_146(s, g) != Y_146].sum())
        else:
            err, s, g = best_stump_146(w)
        alpha = math.log((1 - err) / err)
        out.append({"err": err, "alpha": alpha, "s": float(s), "g": int(g), "w": w.copy()})
        w = np.where(stump_146(s, g) != Y_146, w * math.exp(alpha), w)
        w = w / w.sum()
    return out, w


def vote_146(rounds):
    votes = np.zeros((8, 2))
    for r in rounds:
        votes[np.arange(8), stump_146(r["s"], r["g"])] += r["alpha"]
    return [int(k) for k in np.argmax(votes, axis=1)]


R_146, W_AFTER_1 = adaboost_146(1)
R2_146, _ = adaboost_146(2)
R3_146, _ = adaboost_146(3, stumps=[(R2_146[0]["s"], R2_146[0]["g"]), (R2_146[1]["s"], R2_146[1]["g"]), (2.5, 1)])

# 14.7: gradient boosting by hand, regression stumps, learning rate 0.5
X_147 = np.arange(1.0, 6.0)
Y_147 = np.array([1.0, 2.0, 6.0, 7.0, 9.0])


def stump_147(r):
    """Best regression stump on the residuals r: (threshold, left value, right value)."""
    best = None
    for s in (1.5, 2.5, 3.5, 4.5):
        left, right = r[X_147 <= s], r[X_147 > s]
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, s, float(left.mean()), float(right.mean()))
    return best[1:]


def boost_147(eta, stages):
    F = np.full(5, Y_147.mean())
    trees = []
    for _ in range(stages):
        s, vl, vr = stump_147(Y_147 - F)
        trees.append((s, vl, vr))
        F = F + eta * np.where(X_147 <= s, vl, vr)
    return F, trees


def predict_147(x, eta, trees):
    return float(Y_147.mean() + eta * sum(vl if x <= s else vr for s, vl, vr in trees))


F1_147, T1_147 = boost_147(0.5, 1)
F2_147, T2_147 = boost_147(0.5, 2)'''

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("14.Q1", "Pourquoi un groupe peut battre chacun de ses membres", [
        ("a", "True or False", "majority_right(15, 0.7) > 0.7",
         r'''mistakes={"relis l'encadré sur la loi binomiale : que devient la majorité de votants indépendants justes à plus de 50 % ?": False}'''),
        ("b", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"des membres identiques font-ils jamais des erreurs différentes ?": "A",
          "la profondeur des arbres crée-t-elle des membres différents ?": "C",
          "qu'apporte un nombre pair de votants, à deux classes ?": "D"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"dix votes toujours identiques peuvent-ils être en désaccord avec l'arbre seul ?": True}'''),
        ("d", "True or False", "True",
         r'''mistakes={"quand une copie tombe en panne, combien de copies justes restent, et que décide la majorité ?": False}'''),
    ]),
    Paper("14.Q2", "Un ensemble corrige-t-il un biais des données ?", [
        ("a", "True or False", "False",
         r'''mistakes={"que deviennent les proportions de ce groupe dans un bootstrap des mêmes données ?": True}'''),
        ("b", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"un bootstrap peut-il contenir un groupe absent des données de départ ?": "A",
          "entraîner plus longtemps change-t-il ce que contiennent les données ?": "C",
          "tirer des features change-t-il les exemples sur lesquels on apprend ?": "D"}'''),
        ("c", "True or False", "True",
         r'''mistakes={"relis le paragraphe sur les ensembles hors du ML (§14.2) : que fait-on varier en prévision d'ensemble ?": False}'''),
        ("d", "True or False", "True",
         r'''mistakes={"une erreur que tous les membres font varie-t-elle d'un membre à l'autre ?": False}'''),
    ]),
    Paper("14.Q3", "Pluralité, majorité et vote pondéré", [
        ("a", 'a class: "A", "B" or "C"', '"A"',
         r'''mistakes={"compte de nouveau les voix de chaque classe, sans les poids": "B",
          "compte les voix de chaque classe : C en a-t-elle le plus ?": "C"}'''),
        ("b", "True or False", "False",
         r'''mistakes={"plus de la moitié de 7 voix, c'est au moins combien de voix ?": True}'''),
        ("c", "a number", "int(SCORE_Q3['C'])",
         r'''mistakes={"le votant 6 a un poids négatif : as-tu retiré son poids ?": 2,
          "additionne les poids de tous les votants de C, signe compris": 1}'''),
        ("d", 'a class: "A", "B" or "C"', '"B"',
         r'''mistakes={"avec les poids, recalcule le total de chaque classe": "A",
          "recalcule le total de chaque classe, poids et signes compris": "C"}'''),
    ]),
    Paper("14.Q4", "Tirer avec remise", [
        ("a", "True or False", "True",
         r'''mistakes={"avec remise, un exemple tiré revient dans l'urne : peut-il ressortir ?": False}'''),
        ("b", "a whole number", "1",
         r'''mistakes={"cherche le cas extrême : que se passe-t-il si les quatre tirages donnent le même exemple ?": 4,
          "les tirages identiques comptent pour un seul exemple distinct": 2}'''),
        ("c", "a whole number", "4",
         r'''mistakes={"on demande le plus PETIT nombre d'exemples jamais tirés : combien quatre tirages peuvent-ils en toucher au mieux ?": 7,
          "8 exemples, au plus combien de tirés : combien en reste-t-il dehors ?": 0}'''),
        ("d", "True or False", "True",
         r'''mistakes={"relis l'encadré sur la limite de (1 − 1/n)^n : quelle part des exemples n'est jamais tirée ?": False}'''),
        ("e", "True or False", "True",
         r'''mistakes={"sans remise, un exemple peut-il sortir deux fois, ou rester dehors avec n tirages parmi n ?": False}'''),
    ]),
    Paper("14.Q5", "Ce que le bagging réduit", [
        ("a", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"une moyenne de modèles biaisés de la même façon garde-t-elle ce biais ?": "A",
          "le bagging change-t-il les labels ?": "C",
          "prédire avec cent arbres est-il plus rapide qu'avec un seul ?": "D"}'''),
        ("b", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"une régression linéaire change-t-elle beaucoup d'un bootstrap à l'autre ?": "A",
          "une souche change-t-elle beaucoup d'un bootstrap à l'autre ?": "C",
          "un kNN avec k = n prédit-il autre chose que la classe majoritaire ?": "D"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"un arbre voit-il les exemples que son bootstrap n'a pas tirés ?": True}'''),
        ("d", "True or False", "True",
         r'''mistakes={"avec quels arbres prédit-on un exemple pour son score out-of-bag ?": False}'''),
    ]),
    Paper("14.Q6", "Le hasard d'une forêt aléatoire", [
        ("a", "a whole number", "int(math.sqrt(16))",
         r'''mistakes={"\"sqrt\" : la racine carrée du nombre de features, pas sa moitié": 8, "relis ce que veut dire \"sqrt\"": 16}''',
         ),
        ("b", 'the letter of your choice, e.g. "E"', '"C"',
         r'''mistakes={"relis le pseudo-code du §14.5 : à quelle ligne a lieu le tirage ?": "A",
          "un arbre tire-t-il ses features une seule fois ? Relis le pseudo-code": "B",
          "le tirage porte-t-il sur les exemples ou sur les features d'un nœud ?": "D"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"examiner deux fois la même feature à un nœud aurait-il un sens ?": True}'''),
        ("d", 'two letters, e.g. "EF"', '"AC"',
         r'''mistakes={"un seuil tiré au hasard : est-ce la forêt ou ExtraTrees ? Relis le §14.6": "BC",
          "l'ordre des exemples est-il une source de hasard voulue de la forêt ?": "CD",
          "relis le pseudo-code du §14.5 : que tire-t-on à chaque nœud ?": "AB"}'''),
        ("e", "True or False", "True",
         r'''mistakes={"sans tirage de features, que reste-t-il d'aléatoire dans la forêt ?": False}'''),
    ]),
    Paper("14.Q7", "ExtraTrees : encore plus de hasard", [
        ("a", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"tester tous les seuils, c'est ce que fait une forêt aléatoire": "A",
          "la médiane dépend-elle du hasard ? Relis ce que tire ExtraTrees (§14.6)": "C",
          "les arbres d'ExtraTrees dépendent-ils les uns des autres ?": "D"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"relis la valeur par défaut de bootstrap pour ExtraTrees dans scikit-learn": False}'''),
        ("c", "True or False", "True",
         r'''mistakes={"au même nœud, un seuil tiré peut-il faire baisser l'impureté plus que le meilleur seuil ?": False}'''),
        ("d", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"ExtraTrees examine-t-il moins de features qu'une forêt, à max_features égal ?": "A",
          "un nœud d'ExtraTrees a-t-il des enfants ?": "C",
          "comment choisirait-il sa feature sans calculer d'impureté ?": "D"}'''),
    ]),
    Paper("14.Q8", "Apprenant faible, souche et apprenant fort", [
        ("a", "a whole number", "2",
         r'''mistakes={"une souche pose UNE question : combien de feuilles ?": 1, "une seule question, pas deux": 4}'''),
        ("b", "a whole number", "1",
         r'''mistakes={"combien d'arêtes entre la racine et une feuille d'une souche ?": 0, "une souche a une seule question": 2}'''),
        ("c", "True or False", "True",
         r'''mistakes={"à deux classes, que devient chaque erreur quand on échange les deux réponses ?": False}'''),
        ("d", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"90 %, c'est un apprenant fort": "A",
          "exactement 50 %, c'est le hasard : est-ce « mieux que le hasard » ?": "C",
          "en dessous de 50 %, à deux classes, l'apprenant est pire que le hasard": "D"}'''),
        ("e", "True or False", "False",
         r'''mistakes={"relis la dernière phrase du paragraphe sur les apprenants faibles et forts (§14.7)": True}'''),
    ]),
    Paper("14.Q9", "Ce que repondère AdaBoost", [
        ("a", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"les mal classés sont multipliés par e^α avec α > 0 : plus grand ou plus petit que 1 ?": "A",
          "relis la mise à jour des poids : les mal classés sont-ils modifiés ?": "C"}'''),
        ("b", "a number", "math.log(0.5 / 0.5) + math.log(2 - 1)",
         r'''decimals=3, mistakes={"que vaut ln 1 ?": 1, "à deux classes, que vaut ln(K − 1) ?": 0.693}'''),
        ("c", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"regarde comment varie (1 − ε)/ε quand ε diminue": "A"}'''),
        ("d", "True or False", "False",
         r'''mistakes={"de quoi le membre m a-t-il besoin avant d'être entraîné ?": True}'''),
        ("e", "a number", "1 - 1 / 4",
         r'''decimals=2, mistakes={"1/K est la probabilité d'avoir raison au hasard ; quelle erreur fait le hasard ?": 0.25, "avec K = 4 classes, le hasard ne se trompe pas dans 50 % des cas": 0.5,
          "as-tu pris le bon nombre de classes ?": 0.67}'''),
    ]),
    Paper("14.Q10", "Bagging ou boosting ?", [
        ("a", 'six letters B or O, e.g. "BBBBBB"', '"BOOBOB"',
         r'''mistakes={"relis la ligne « ce qui baisse » du tableau Bilan pour les phrases 3 et 6": "BOBBOO",
          "relis la ligne « bruit des labels » du tableau Bilan pour la phrase 5": "BOOBBB",
          "pour chaque phrase, demande-toi si les membres sont entraînés indépendamment ou l'un après l'autre": "OBBOBO"}'''),
        ("b", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"une souche qui underfit souffre-t-elle de biais ou de variance, et que réduit le bagging ?": "A"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"relis le §14.4 : que se passe-t-il quand on ajoute des arbres à une forêt ?": True}'''),
    ]),
    Paper("14.Q11", "Gradient boosting : corriger les résidus", [
        ("a", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"0 est-il la meilleure constante pour la MSE ?": "A",
          "la médiane minimise la somme des valeurs absolues ; et la MSE ?": "C",
          "la cible d'un seul exemple résume-t-elle les autres ?": "D"}'''),
        ("b", "a number", "7 - 5",
         r'''mistakes={"le résidu est la cible moins la prédiction, pas l'inverse": -2}'''),
        ("c", "a number", "5 + 0.1 * 2",
         r'''decimals=1, mistakes={"multiplie la correction par le learning rate": 7, "on part de la prédiction actuelle, pas de 0": 0.2,
          "on ajoute 0,1 × le résidu, on ne le soustrait pas": 4.8}'''),
        ("d", "True or False", "True",
         r'''mistakes={"si chaque arbre ne corrige qu'un dixième de l'erreur, en faut-il plus ou moins ?": False}'''),
        ("e", "True or False", "True",
         r'''mistakes={"relis la fiche : que fait un gradient boosting auquel on ajoute des arbres sans contrôle ?": False}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("14.R1", "L'impureté de Gini d'un nœud", [
        ("a", "a number", "gini([6, 3, 1])",
         r'''decimals=2, mistakes={"c'est la somme des carrés ; Gini, c'est 1 moins cette somme": 0.46}'''),
        ("b", "a number with 3 decimals", "gini([0, 3, 1])",
         r'''decimals=3, mistakes={"c'est la somme des carrés ; Gini, c'est 1 moins cette somme": 0.625,
          "les proportions se calculent sur les 4 exemples de l'enfant, pas sur les 10 du parent": 0.9}'''),
        ("c", "a number", "gini([6, 3, 1]) - 0.4 * gini([0, 3, 1])",
         r'''decimals=3, mistakes={"pondère l'impureté de l'enfant droit par sa part des exemples (4 sur 10)": 0.165,
          "le gain, c'est l'impureté du parent MOINS la moyenne pondérée des enfants": 0.15,
          "l'enfant droit contient 4 exemples sur 10, pas 6": 0.315}'''),
        ("d", "a number with 3 decimals", "gini([6, 3, 4])",
         r'''decimals=3, mistakes={"as-tu divisé par la somme des poids (13), et pas par le nombre d'exemples ?": 0.39,
          "as-tu tenu compte du poids 4 de l'exemple de la classe 2 ?": 0.54,
          "c'est la somme des carrés ; Gini, c'est 1 moins cette somme": 0.361}'''),
        ("e", "True or False", "True",
         r'''mistakes={"les proportions changent-elles si tous les poids sont multipliés par 2 ?": False}'''),
    ]),
    Paper("14.R2", "Induction et généralisation", [
        ("a", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"la conclusion d'un modèle sur un nouveau cas est-elle certaine ?": "A",
          "relis les deux façons de raisonner du ch. 11 : laquelle tire une règle d'observations ?": "C"}'''),
        ("b", "True or False", "False",
         r'''mistakes={"un bootstrap peut-il contenir ce que l'échantillon ne contient pas ?": True}'''),
        ("c", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"personne n'invoque ici un expert": "A",
          "l'équipe refuse-t-elle une conclusion qu'imposent les données ?": "C",
          "l'équipe écarte-t-elle des résultats gênants ?": "D"}'''),
        ("d", "True or False", "False",
         r'''mistakes={"un ensemble est-il entraîné sur autre chose que des données passées ?": True}'''),
    ]),
    Paper("14.R3", "Un-contre-tous et un-contre-un", [
        ("a", "a whole number", "5",
         r'''mistakes={"un-contre-tous oppose chaque classe au reste : combien de classifieurs pour 5 classes ?": 10, "chaque classe a-t-elle son classifieur ?": 4}'''),
        ("b", "a whole number", "math.comb(5, 2)",
         r'''mistakes={"un-contre-un : un classifieur par paire de classes, sans compter deux fois la même paire": 20,
          "un-contre-un : un classifieur par paire de classes": 5, "compte les paires de classes distinctes, pas les couples ordonnés ni les classes seules": 25}'''),
        ("c", "a whole number", "math.comb(10, 2)",
         r'''mistakes={"un-contre-un : un classifieur par paire de classes, sans compter deux fois la même paire": 90,
          "un-contre-un : un classifieur par paire de classes": 10}'''),
        ("d", "True or False", "True",
         r'''mistakes={"le classifieur « classe 1 contre classe 3 » a-t-il besoin des exemples des autres classes ?": False}'''),
        ("e", "a number", "1 / 4",
         r'''decimals=2, mistakes={"les positifs sont les exemples d'une seule classe : quelle part du total ?": 0.75, "avec 4 classes, les positifs ne sont pas la moitié des exemples": 0.5}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("14.1", "Dépouiller un vote pondéré", [
        ("a", "a number", "float(scores_141(W_141)[0, 1])",
         r'''decimals=1, mistakes={"additionne les poids de tous les votants qui prédisent la classe 1 pour l'exemple 0": 1.5,
          "c'est le nombre de voix ; on demande la somme de leurs poids": 2}'''),
        ("b", "a class: 0, 1 or 2", "winners_141(W_141)[0]",
         r'''mistakes={"le votant 3 a un poids négatif : as-tu retiré son poids à la classe 0 ?": 0,
          "recalcule le score de la classe 2 : un seul votant la prédit": 2}'''),
        ("c", "a class: 0, 1 or 2", "winners_141(W_141)[1]",
         r'''mistakes={"le poids du votant 3 est négatif, et une égalité va à la plus petite classe : recalcule les deux scores": 1,
          "recalcule le score de la classe 2 pour l'exemple 1": 2}'''),
        ("d", "a list of four classes", "winners_141(W_141)",
         r'''mistakes={"c'est le vote sans les poids ; ajoute le poids de chaque votant": [0, 1, 0, 0],
          "le votant 3 a un poids négatif : as-tu retiré son poids aux classes qu'il prédit ?": [0, 1, 2, 0],
          "une égalité donne la plus petite classe : vérifie l'exemple 1": [1, 1, 2, 2]}'''),
        ("e", "a list of four classes", "winners_141(np.ones(5))",
         r'''mistakes={"sans les poids, chaque votant a une voix ; en cas d'égalité, la plus petite classe gagne": [1, 1, 2, 2],
          "as-tu gardé les poids ? Donne une voix à chaque votant": [1, 0, 2, 2]}'''),
        ("f", "a whole number", "0",
         r'''mistakes={"avec ce poids, les deux classes sont à égalité : qui gagne une égalité ?": -1,
          "on demande la PLUS PETITE valeur entière : essaie une valeur plus petite": 1}'''),
        ("g", "True or False", "True",
         r'''mistakes={"multiplier tous les scores par 3 change-t-il leur ordre ?": False}'''),
        ("h", "True or False", "False",
         r'''mistakes={"ajouter 10 à chaque voix ajoute à chaque classe 10 fois son NOMBRE de voix : refais l'exemple 1": True}'''),
    ]),
    Paper("14.2", "Additionner les voix de trois droites, région par région", [
        ("a", "a number", "score_142(*REGIONS_142['C'])",
         r'''mistakes={"la région « C » ne reçoit que le poids de C, signe compris": 3, "la région « C » est du côté positif de C : elle reçoit son poids": 0}'''),
        ("b", "a number", "score_142(1.5, 1.2)",
         r'''mistakes={"vérifie les trois conditions pour ce point, dont x₁ + x₂ ≤ 3": 4, "le point est-il du côté positif de A et de B ?": -3}'''),
        ("c", "a class: 0 or 1", "int(score_142(0.5, 0.5) > 0)",
         r'''mistakes={"le point est classé 1 seulement si son score est strictement positif : calcule son score": 1,
          "réponds par la classe, 0 ou 1": -3}'''),
        ("d", "a whole number", "sum(score_142(*p) > 0 for p in REGIONS_142.values())",
         r'''mistakes={"recalcule le score de chaque région, et compte celles de score strictement positif": 3,
          "recalcule le score de chaque région, signe compris": 5}'''),
        ("e", "a list of five classes", "[int(score_142(*p) > 0) for p in POINTS_142]",
         r'''mistakes={"pour P₂, calcule x₁ + x₂ avant de conclure sur C": [1, 0, 0, 0, 1]}'''),
        ("f", "a whole number", "sum(score_142(*p, wC=-1) > 0 for p in REGIONS_142.values())",
         r'''mistakes={"recalcule les sept scores avec w_C = −1": 4, "recalcule les sept scores avec w_C = −1, signe compris": 7}'''),
        ("g", "a number", "float(np.mean([int(p[0] >= 1) == t for p, t in zip(POINTS_142, TRUE_142)]))",
         r'''decimals=1, mistakes={"A prédit 1 seulement de son côté positif (x₁ ≥ 1) : compare point par point": 0.4,
          "compare les prédictions de A seul aux vraies classes, point par point": 1}'''),
        ("h", "a number", "(2 + 2 - 3) / 2",
         r'''decimals=1, mistakes={"remplace chaque indicatrice par (s + 1)/2 : as-tu gardé le facteur 1/2 ?": 1,
          "la constante dépend du signe de chaque poids : as-tu pris w_C avec son signe ?": 3.5}'''),
    ]),
    Paper("14.3", "Probabilité d'être oublié par un bootstrap : vers 36,8 %", [
        ("a", "a number with 3 decimals", "(7 / 8) ** 4",
         r'''decimals=3, mistakes={"c'est la probabilité d'être tiré à chaque tirage ; on demande de ne jamais l'être": 0.0002,
          "les probabilités des tirages indépendants se multiplient, elles ne s'additionnent pas": 0.5,
          "c'est la probabilité d'être tiré au moins une fois": 0.414,
          "as-tu échangé n et d ? Il y a 4 tirages parmi 8 exemples": 0.1}'''),
        ("b", "a number with 2 decimals", "8 * (1 - (7 / 8) ** 4)",
         r'''decimals=2, mistakes={"c'est le nombre de tirages ; des tirages peuvent tomber sur le même exemple": 4,
          "c'est le nombre moyen d'exemples JAMAIS tirés": 4.69}'''),
        ("c", "a number with 4 decimals", "0.9 ** 10",
         r'''decimals=4, mistakes={"c'est la probabilité d'être tiré au moins une fois": 0.6513,
          "calcule 0,9 puissance 10 lui-même : n = 10 est loin de la limite": 0.3679}'''),
        ("d", "a number with 4 decimals", "0.999 ** 1000",
         r'''decimals=4, mistakes={"c'est la probabilité d'être tiré au moins une fois": 0.6323, "calcule 0,999 puissance 1 000 lui-même, sans passer à la limite": 0.3679}'''),
        ("e", "a number with 4 decimals", "math.exp(-1)",
         r'''decimals=4, mistakes={"relis ce que demande e) : l'exposant est −1": 2.7183, "relis ce que demande e) : une seule exponentielle, sans complément": 0.6321}'''),
        ("f", "a number with 4 decimals", "100 * math.log(1 - 1 / 100)",
         r'''decimals=4, mistakes={"le logarithme est népérien, pas décimal": -0.4365, "c'est l'approximation −1 ; calcule la valeur exacte": -1}'''),
        ("g", "a number with 1 decimal", "1000 * (1 - 0.999 ** 1000)",
         r'''decimals=1, mistakes={"c'est le nombre moyen d'exemples JAMAIS tirés": 367.7, "c'est l'arrondi avec la limite e⁻¹ ; calcule avec (1 − 1/1 000)^1 000": 632.1}'''),
        ("h", "a number with 3 decimals", "1 - (1 - math.exp(-1)) ** 10",
         r'''decimals=3, mistakes={"c'est la probabilité d'être out-of-bag pour TOUS les arbres": 0.0,
          "c'est la probabilité d'être dans les dix bootstraps": 0.010,
          "c'est la probabilité pour un seul arbre": 0.368}'''),
    ]),
    Paper("14.5", "Quand la majorité de votants indépendants se trompe", [
        ("a", "a number with 3 decimals", "majority_right(3, 0.7)",
         r'''decimals=3, mistakes={"il y a 3 façons d'avoir exactement deux votants justes : as-tu le coefficient binomial ?": 0.49,
          "c'est la probabilité que les trois aient raison ; ajoute le cas de deux justes": 0.343,
          "la majorité est juste aussi quand les trois ont raison": 0.441}'''),
        ("b", "a number with 3 decimals", "majority_right(5, 0.7)",
         r'''decimals=3, mistakes={"la majorité de 5, c'est 3, 4 ou 5 votants justes : as-tu tous les termes ?": 0.309,
          "as-tu ajouté le terme des cinq votants justes ?": 0.669}'''),
        ("c", "a number with 3 decimals", "majority_right(3, 0.4)",
         r'''decimals=3, mistakes={"il y a 3 façons d'avoir exactement deux votants justes : as-tu le coefficient binomial ?": 0.16,
          "c'est la probabilité que la majorité se trompe": 0.648}'''),
        ("d", "a number with 4 decimals", "majority_right(11, 0.6)",
         r'''decimals=4, mistakes={"la majorité de 11 votants, c'est au moins 6 votants justes, pas 5": 0.9006,
          "la majorité de 11 votants, c'est au moins 6 votants justes, pas 7": 0.5328,
          "c'est la probabilité que la majorité se trompe": 0.2465}'''),
        ("e", "a number with 4 decimals", "math.comb(5, 3) * 0.7 ** 3 * 0.3 ** 2",
         r'''decimals=4, mistakes={"il y a 10 façons de choisir les trois votants justes parmi cinq": 0.0309,
          "on demande exactement trois votants justes : un seul terme de la somme": 0.8369}'''),
        ("f", "a number", "0.7",
         r'''decimals=1, mistakes={"les trois votes sont toujours identiques : la loi binomiale s'applique-t-elle ?": 0.8}'''),
        ("g", "an odd whole number", "next(M for M in range(1, 50, 2) if majority_right(M, 0.7) >= 0.9)",
         r'''mistakes={"calcule la probabilité pour ce nombre de votants : atteint-elle 0,9 ?": 7, "existe-t-il un nombre impair plus petit qui atteint 0,9 ?": 11}''',
         ),
        ("h", "a number with 3 decimals", "majority_right(101, 0.51)",
         r'''decimals=3, mistakes={"c'est la probabilité que la majorité se trompe": 0.42, "c'est p ; on demande la probabilité que la majorité ait raison": 0.51}'''),
        ("i", "a number with 3 decimals", "majority_right(25, 0.45)",
         r'''decimals=3, mistakes={"c'est la probabilité que la majorité se trompe": 0.694, "c'est p ; on demande la probabilité que la majorité ait raison": 0.45}'''),
    ]),
    Paper("14.6", "Une itération d'AdaBoost à la main", [
        ("a", "a number", "R_146[0]['err']",
         r'''decimals=3, mistakes={"cherche une souche qui ne laisse qu'un seul exemple du mauvais côté": 0.25,
          "c'est l'accuracy ; on demande l'erreur pondérée": 0.875}'''),
        ("b", "a number with 3 decimals", "R_146[0]['alpha']",
         r'''decimals=3, mistakes={"le logarithme est népérien, pas décimal": 0.845, "c'est (1 − ε)/ε ; prends son logarithme": 7,
          "la formule de la fiche n'a pas de facteur 1/2": 0.973,
          "à deux classes, ln(K − 1) vaut 0": 2.639}'''),
        ("c", "a number", "float(W_AFTER_1[1])",
         r'''decimals=3, mistakes={"as-tu normalisé les poids pour qu'ils fassent 1 au total ?": 0.875,
          "le poids de l'exemple mal classé est multiplié par e^α, puis on normalise": 0.125}'''),
        ("d", "a number with 4 decimals", "float(W_AFTER_1[0])",
         r'''decimals=4, mistakes={"les poids des exemples bien classés ne changent qu'à la normalisation : divise par le nouveau total": 0.125}'''),
        ("e", "a list [threshold, class on the left]", "[R2_146[1]['s'], R2_146[1]['g']]",
         r'''decimals=1, mistakes={"avec les nouveaux poids, la souche du tour 1 a une erreur bien plus grande : recalcule": [4.5, 0],
          "le côté gauche de ta souche prédit l'autre classe : vérifie-la sur x = 1": [1.5, 1]}'''),
        ("f", "a number with 4 decimals", "R2_146[1]['err']",
         r'''decimals=4, mistakes={"additionne les poids des exemples mal classés par la souche du tour 2, avec les poids de c) et d)": 0.25,
          "c'est l'erreur de la souche du tour 1 avec les nouveaux poids": 0.5}'''),
        ("g", "a number with 3 decimals", "R2_146[1]['alpha']",
         r'''decimals=3, mistakes={"le logarithme est népérien, pas décimal": 0.778, "c'est (1 − ε)/ε ; prends son logarithme": 6,
          "la formule de la fiche n'a pas de facteur 1/2": 0.896}'''),
        ("h", "a list of eight classes", "vote_146(R2_146)",
         r'''mistakes={"là où les deux souches se contredisent, compare leurs poids α": [0, 1, 1, 1, 1, 1, 1, 1]}'''),
        ("i", "a number with 4 decimals", "R3_146[2]['err']",
         r'''decimals=4, mistakes={"as-tu mis à jour les poids après le tour 2 (erreurs de la souche 2 multipliées par e^α₂, puis normalisation) ?": 0.3571,
          "les poids ne sont plus uniformes au tour 3 : utilise les poids mis à jour": 0.625,
          "as-tu bien compté tous les exemples que cette souche classe mal (à gauche et à droite) ?": 0.0417}'''),
        ("j", "a number with 3 decimals", "R3_146[2]['alpha']",
         r'''decimals=3, mistakes={"le logarithme est népérien, pas décimal": 0.580,
          "l'erreur de i) se calcule avec les poids mis à jour après le tour 2 : refais i) d'abord": 0.588,
          "la formule de la fiche n'a pas de facteur 1/2": 0.668, "la formule de la fiche n'a pas de facteur 1/2 (et garde α non arrondi)": 0.667}'''),
        ("k", "a list of eight classes", "vote_146(R3_146)",
         r'''mistakes={"additionne les α des souches qui votent pour chaque classe, exemple par exemple": [0, 0, 0, 0, 1, 1, 1, 1],
          "pour x = 3 et x = 4, compare la somme des α de chaque classe": [0, 1, 1, 1, 1, 1, 1, 1]}'''),
        ("l", "a number with 3 decimals", "math.log(0.7 / 0.3) + math.log(3 - 1)",
         r'''decimals=3, mistakes={"avec K = 3, ajoute ln(K − 1)": 0.847, "le terme ajouté est ln(K − 1), pas ln K": 1.946,
          "le logarithme est népérien, pas décimal": 0.669}'''),
        ("m", "True or False", "0.55 < 1 - 1 / 3",
         r'''mistakes={"avec K = 3 classes, le seuil de rejet n'est pas 0,5 : compare 0,55 à 1 − 1/K": False}'''),
    ]),
    Paper("14.7", "Deux itérations de gradient boosting sur cinq points", [
        ("a", "a number", "int(Y_147.mean())",
         r'''mistakes={"c'est la médiane ; la MSE demande une autre constante": 6}'''),
        ("b", "a list of five numbers", "[int(v) for v in Y_147 - Y_147.mean()]",
         r'''mistakes={"le résidu est la cible moins la prédiction": [4, 3, -1, -2, -4]}'''),
        ("c", "a threshold", "T1_147[0][0]",
         r'''decimals=1, mistakes={"compare les sommes des carrés des écarts des quatre seuils, sur les résidus": 3.5}'''),
        ("d", "a list [left value, right value], 3 decimals", "[T1_147[0][1], T1_147[0][2]]",
         r'''decimals=3, mistakes={"la souche apprend les résidus : chaque feuille prédit la moyenne des résidus, pas des cibles y": [1.5, 7.333],
          "as-tu pris le seuil de la question c) ?": [-2.0, 3.0]}'''),
        ("e", "a number with 2 decimals", "float(F1_147[0])",
         r'''decimals=2, mistakes={"multiplie la valeur de la feuille par le learning rate 0,5": 1.5,
          "x = 1 est dans la feuille gauche, x ≤ 2,5": 6.17,
          "F₁ = F₀ + 0,5 × la feuille : as-tu ajouté F₀ ?": -1.75}'''),
        ("f", "a number with 3 decimals", "float(np.mean((Y_147 - F1_147) ** 2))",
         r'''decimals=3, mistakes={"divise la somme des carrés par 5, le nombre d'exemples": 15.375,
          "c'est la MSE avant la première étape": 9.2,
          "as-tu multiplié la feuille par le learning rate 0,5 dans F₁ ?": 1.033}'''),
        ("g", "a threshold", "T2_147[1][0]",
         r'''decimals=1, mistakes={"la deuxième souche s'ajuste aux NOUVEAUX résidus y − F₁ : recompare les quatre seuils": 2.5,
          "as-tu calculé F₁ avec le learning rate 0,5 ?": 4.5}'''),
        ("h", "a number with 4 decimals", "predict_147(3, 0.5, T2_147)",
         r'''decimals=4, mistakes={"as-tu ajouté la contribution de la deuxième souche ?": 6.1667,
          "multiplie aussi la feuille de la deuxième souche par 0,5": 4.9444}'''),
        ("i", "a number with 4 decimals", "predict_147(4.2, 0.5, T2_147)",
         r'''decimals=4, mistakes={"as-tu ajouté la contribution de la deuxième souche ?": 6.1667,
          "multiplie aussi la feuille de la deuxième souche par 0,5": 8.0}'''),
        ("j", "a number with 4 decimals", "float(np.mean((Y_147 - F2_147) ** 2))",
         r'''decimals=4, mistakes={"as-tu ajouté la deuxième souche avant de calculer la MSE ?": 3.075, "divise la somme des carrés par 5": 6.9722}'''),
        ("k", "a number", "predict_147(1, 1.0, boost_147(1.0, 1)[1])",
         r'''decimals=1, mistakes={"avec η = 1, la feuille s'ajoute entière": 3.25, "F₁ = F₀ + la feuille : as-tu ajouté F₀ ?": -3.5}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 14.1, 14.2, 14.5 à 14.7 et ∂ 14.3)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "une suite de codes : les lettres dans l'ordre (`\"XYZ\"`) ; plusieurs nombres : une liste "
               "(`[4, 7]`). Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas "
               "encore remplies affichent ⏳. Les questions « dans ta copie », la preuve ∂ 14.4, la réflexion "
               "(🗣️ 14.8, ⚖️ 14.9, 📄 14.10) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 14.11 to 14.28)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools for the notebook exercises (parts A to D), and the penguins
import contextlib
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from wb.errors import MylearnMissingError


def verdict(ex_id, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message."""
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")


def filled(*values):
    """True when none of the values is still `...` (or None): the answer has been written."""
    return all(value is not ... and value is not None for value in values)


def fr(x, digits=3):
    """A number written the French way, with a decimal comma (for the messages in French)."""
    return f"{x:.{digits}f}".replace(".", ",")


def returned(ex_id, name, value):
    """False, with a ❌ message, when the function `name` returned None (a forgotten return); True otherwise."""
    if value is None:
        print(f"❌ Ex {ex_id} : {name} renvoie None : as-tu oublié le return ?")
        return False
    return True


def print_answer(ex_id, value, **_):
    """Solutions notebook: show the value that the exercise notebook checks with wb.check."""
    try:
        shown = np.round(np.asarray(value, dtype=float), 4).tolist() if isinstance(value, (np.ndarray, list, tuple)) else value
    except (TypeError, ValueError):     # a list of names
        shown = list(value)
    print(f"{ex_id}:", shown)


def fitted(model, X, y=None, **fit_params):
    """The model after model.fit(X, y): no chained call, so that a fit that forgets `return self` (the tests report
    it) does not stop the check cells."""
    if y is None:
        model.fit(X, **fit_params)
    else:
        model.fit(X, y, **fit_params)
    return model


@contextlib.contextmanager
def checked(ex_id):
    """Like wb.attempt (⏳ while the code is not written yet), but any other error raised by YOUR code becomes one ❌
    line instead of stopping the cell, so that the tests run after it can say more."""
    with wb.attempt(ex_id):
        try:
            yield
        except (NotImplementedError, ImportError, MylearnMissingError):
            raise
        except Exception as exc:                              # a bug in the learner's code: report it, go on
            if isinstance(exc, TypeError) and "NoneType" in str(exc):
                raise                                         # wb.attempt explains the forgotten return
            hint = " ; une fonction ou une méthode oublie-t-elle son return ?" if "NoneType" in str(exc) else ""
            print(f"❌ Ex {ex_id} : ton code lève {type(exc).__name__} : {str(exc)[:300]}{hint}")


def written(probe):
    """False while probe() stops on code not written yet (NotImplementedError, or no mylearn at all); True otherwise,
    even when it raises another error (the tests then say which)."""
    try:
        probe()
    except (NotImplementedError, ImportError, MylearnMissingError):
        return False
    except Exception:
        return True
    return True


def rmse(y_true, y_pred):
    """Root mean squared error, a Python float."""
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


TEST_FILES = {"ensemble": "tests/test_ch14_ensemble.py", "tree": "tests/test_ch13_tree.py"}


def run_mylearn_tests(module, keyword, impl="learner"):
    """Run the tests of mylearn.<module> selected by `keyword` (on YOUR code by default)."""
    test_file = TEST_FILES[module]
    command = [sys.executable, "-m", "pytest", test_file, "-k", keyword, "-q", "-p", "no:cacheprovider",
               "--color=no", "-rf", "--tb=no"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600,
                                env={**os.environ, "COLUMNS": "1000"})   # long lines: the reason of each failure
    except subprocess.TimeoutExpired:
        print("❌ les tests n'ont pas fini en 10 min : une boucle de ton code ne s'arrête-t-elle pas ?")
        return
    lines = result.stdout.strip().splitlines()
    failed = [line for line in lines if line.startswith("FAILED ")]
    if any("'NoneType' object has no attribute" in line for line in failed):
        print("💡 un attribut est lu sur None : une méthode oublie-t-elle son return ?")
    for line in failed[:8]:                                  # the test, then the reason of its failure
        name, _, reason = line.removeprefix(f"FAILED {test_file}::").partition(" - ")
        print(f"❌ {name}\n   {reason[:800]}")
    if len(failed) > 8:
        print(f"   ... and {len(failed) - 8} other failed test(s)")
    print("pytest:", lines[-1] if lines else result.stderr.strip()[-300:])


PENGUINS = wb.datasets.load_penguins(dropna=True).reset_index(drop=True)   # the 333 complete penguins
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
P_TRAIN, P_TEST = train_test_split(PENGUINS, test_size=100, random_state=1314, stratify=PENGUINS["species"])
print(len(P_TRAIN), "training penguins,", len(P_TEST), "test penguins")'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="14")   # reload your saved file\n'


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre `mon_travail/mylearn/ensemble.py` "
            "(créé par `python tools/start_chapter.py 14`), lis la docstring de chaque fonction et de chaque méthode, "
            "remplace les `raise NotImplementedError(...)` par ton code et **enregistre**. NumPy seulement, plus tes "
            "modules des chapitres précédents : `clone` (ch. 8, `model_selection.py`) et tes arbres (ch. 13, "
            "`tree.py`) ; scikit-learn est l'**oracle** des tests. La cellule de vérification recharge ta librairie, "
            "compare quelques résultats à scikit-learn, puis lance les tests de l'exercice ; `python -m pytest "
            "tests/test_ch14_ensemble.py -q` les lance tous dans un terminal (compte quelques minutes si ton arbre "
            "est lent).")


def ref_check(check: str, ex_id: str) -> str:
    """The solutions version of a 🔨 check cell: no wb.attempt, and the tests run on the reference."""
    return (check.replace(f'with wb.attempt("{ex_id}"):\n', "if True:\n")
            .replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", module="'))


def as_printed(check: str, ex_id: str) -> str:
    """The solutions version of a check cell that reads the learner's values: print them instead of checking."""
    return check.replace(f'with wb.attempt("{ex_id}"):\n', "if True:\n").replace("wb.check(", "print_answer(")


# ---- 14.11 -----------------------------------------------------------------
GIVEN_11 = r'''X_11 = PENGUINS[["bill_depth_mm", "flipper_length_mm"]].to_numpy()   # two measures only
Y_11 = PENGUINS["species"].to_numpy()
CV_11 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1411)
MEMBERS_11 = {"A": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
              "B": GaussianNB(),
              "C": DecisionTreeClassifier(max_depth=3, random_state=0)}'''

EXPERIMENT_11 = r'''# every penguin predicted by models that did not see it (10-fold cross-validation)
predictions_11 = {letter: cross_val_predict(model, X_11, Y_11, cv=CV_11) for letter, model in MEMBERS_11.items()}
accuracies_11 = {letter: float(np.mean(predicted == Y_11)) for letter, predicted in predictions_11.items()}
stacked_11 = np.array(list(predictions_11.values()))               # (3 voters, 333 penguins)
species_11 = np.unique(Y_11)
votes_11 = np.array([(stacked_11 == s).sum(axis=0) for s in species_11])
vote_result_11 = species_11[np.argmax(votes_11, axis=0)]           # plurality vote, ties: the first species
for letter, accuracy in accuracies_11.items():
    print(f"member {letter}: accuracy {accuracy:.3f}")
print(f"plurality vote of the three: accuracy {np.mean(vote_result_11 == Y_11):.3f}")
wrong_11 = stacked_11 != Y_11
best_11 = int(np.argmax(list(accuracies_11.values())))
print(f"the best member ({'ABC'[best_11]}) misclassifies {wrong_11[best_11].sum()} penguins; "
      f"the other two misclassify {wrong_11.all(axis=0)[wrong_11[best_11]].sum()} of them as well")
print(pd.crosstab(pd.Series(Y_11, name="species"), pd.Series(vote_result_11, name="vote")))
fig, ax = plt.subplots(figsize=(6, 4.5))
for s in species_11:
    ax.scatter(*X_11[Y_11 == s].T, s=14, label=s)
ax.scatter(*X_11[vote_result_11 != Y_11].T, s=60, facecolors="none", edgecolors="black", label="vote wrong")
ax.set_xlabel("bill_depth_mm")
ax.set_ylabel("flipper_length_mm")
ax.legend()
plt.show()
wb.check("14.11a", vote_11)
wb.check("14.11b", shared_11)
wb.check("14.11c", pair_11)'''

# ---- 14.12 -----------------------------------------------------------------
CHECK_12 = r'''with checked("14.12"):
    doc_12 = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 1]])
    plain_12 = mylearn.ensemble.plurality_vote(doc_12)
    weighted_12 = mylearn.ensemble.plurality_vote(doc_12, weights=[1.0, 1.0, 3.0])
    if returned("14.12", "plurality_vote", plain_12) and returned("14.12", "plurality_vote", weighted_12):
        print("docstring examples:", np.asarray(plain_12).tolist(), np.asarray(weighted_12).tolist(),
              "(expected [0, 0, 1] and [1, 0, 1])")
with checked("14.12"):
    ballots_12 = np.random.default_rng(1412).integers(0, 4, size=(7, 500))      # 7 voters, 500 samples, labels 0 to 3
    mine_12 = np.asarray(mylearn.ensemble.plurality_vote(ballots_12))
    oracle_12 = stats.mode(ballots_12, axis=0).mode                                # ties: the smallest label
    verdict("14.12", mine_12.shape == (500,) and np.array_equal(mine_12, oracle_12),
            "sans poids, tes 500 votes sont ceux de scipy.stats.mode.",
            ("sans poids, tes votes diffèrent de scipy.stats.mode (relis la règle d'égalité : la plus petite classe) ; "
             if mine_12.shape == (500,) else f"sans poids, ton vote rend un tableau de forme {mine_12.shape}, au lieu "
             "d'une classe par exemple (500,) ; ") + "les tests ci-dessous disent où.")
    w_12 = np.array([2.0, 1.0, 1.5, -1.0, 0.5, 0.0, 3.0])
    totals_12 = np.array([[w_12[ballots_12[:, j] == k].sum() for k in range(4)] for j in range(500)])
    verdict("14.12", np.array_equal(np.asarray(mylearn.ensemble.plurality_vote(ballots_12, w_12)),
                                    np.argmax(totals_12, axis=1)),
            "avec des poids (un négatif, un nul), tes votes sont ceux d'une boucle qui additionne les poids.",
            "avec des poids, tes votes diffèrent d'une boucle qui additionne les poids de chaque classe (un poids "
            "négatif retire des voix).")
with checked("14.12"):
    draws_12 = mylearn.ensemble.bootstrap_indices(1000, rng=np.random.default_rng(1412))
    if returned("14.12", "bootstrap_indices", draws_12):
        draws_12 = np.asarray(draws_12)
        distinct_12 = len(np.unique(draws_12)) / 1000
        print(f"bootstrap of 1000 among 1000: {distinct_12:.3f} distinct indices (about 0.632 expected)")
        verdict("14.12", draws_12.shape == (1000,) and 0.60 < distinct_12 < 0.66
                and np.array_equal(draws_12, np.asarray(mylearn.ensemble.bootstrap_indices(1000, rng=np.random.default_rng(1412))))
                and np.array_equal(np.asarray(mylearn.ensemble.bootstrap_indices(50, rng=7)),
                                   np.asarray(mylearn.ensemble.bootstrap_indices(50, rng=7))),
                "1000 tirages, environ 63 % d'indices distincts, les mêmes avec le même générateur ou le même entier.",
                "attendu : 1000 indices, environ 63 % de distincts, et les mêmes tirages pour deux générateurs de même "
                "graine (ou deux fois rng=7) ; les tests ci-dessous disent où.")
if written(lambda: mylearn.ensemble.plurality_vote(np.array([[0]]))) or written(lambda: mylearn.ensemble.bootstrap_indices(2, rng=0)):
    run_mylearn_tests("ensemble", keyword="test_plurality_vote_ or test_bootstrap_indices_")
else:
    print("⏳ Ex 14.12 : les tests attendent ton plurality_vote et ton bootstrap_indices.")'''

# ---- 14.13 -----------------------------------------------------------------
GIVEN_13 = r'''MS_13 = [1, 3, 5, 11, 21, 51, 101]   # sizes of the jury (odd)
Q_13 = 0.58                           # each voter is right with probability 0.58


def jury_13(M, q, c, n_trials=20_000, seed=1413):
    """Simulate n_trials votes of a jury of M voters, each right with probability q. In each trial, with probability c,
    every voter copies one common vote (itself right with probability q); otherwise each voter decides on its own.
    Return the boolean array of shape (n_trials, M): True where a voter is right."""
    rng = np.random.default_rng(seed)
    common = rng.random(n_trials) < q
    own = rng.random((n_trials, M)) < q
    copy = rng.random(n_trials) < c
    return np.where(copy[:, None], common[:, None], own)'''

TODO_13 = r'''exact_13 = ...       # a) list: for each M of MS_13, the exact probability that the majority of M independent voters,
                     #    each right with probability Q_13, is right (binomial law)
first_090_13 = ...   # b) the smallest M of MS_13 whose probability in a) is at least 0.9
simulated_13 = ...   # c) list: for each M of MS_13, the share of the trials of jury_13(M, Q_13, 0.0) where the majority
                     #    of the voters is right
correlated_13 = ...  # d) the same list as c) with c = 0.3: jury_13(M, Q_13, 0.3)
corr_13 = ...        # e) the correlation coefficient between the columns 0 and 1 of jury_13(11, Q_13, 0.3)
plateau_13 = ...     # f) the limit of the probability in d) when M grows without bound (exact, by reasoning)
q_needed_13 = ...    # g) with c = 0.3, the smallest q for which this limit reaches 0.9 (exact)'''

SOLUTION_13 = r'''def majority_right_13(M, q):
    """Exact probability that more than half of M independent voters are right (binomial law)."""
    return float(stats.binom.sf(M // 2, M, q))          # P(X > M // 2) = P(X >= (M + 1) / 2) for an odd M


def majority_share_13(right):
    """Share of the trials (rows) where more than half of the voters (columns) are right."""
    return float(np.mean(right.sum(axis=1) > right.shape[1] / 2))


exact_13 = [majority_right_13(M, Q_13) for M in MS_13]
first_090_13 = next(M for M, p in zip(MS_13, exact_13) if p >= 0.9)
simulated_13 = [majority_share_13(jury_13(M, Q_13, 0.0)) for M in MS_13]
correlated_13 = [majority_share_13(jury_13(M, Q_13, 0.3)) for M in MS_13]
corr_13 = float(np.corrcoef(jury_13(11, Q_13, 0.3)[:, 0], jury_13(11, Q_13, 0.3)[:, 1])[0, 1])
plateau_13 = 0.3 * Q_13 + 0.7 * 1.0      # copy: right with probability q; own votes: the majority is right for sure
q_needed_13 = (0.9 - 0.7) / 0.3
'''

CHECK_13 = r'''wb.check("14.13a", exact_13)
wb.check("14.13b", first_090_13)
wb.check("14.13c", simulated_13)
wb.check("14.13d", correlated_13)
wb.check("14.13e", corr_13)
wb.check("14.13f", plateau_13)
wb.check("14.13g", q_needed_13)
with wb.attempt("14.13"):
    if not filled(exact_13, simulated_13, correlated_13):
        raise NotImplementedError("exact_13, simulated_13, correlated_13")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(MS_13, exact_13, marker="o", label="independent, exact")
    ax.plot(MS_13, simulated_13, marker="x", ls="--", label="independent, simulated")
    ax.plot(MS_13, correlated_13, marker="s", label="c = 0.3, simulated")
    if filled(plateau_13):
        ax.axhline(plateau_13, color="grey", ls=":", label="your limit for c = 0.3")
    ax.axhline(Q_13, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("number of voters M")
    ax.set_ylabel("probability that the majority is right")
    ax.legend()
    plt.show()'''

# ---- 14.14 -----------------------------------------------------------------
GIVEN_14 = r'''X_14 = PENGUINS[MEASURES].to_numpy()
Y_14 = PENGUINS["species"].to_numpy()
CV_14 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1414)


def on_column(column, model):
    """A pipeline that gives `model` only the column `column` of X."""
    return make_pipeline(ColumnTransformer([("keep", "passthrough", [column])]), model)


MEMBERS_14 = [("A", on_column(0, make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)))),   # bill length
              ("B", on_column(1, GaussianNB())),                                                           # bill depth
              ("C", on_column(2, DecisionTreeClassifier(max_depth=3, random_state=0)))]                    # flipper
for letter, member in MEMBERS_14:
    print(f"member {letter} alone: mean accuracy {cross_val_score(member, X_14, Y_14, cv=CV_14).mean():.3f}")'''

TODO_14 = r'''hard_14 = ...     # a) mean accuracy (cross_val_score with CV_14) of VotingClassifier(MEMBERS_14, voting="hard")
soft_14 = ...     # b) the same with voting="soft"
stack_14 = ...    # c) the same for StackingClassifier(MEMBERS_14, final_estimator=LogisticRegression(max_iter=1000))
single_14 = ...   # d) the same for ONE model that sees the 4 measures:
                  #    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))'''

SOLUTION_14 = r'''hard_14 = cross_val_score(VotingClassifier(MEMBERS_14, voting="hard"), X_14, Y_14, cv=CV_14).mean()
soft_14 = cross_val_score(VotingClassifier(MEMBERS_14, voting="soft"), X_14, Y_14, cv=CV_14).mean()
stack_14 = cross_val_score(StackingClassifier(MEMBERS_14, final_estimator=LogisticRegression(max_iter=1000)),
                           X_14, Y_14, cv=CV_14).mean()
single_14 = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X_14, Y_14,
                            cv=CV_14).mean()
'''

CHECK_14 = r'''wb.check("14.14a", hard_14)
wb.check("14.14b", soft_14)
wb.check("14.14c", stack_14)
wb.check("14.14d", single_14)'''

PART_A = Part("A", "Faire voter des modèles",
              "Trois classifieurs du ch. 13 qui votent (14.11), ton vote pondéré et ton bootstrap (14.12), ce que la "
              "corrélation fait à un jury (14.13), puis voting et stacking avec scikit-learn (14.14). La cellule "
              "suivante importe les ensembles de scikit-learn, définit les outils des parties A à D et découpe les 333 "
              "manchots complets (233 pour l'entraînement, 100 pour le test), comme au ch. 13. Ne recopie jamais un "
              "nombre à la main : chaque réponse est calculée par ton code, et pas arrondie.",
              given=PART_A_GIVEN, exercises=[
    Ex("14.11", "🔮", 1, 10, "Trois classifieurs du ch. 13 votent : mieux que le meilleur ?",
       "prévoir, avant de le mesurer, ce qu'un vote apporte à trois classifieurs entraînés sur les mêmes données.",
       "ch. 13 (kNN, Naïve Bayes, arbres) · fiche §14.1 à §14.3", thread="Penguins", tracks="R, C",
       hypothesis=True,
       body=r"""Les 333 manchots, décrits par **deux** mesures seulement : l'épaisseur du bec (`bill_depth_mm`) et la longueur de la nageoire (`flipper_length_mm`). Trois classifieurs du ch. 13 : A un kNN à 15 voisins sur les mesures standardisées, B `GaussianNB()`, C un arbre de profondeur 3. Chaque manchot est prédit par des modèles qui ne l'ont pas vu (validation croisée à 10 folds, `cross_val_predict`), puis les trois prédictions votent à la pluralité. Les trois membres font entre 79 et 82 % d'accuracy.

Prédis, **avant** d'exécuter quoi que ce soit (un point = 0,01 d'accuracy) :
a) `vote_11` : l'accuracy du vote, comparée à celle du meilleur des trois : `"A"` plus d'un point **sous** le meilleur ; `"B"` à moins d'un point du meilleur (au-dessus ou au-dessous) ; `"C"` de 1 à 3 points au-dessus ; `"D"` plus de 3 points au-dessus ;
b) `shared_11` : parmi les manchots que le meilleur membre classe mal, la part que les **deux autres** classent mal aussi : `"A"` moins de 25 % ; `"B"` de 25 à 50 % ; `"C"` de 50 à 75 % ; `"D"` plus de 75 % ;
c) `pair_11` : les erreurs du vote confondent surtout : `"A"` Adélie et Chinstrap ; `"B"` Adélie et Gentoo ; `"C"` Chinstrap et Gentoo ; `"D"` les erreurs se répartissent sur les trois paires.

Puis exécute l'expérience : elle affiche l'accuracy de chaque membre et du vote, les erreurs partagées, la table des espèces et des votes, et entoure les manchots mal classés.""",
       given=GIVEN_11,
       todo=r'''vote_11 = ...     # a) "A", "B", "C" or "D"
shared_11 = ...   # b) "A", "B", "C" or "D"
pair_11 = ...     # c) "A", "B", "C" or "D"''',
       solution=r'''vote_11, shared_11, pair_11 = "B", "D", "A"   # the answers, for the record''',
       after=[("code", guarded(EXPERIMENT_11, ["vote_11", "shared_11", "pair_11"],
                               "⏳ Ex 14.11 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : compare avec tes prédictions. Les deux conditions de la fiche (§14.2) sont-elles "
                     "remplies par ces trois membres ? Leurs erreurs se ressemblent-elles, et que faudrait-il changer "
                     "pour qu'elles se ressemblent plus ou moins ? Ce que tu as observé sur ces deux mesures "
                     "vaudrait-il encore avec les quatre ?")],
       record=r'''wb.record("14.11a", "B", mistakes={"un vote à la pluralité ne descend nettement sous son meilleur membre que si les deux autres se trompent ensemble là où il a raison : est-ce le cas de membres qui butent sur les mêmes manchots ?": "A",
                                    "pour gagner des points, il faudrait des manchots que le meilleur membre rate et que les deux autres réussissent : relis b), et pense à ce que ces deux mesures laissent indistinct": "C",
                                    "plus de trois points, c'est une dizaine de manchots rattrapés : des membres entraînés sur les mêmes deux mesures se trompent-ils sur des manchots différents ?": "D"})
wb.record("14.11b", "D", mistakes={"les trois membres voient les mêmes deux mesures : les manchots difficiles pour l'un sont-ils faciles pour les autres ?": "A",
                                    "les erreurs d'un membre viennent-elles surtout de son modèle, ou d'une zone où deux espèces se mélangent dans ces deux mesures ?": "B",
                                    "pense à la zone où deux espèces ont les mêmes mesures : un modèle, quel qu'il soit, peut-il y faire mieux que les autres ?": "C"})
wb.record("14.11c", "A", mistakes={"relis les mesures des Gentoo au ch. 12 ou 13 : leurs nageoires et leurs becs les distinguent-ils des deux autres espèces ?": "B",
                                    "relis les mesures des Gentoo au ch. 12 ou 13 : leurs nageoires et leurs becs ressemblent-ils à ceux d'une autre espèce ?": "C",
                                    "l'une des trois espèces se distingue nettement des deux autres sur ces deux mesures : laquelle, et que reste-t-il à confondre ?": "D"})''',
       note="Les trois membres font 0,802 (kNN), 0,817 (Naïve Bayes) et 0,799 (arbre) ; le vote fait 0,814, à moins "
            "d'un point du meilleur. Le meilleur classe mal 61 manchots, et les deux autres en classent mal 55 aussi "
            "(90 %). Toutes les erreurs du vote (62) confondent Adélie et Chinstrap : sur l'épaisseur du bec et la "
            "nageoire, ces deux espèces se superposent, alors que les Gentoo (bec mince, longue nageoire) se "
            "détachent. Les trois membres font mieux que le hasard, mais leurs erreurs sont presque les mêmes : la "
            "seconde condition du §14.2 manque. Avec d'autres découpages en folds (graines 1 à 4), le vote reste "
            "entre −0,3 et +0,3 point du meilleur et la part des erreurs partagées entre 87 et 92 %. La longueur du "
            "bec, qui sépare Adélie et Chinstrap, changerait tout ; 📦 14.14 donne à chaque membre une mesure "
            "différente."),

    Ex("14.12", "🔨", 2, 25, "Coder le vote pondéré et le tirage bootstrap",
       "écrire les deux briques de tous les ensembles : un vote pondéré vectorisé et un tirage avec remise.",
       "✏️ 14.1 · ∂ 14.3 · ch. 2 (bootstrap) · fiche §14.3 et §14.4", tracks="C", mylearn="ensemble.py",
       body=mylearn_howto() + r"""

Écris `plurality_vote` et `bootstrap_indices`. **Contrat** (les docstrings, plus ces précisions que vérifient les tests) :
- `plurality_vote(predictions, weights=None)` reçoit un tableau 2-D (ou une liste de listes) de forme `(n_voters, n_samples)`, une ligne par votant, et rend un tableau `(n_samples,)` : pour chaque exemple, la classe dont la somme des poids des votants est la plus grande ; une classe que personne ne désigne a un total nul, un poids peut être négatif ou nul ; à égalité, la **plus petite** classe, dans l'ordre de tri des labels (des entiers ou des chaînes) ; `weights=None` donne une voix à chacun ; `ValueError` si `predictions` n'est pas 2-D, si `weights` n'a pas une valeur par votant ou contient `NaN` ; elle ne modifie pas ses arguments ;
- `bootstrap_indices(n_samples, n_draws=None, rng=None)` rend un tableau d'**entiers** de forme `(n_draws,)` (`n_samples` tirages si `n_draws` vaut `None`), tirés uniformément **avec remise** dans $0, \dots, n - 1$ ; `rng` accepte `None`, un entier ou un `Generator` ; deux appels avec deux générateurs de même graine rendent les mêmes indices, deux appels successifs avec le **même** générateur en rendent d'autres ; `ValueError` si `n_samples < 1` ou `n_draws < 1`.

La vérification essaie les exemples des docstrings, compare tes votes à `scipy.stats.mode` (sans poids) et à une boucle (avec des poids), mesure la part d'indices distincts d'un bootstrap, puis lance les tests.

Dans tes notes : combien d'opérations ton vote fait-il pour $M$ votants, $n$ exemples et $K$ classes ? Pourquoi `bootstrap_indices` reçoit-elle un générateur plutôt qu'une graine fixée à l'intérieur ?""",
       check=RELOAD + CHECK_12, solution=ref_check(CHECK_12, "14.12"),
       note="Un vote vectorisé compare la matrice des prédictions à chacune des $K$ classes et additionne les poids "
            "colonne par colonne : $M \\times n \\times K$ opérations, sans boucle Python sur les exemples. Un "
            "générateur passé en argument continue sa suite d'un appel à l'autre : un bagging tire ainsi un "
            "bootstrap **différent** pour chaque membre à partir d'une seule graine, et toute l'expérience se "
            "reproduit ; une graine fixée à l'intérieur donnerait le même bootstrap à tous les membres, qui "
            "deviendraient identiques."),

    Ex("14.13", "🔬", 2, 30, "Jury de votants : indépendants puis corrélés",
       "mesurer ce que la corrélation des erreurs coûte à un vote, et retrouver la limite par le calcul.",
       "✏️ 14.5 · ch. 2 (corrélation) · fiche §14.2 (encadré sur la loi binomiale)", thread="synthétique", tracks="R, M, C",
       body=r"""Un jury de $M$ votants (`MS_13` : 1 à 101), chacun juste avec la probabilité $q$ = `Q_13` = 0,58, à deux classes. La fonction fournie `jury_13(M, q, c)` simule 20 000 votes : à chaque vote, avec la probabilité $c$, **tous** les votants recopient un même avis (juste avec la probabilité $q$) ; sinon, chacun décide seul. Chaque votant reste juste avec la probabilité $q$, mais leurs erreurs sont corrélées dès que $c > 0$. La majorité a raison quand plus de la moitié des votants ont raison.

Calcule :
a) `exact_13` : pour chaque $M$ de `MS_13`, la probabilité **exacte** que la majorité de $M$ votants indépendants ait raison (loi binomiale : `scipy.stats.binom`, importé sous le nom `stats`, ou `math.comb`) ;
b) `first_090_13` : le plus petit $M$ de `MS_13` dont la probabilité en a) atteint 0,9 ;
c) `simulated_13` : pour chaque $M$, la part des votes de `jury_13(M, Q_13, 0.0)` où la majorité a raison (elle doit suivre a) ;
d) `correlated_13` : la même liste avec $c = 0{,}3$ ;
e) `corr_13` : le coefficient de corrélation entre les colonnes 0 et 1 de `jury_13(11, Q_13, 0.3)` (deux votants : `np.corrcoef`) ;
f) `plateau_13` : la limite de la probabilité de d) quand $M$ grandit sans fin, **par le raisonnement** (exacte) ;
g) `q_needed_13` : avec $c = 0{,}3$, le plus petit $q$ pour lequel cette limite atteint 0,9 (exact).

La vérification contrôle a) à g), puis trace les trois courbes et ta limite.

Dans tes notes : compare e) à $c$, et explique pourquoi. Que dit f) sur un ensemble d'arbres entraînés sur les mêmes données ? Relie-le à la variance d'une moyenne corrélée (fiche §14.4).""",
       given=GIVEN_13, todo=TODO_13, check=CHECK_13, solution=SOLUTION_13 + as_printed(CHECK_13, "14.13"),
       record=r'''sums_13 = [float(stats.binom.sf(M // 2 - 1, M, Q_13)) for M in MS_13]
wb.record("14.13a", exact_13, decimals=4,
          mistakes={"la majorité de M votants (M impair), c'est au moins (M + 1)/2 votants justes : as-tu inclus un terme de trop dans la somme ?": sums_13})
wb.record("14.13b", first_090_13, choices=MS_13,
          fractional="un nombre non entier : est-ce une probabilité ? on demande un M de MS_13",
          mistakes={f"M = {M} : sa probabilité en a) atteint-elle 0,9 ? et est-ce le PREMIER M de MS_13 qui l'atteint ?": M
                    for M in MS_13 if M != first_090_13})
wb.record("14.13c", simulated_13, decimals=3,
          mistakes={"relis la question : on demande les parts mesurées sur les votes simulés de jury_13(M, Q_13, 0.0), pas un calcul exact": exact_13})
wb.record("14.13d", correlated_13, decimals=3,
          mistakes={"relis la question : les votes de d) viennent de jury_13(M, Q_13, 0.3)": simulated_13})
independent_corr_13 = float(np.corrcoef(jury_13(11, Q_13, 0.0)[:, 0], jury_13(11, Q_13, 0.0)[:, 1])[0, 1])
wb.record("14.13e", corr_13, decimals=3,
          mistakes={"relis la question : la corrélation se mesure sur jury_13(11, Q_13, 0.3), avec c = 0,3": independent_corr_13,
                    "c'est c lui-même : on demande la corrélation mesurée sur les votes simulés": 0.3})
wb.record("14.13f", plateau_13, decimals=3,
          mistakes={"tu n'as gardé que le cas où tous recopient : quand chacun décide seul, vers quoi tend la probabilité que la majorité ait raison, et avec quel poids compte ce cas ?": 0.3 * Q_13,
                    "relis la question : c'est la limite pour c = 0,3, pas la probabilité qu'un votant ait raison": Q_13,
                    "quand tous recopient, la majorité a raison avec la probabilité q, pas à coup sûr": 1.0})
wb.record("14.13g", q_needed_13, decimals=3,
          mistakes={"écris la limite de f) avec un q quelconque, puis résous « limite = 0,9 » pour q": 0.9,
                    "tu as résolu 0,3 q + 0,7 = 0,9 sans diviser par 0,3 : isole q": 0.9 - 0.7})''',
       note="Le jury indépendant monte de 0,58 ($M = 1$) à 0,948 ($M = 101$) ; il faut attendre $M = 101$ pour "
            "dépasser 0,9 (0,876 à $M = 51$) : avec des votants à peine meilleurs que le hasard, il en faut beaucoup. "
            "La simulation suit la loi binomiale à 0,005 près. Avec $c = 0{,}3$, la courbe plafonne (0,836 à $M = 101$) : quand tous "
            "recopient le même avis (30 % des votes), la majorité a raison avec la probabilité $q$ quel que soit "
            "$M$ ; sinon, elle a raison presque à coup sûr pour un grand jury : la limite vaut $0{,}3 \\times 0{,}58 "
            "+ 0{,}7 = 0{,}874$, sous 0,9 pour tout $M$, et il faudrait $q \\ge 2/3$ pour l'atteindre. Deux votants "
            "ont une corrélation proche de $c$ (0,304 mesurée) : avec la probabilité $c$ ils sont identiques, sinon "
            "indépendants. C'est le plancher $\\rho\\sigma^2$ de la variance d'une moyenne corrélée : aucun nombre "
            "de membres ne compense des erreurs communes."),

    Ex("14.14", "📦", 2, 25, "Voting et stacking avec les classifieurs du ch. 13",
       "comparer vote dur, vote souple et stacking, et voir ce que vaut une combinaison face à un seul bon modèle.",
       "Ex 14.11 · ch. 8 (validation croisée) · fiche §14.3 (voting et stacking)", thread="Penguins", tracks="C",
       body=r"""Pour que les membres fassent des erreurs **différentes**, chacun ne voit ici qu'**une** mesure : A un kNN à 15 voisins sur la longueur du bec, B `GaussianNB()` sur l'épaisseur du bec, C un arbre de profondeur 3 sur la nageoire (`MEMBERS_14`, une liste de paires `(nom, modèle)`, le format de scikit-learn). La cellule fournie affiche l'accuracy de chacun en validation croisée à 5 folds (`CV_14`, sur les 333 manchots et leurs 4 mesures `X_14` ; chaque membre choisit sa colonne).

Calcule, avec `cross_val_score(..., X_14, Y_14, cv=CV_14)` et la **moyenne** des 5 folds :
a) `hard_14` : `VotingClassifier(MEMBERS_14, voting="hard")` ;
b) `soft_14` : le même avec `voting="soft"` ;
c) `stack_14` : `StackingClassifier(MEMBERS_14, final_estimator=LogisticRegression(max_iter=1000))` (par défaut, son méta-modèle apprend sur les `predict_proba` des membres, obtenus par une validation croisée interne à 5 folds) ;
d) `single_14` : **un seul** modèle qui voit les 4 mesures, `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`.

Dans tes notes : classe les quatre et les trois membres. Pourquoi le vote souple bat-il le vote dur ici ? Que sait faire le méta-modèle que les deux votes ne savent pas ? Pourquoi un seul modèle sur les 4 mesures fait-il encore mieux, et quand un stacking vaut-il vraiment la peine ?""",
       given=GIVEN_14, todo=TODO_14, check=CHECK_14, solution=SOLUTION_14 + as_printed(CHECK_14, "14.14"),
       record=r'''members_14 = [cross_val_score(member, X_14, Y_14, cv=CV_14).mean() for _, member in MEMBERS_14]
wb.record("14.14a", hard_14, decimals=4,
          mistakes={"relis la question : voting=\"hard\" compte les classes prédites, sans les probabilités": soft_14})
wb.record("14.14b", soft_14, decimals=4,
          mistakes={"relis la question : voting=\"soft\" moyenne les predict_proba des membres": hard_14})
wb.record("14.14c", stack_14, decimals=4)
wb.record("14.14d", single_14, decimals=4)''',
       note="Seuls, les membres font 0,778 (bec), 0,754 (épaisseur) et 0,811 (nageoire). Le vote dur fait 0,799, "
            "sous le meilleur membre : quand deux membres faibles se trompent ensemble, ils l'emportent sur le "
            "troisième. Le vote souple fait 0,856 : un membre peu sûr de lui pèse peu, un membre sûr pèse beaucoup. "
            "Le stacking fait 0,949 : le méta-modèle apprend, sur des prédictions hors échantillon, quel membre "
            "croire **pour quelle espèce** (la nageoire pour les Gentoo, la longueur du bec pour séparer Adélie et "
            "Chinstrap) ; il combine en fait l'information des trois mesures. Un seul modèle qui voit les 4 mesures "
            "fait 0,991 : combiner des modèles ne remplace pas une bonne représentation des données. Un stacking "
            "vaut la peine pour combiner des modèles **différents et bons** (un gradient boosting, un réseau, un "
            "kNN), et c'est souvent ainsi que se gagnent les compétitions, au prix d'un modèle plus lourd."),
])

# ---- Part B ----------------------------------------------------------------
PART_B_GIVEN = r'''# California: the split of the challenge 14.28 (20 % of test districts, random_state=42), shared by parts B to D
CAL = wb.datasets.load_california()
CAL_FEATURES = [column for column in CAL.columns if column != "MedHouseVal"]
X_CAL, Y_CAL = CAL[CAL_FEATURES].to_numpy(), CAL["MedHouseVal"].to_numpy()
X_CAL_TRAIN, X_CAL_TEST, Y_CAL_TRAIN, Y_CAL_TEST = train_test_split(X_CAL, Y_CAL, test_size=0.2, random_state=42)
print(len(X_CAL_TRAIN), "training districts,", len(X_CAL_TEST), "test districts")'''

GIVEN_15 = r'''X_TRAIN_15, Y_TRAIN_15 = X_CAL_TRAIN[:5000], Y_CAL_TRAIN[:5000]   # 5,000 training districts
NS_15 = [10, 25, 50, 100, 200]'''

TODO_15 = r'''forest_15 = ...       # RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0, n_jobs=-1)
                      # fitted on X_TRAIN_15, Y_TRAIN_15
oob_15 = ...          # a) its out-of-bag R² (oob_score_)
test_15 = ...         # b) its R² on the test districts X_CAL_TEST, Y_CAL_TEST
oob_curve_15 = ...    # c) list: the oob_score_ of the same forest with n_estimators = n, for each n of NS_15
first_close_15 = ...  # d) the smallest n of NS_15 whose oob_score_ is within 0.01 of the test R² of the same forest'''

SOLUTION_15 = r'''forest_15 = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0, n_jobs=-1).fit(X_TRAIN_15, Y_TRAIN_15)
oob_15 = forest_15.oob_score_
test_15 = forest_15.score(X_CAL_TEST, Y_CAL_TEST)
oob_curve_15, test_curve_15 = [], []
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)        # with 10 trees, a few districts have no out-of-bag tree
    for n in NS_15:
        model = RandomForestRegressor(n_estimators=n, oob_score=True, random_state=0, n_jobs=-1).fit(X_TRAIN_15, Y_TRAIN_15)
        oob_curve_15.append(model.oob_score_)
        test_curve_15.append(model.score(X_CAL_TEST, Y_CAL_TEST))
first_close_15 = next(n for n, oob, test in zip(NS_15, oob_curve_15, test_curve_15) if abs(oob - test) < 0.01)
print("test R² for each n:", np.round(test_curve_15, 4).tolist())
'''

CHECK_15 = r'''wb.check("14.15a", oob_15)
wb.check("14.15b", test_15)
wb.check("14.15c", oob_curve_15)
wb.check("14.15d", first_close_15)
with wb.attempt("14.15"):
    if not filled(forest_15):
        raise NotImplementedError("forest_15")
    seconds_15, predictions_15 = {}, {}
    for n_jobs in [1, -1]:                                  # one core, then all the cores
        start_15 = time.perf_counter()
        model_15 = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=n_jobs).fit(X_TRAIN_15, Y_TRAIN_15)
        seconds_15[n_jobs] = time.perf_counter() - start_15
        predictions_15[n_jobs] = model_15.predict(X_CAL_TEST)
    print(f"{os.cpu_count()} core(s): fit in {seconds_15[1]:.2f} s with n_jobs=1, in {seconds_15[-1]:.2f} s with "
          f"n_jobs=-1 (speed-up × {seconds_15[1] / seconds_15[-1]:.1f})")
    print("same predictions to the last bit:", np.array_equal(predictions_15[1], predictions_15[-1]),
          "· largest difference:", float(np.max(np.abs(predictions_15[1] - predictions_15[-1]))))
    verdict("14.15", np.allclose(predictions_15[1], predictions_15[-1], rtol=0, atol=1e-9),
            "avec ou sans parallélisme, la même forêt (même random_state) donne les mêmes prédictions, à 1e-9 près.",
            "les prédictions changent avec n_jobs : est-ce bien la même forêt, avec le même random_state ?")'''

# ---- 14.16 -----------------------------------------------------------------
GIVEN_16 = r'''NS_16 = [1, 2, 5, 10, 25, 50, 100, 200, 400]


def moons_16(n_noise):
    """Two noisy moons plus n_noise columns of standard normal noise: 500 training points, 3,000 test points."""
    X, y = wb.synth.make_moons(n=3500, noise=0.35, seed=1416)
    X = np.hstack([X, np.random.default_rng(1416).normal(size=(len(X), n_noise))])
    return X[:500], y[:500], X[500:], y[500:]


ENSEMBLES_16 = {"bagging": lambda n: BaggingClassifier(DecisionTreeClassifier(), n_estimators=n, random_state=0),
                "forest": lambda n: RandomForestClassifier(n_estimators=n, random_state=0),
                "extra": lambda n: ExtraTreesClassifier(n_estimators=n, random_state=0)}'''

TODO_16 = r'''def curves_16(n_noise):
    """Return a dict {"bagging": [...], "forest": [...], "extra": [...]}: for each ensemble of ENSEMBLES_16 (called with
    the number of trees), the list of its test accuracies on moons_16(n_noise), one per number of trees of NS_16."""
    raise NotImplementedError("curves_16")


clean_16 = ...        # curves_16(0)
noisy_16 = ...        # a) curves_16(30)
worst_16 = ...        # b) the ensemble with the lowest test accuracy with 400 trees and 30 noise columns:
                      #    "A" bagging, "B" forest, "C" ExtraTrees
gain_extra_16 = ...   # c) ExtraTrees with 30 noise columns: accuracy with 400 trees minus accuracy with 25 trees
drop_16 = ...         # d) True if one of the six curves (clean_16 and noisy_16) loses more than 0.005 between 100 and
                      #    400 trees, False otherwise'''

SOLUTION_16 = r'''def curves_16(n_noise):
    """Return a dict {"bagging": [...], "forest": [...], "extra": [...]}: for each ensemble of ENSEMBLES_16 (called with
    the number of trees), the list of its test accuracies on moons_16(n_noise), one per number of trees of NS_16."""
    X_train, y_train, X_test, y_test = moons_16(n_noise)
    return {name: [make(n).fit(X_train, y_train).score(X_test, y_test) for n in NS_16]
            for name, make in ENSEMBLES_16.items()}


clean_16 = curves_16(0)
noisy_16 = curves_16(30)
worst_16 = "ABC"[int(np.argmin([noisy_16[name][-1] for name in ENSEMBLES_16]))]
gain_extra_16 = noisy_16["extra"][NS_16.index(400)] - noisy_16["extra"][NS_16.index(25)]
drop_16 = any(curve[NS_16.index(100)] - curve[NS_16.index(400)] > 0.005
              for curves in (clean_16, noisy_16) for curve in curves.values())
'''

CHECK_16 = r'''wb.check("14.16a", noisy_16.get("extra") if isinstance(noisy_16, dict) else noisy_16)
wb.check("14.16b", worst_16)
wb.check("14.16c", gain_extra_16)
wb.check("14.16d", drop_16)
with wb.attempt("14.16"):
    if not filled(clean_16, noisy_16):
        raise NotImplementedError("clean_16, noisy_16")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (family, title, curves) in zip(axes, [("clean_16", "2 features (the moons)", clean_16),
                                                  ("noisy_16", "2 features + 30 noise columns", noisy_16)]):
        if not isinstance(curves, dict):
            print(f"❌ Ex 14.16 : {family} doit être un dictionnaire {{nom de l'ensemble: liste d'accuracies}}, pas un {type(curves).__name__}.")
            continue
        for name, curve in curves.items():
            if np.ndim(curve) != 1 or len(curve) != len(NS_16):
                print(f"❌ Ex 14.16 : {family}[{name!r}] doit être une liste de {len(NS_16)} accuracies, une par nombre "
                      f"d'arbres de NS_16 (forme reçue : {np.shape(curve)}).")
                continue
            ax.plot(NS_16, curve, marker="o", label=name)
        ax.set_xscale("log")
        ax.set_xlabel("number of trees")
        ax.set_title(title)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend()
    plt.show()'''

# ---- 14.17 -----------------------------------------------------------------
CHECK_17 = r'''with checked("14.17"):
    X_doc_17 = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    y_doc_17 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    doc_17 = fitted(mylearn.ensemble.BaggingClassifier(n_estimators=5, random_state=0), X_doc_17, y_doc_17)
    predicted_doc_17 = doc_17.predict(np.array([[1.5], [11.5]]))
    if returned("14.17", "predict", predicted_doc_17):
        print("docstring example:", len(doc_17.estimators_), "members, predict", np.asarray(predicted_doc_17).tolist(),
              "(expected 5 members and [0, 1])")
with checked("14.17"):
    X_17, y_17 = P_TRAIN[["bill_depth_mm", "flipper_length_mm"]].to_numpy(), P_TRAIN["species"].to_numpy()
    X_test_17, y_test_17 = P_TEST[["bill_depth_mm", "flipper_length_mm"]].to_numpy(), P_TEST["species"].to_numpy()
    mine_17, oracle_17 = [], []
    start_17 = time.perf_counter()
    for seed in range(5):
        model = fitted(mylearn.ensemble.BaggingClassifier(n_estimators=25, oob_score=True, random_state=seed), X_17, y_17)
        mine_17.append((model.score(X_test_17, y_test_17), model.oob_score_))
    seconds_17 = time.perf_counter() - start_17
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # scikit-learn warns when a penguin is in every bootstrap
        for seed in range(5):
            model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=25, oob_score=True, random_state=seed).fit(X_17, y_17)
            oracle_17.append((model.score(X_test_17, y_test_17), model.oob_score_))
    (acc_17, oob_17), (sk_acc_17, sk_oob_17) = np.mean(mine_17, axis=0), np.mean(oracle_17, axis=0)
    print(f"5 baggings of 25 trees (2 measures of the penguins), mean over random_state 0 to 4:\n"
          f"  yours:        test accuracy {acc_17:.3f}, oob_score_ {oob_17:.3f} ({seconds_17:.1f} s for the 5)\n"
          f"  scikit-learn: test accuracy {sk_acc_17:.3f}, oob_score_ {sk_oob_17:.3f}")
    verdict("14.17", abs(acc_17 - sk_acc_17) < 0.03 and abs(oob_17 - sk_oob_17) < 0.05,
            "ton bagging est à moins de 3 points de scikit-learn en test, et ton score out-of-bag à moins de 5 points.",
            "ton bagging s'écarte de scikit-learn de plus de 3 points en test, ou ton score out-of-bag de plus de 5 : "
            "les tests ci-dessous disent où.")
with checked("14.17"):
    template_17 = mylearn.tree.DecisionTreeClassifier(max_depth=3)
    bag_17 = fitted(mylearn.ensemble.BaggingClassifier(template_17, n_estimators=10, random_state=1), X_17, y_17)
    again_17 = fitted(mylearn.ensemble.BaggingClassifier(template_17, n_estimators=10, random_state=1), X_17, y_17)
    samples_17 = [np.asarray(rows) for rows in bag_17.estimators_samples_]
    checks_17 = {
        "10 membres, chacun avec ses 233 indices tirés avec remise": len(bag_17.estimators_) == 10
            and all(rows.shape == (233,) and len(np.unique(rows)) < 233 for rows in samples_17),
        "des échantillons différents d'un membre à l'autre": len({tuple(rows) for rows in samples_17}) == 10,
        "le gabarit n'est jamais entraîné lui-même (des clones)": getattr(template_17, "root_", None) is None
            and all(member is not template_17 for member in bag_17.estimators_),
        "les mêmes probabilités pour le même random_state": np.allclose(bag_17.predict_proba(X_test_17),
                                                                       again_17.predict_proba(X_test_17)),
    }
    for what, ok in checks_17.items():
        verdict("14.17", ok, what + ".", "attendu : " + what + " ; les tests ci-dessous disent où.")
if written(lambda: mylearn.ensemble.BaggingClassifier(n_estimators=1).fit(np.array([[0.0], [1.0]]), np.array([0, 1]))):
    run_mylearn_tests("ensemble", keyword="test_bagging_classifier_")
else:
    print("⏳ Ex 14.17 : les tests attendent le fit de ton BaggingClassifier.")'''

# ---- 14.18 -----------------------------------------------------------------
CHECK_18 = r'''with checked("14.18"):
    X_doc_18 = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    y_doc_18 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    forest_doc_18 = fitted(mylearn.ensemble.RandomForestClassifier(n_estimators=5, random_state=0), X_doc_18, y_doc_18)
    extra_doc_18 = fitted(mylearn.ensemble.ExtraTreesClassifier(n_estimators=5, random_state=0), X_doc_18, y_doc_18)
    print("docstring examples: forest", np.asarray(forest_doc_18.predict(np.array([[1.5], [11.5]]))).tolist(),
          "importances", np.round(np.asarray(forest_doc_18.feature_importances_, dtype=float), 4).tolist(), "· extra",
          np.asarray(extra_doc_18.predict(np.array([[1.5], [11.5]]))).tolist(), "(expected [0, 1], [1.0] and [0, 1])")
with checked("14.18"):
    X_18, y_18 = wb.synth.make_moons(n=1300, noise=0.3, seed=1418)
    X_18 = np.hstack([X_18, np.random.default_rng(1418).normal(size=(1300, 3))])   # 2 useful features, 3 of noise
    X_train_18, y_train_18, X_test_18, y_test_18 = X_18[:300], y_18[:300], X_18[300:], y_18[300:]
    pairs_18 = [("RandomForestClassifier", mylearn.ensemble.RandomForestClassifier, RandomForestClassifier),
                ("ExtraTreesClassifier", mylearn.ensemble.ExtraTreesClassifier, ExtraTreesClassifier)]
    for name, mine_class, oracle_class in pairs_18:
        start_18 = time.perf_counter()
        mine_18 = [fitted(mine_class(n_estimators=20, random_state=seed), X_train_18, y_train_18) for seed in range(3)]
        seconds_18 = time.perf_counter() - start_18
        oracles_18 = [oracle_class(n_estimators=20, random_state=seed).fit(X_train_18, y_train_18) for seed in range(3)]
        acc_18 = np.mean([model.score(X_test_18, y_test_18) for model in mine_18])
        sk_acc_18 = np.mean([model.score(X_test_18, y_test_18) for model in oracles_18])
        importances_18 = np.mean([np.asarray(model.feature_importances_, dtype=float) for model in mine_18], axis=0)
        print(f"{name}, 20 trees, mean over random_state 0 to 2: yours {acc_18:.3f} ({seconds_18:.1f} s), "
              f"scikit-learn {sk_acc_18:.3f}; your importances {np.round(importances_18, 3).tolist()}")
        verdict("14.18", abs(acc_18 - sk_acc_18) < 0.03,
                f"{name} : à moins de 3 points de scikit-learn.",
                f"{name} : plus de 3 points d'écart avec scikit-learn ; les tests ci-dessous disent où.")
        verdict("14.18", importances_18.shape == (5,) and set(np.argsort(importances_18)[-2:]) == {0, 1}
                and abs(importances_18.sum() - 1) < 1e-6,
                f"{name} : les deux features utiles (0 et 1) ont les plus grandes importances, qui somment à 1.",
                f"{name} : attendu des importances de forme (5,), qui somment à 1, les plus grandes pour les features 0 "
                "et 1 (les trois autres sont du bruit).")
    trees_18 = {"forest": fitted(mylearn.ensemble.RandomForestClassifier(n_estimators=3, max_depth=4, random_state=0),
                                 X_train_18, y_train_18).estimators_[0],
                "extra": fitted(mylearn.ensemble.ExtraTreesClassifier(n_estimators=3, max_depth=4, random_state=0),
                                X_train_18, y_train_18).estimators_[0]}
    verdict("14.18", getattr(trees_18["forest"], "max_features", None) == "sqrt"
            and getattr(trees_18["forest"], "max_depth", None) == 4 and getattr(trees_18["extra"], "splitter", None) == "random",
            "les arbres reçoivent les hyperparamètres de l'ensemble (max_features=\"sqrt\", max_depth, et "
            "splitter=\"random\" pour ExtraTrees).",
            "chaque arbre doit recevoir max_depth, min_samples_split, min_samples_leaf et max_features de l'ensemble, "
            "et splitter=\"random\" pour ExtraTrees.")
if written(lambda: mylearn.ensemble.RandomForestClassifier(n_estimators=1).fit(np.array([[0.0], [1.0]]), np.array([0, 1]))):
    run_mylearn_tests("ensemble", keyword="test_random_forest_classifier_ or test_extra_trees_classifier_")
else:
    print("⏳ Ex 14.18 : les tests attendent le fit de ta RandomForestClassifier.")'''

# ---- 14.19 -----------------------------------------------------------------
GIVEN_19 = r'''rng_19 = np.random.default_rng(1419)
DATA_19 = pd.concat([pd.DataFrame({"row_id": rng_19.permutation(len(CAL)).astype(float),   # a distinct number per
                                   "coin": rng_19.integers(0, 2, len(CAL)).astype(float)}),  # district; 0 or 1 at random:
                     CAL[CAL_FEATURES]], axis=1)                                             # both unrelated to y
FEATURES_19 = list(DATA_19.columns)
rows_train_19, _ = train_test_split(np.arange(len(DATA_19)), test_size=0.2, random_state=42)   # the split of part B
X_TRAIN_19, Y_TRAIN_19 = DATA_19.iloc[rows_train_19[:5000]], Y_CAL[rows_train_19[:5000]]
X_VAL_19, Y_VAL_19 = DATA_19.iloc[rows_train_19[5000:7000]], Y_CAL[rows_train_19[5000:7000]]
print(FEATURES_19)'''

TODO_19 = r'''forest_19 = ...      # RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1) fitted on X_TRAIN_19, Y_TRAIN_19
mdi_19 = ...         # pandas Series: forest_19.feature_importances_, indexed by FEATURES_19, sorted in decreasing order
perm_19 = ...        # pandas Series: the mean permutation importances on X_VAL_19, Y_VAL_19
                     #   (permutation_importance(forest_19, X_VAL_19, Y_VAL_19, n_repeats=5, random_state=0, n_jobs=-1)),
                     #   indexed by FEATURES_19, sorted in decreasing order
rank_id_19 = ...     # a) the rank of "row_id" in mdi_19 (1 = the most important)
ratio_19 = ...       # b) the MDI importance of "row_id" divided by that of "coin"
second_mdi_19 = ...  # c) the name of the second most important feature in mdi_19
second_perm_19 = ... # d) the name of the second most important feature in perm_19
id_train_19 = ...    # e) the mean permutation importance of "row_id" on the TRAINING districts (same call on X_TRAIN_19)
joint_19 = ...       # f) the mean drop of R² on X_VAL_19 when Latitude and Longitude are permuted TOGETHER (see the statement)'''

SOLUTION_19 = r'''forest_19 = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1).fit(X_TRAIN_19, Y_TRAIN_19)
mdi_19 = pd.Series(forest_19.feature_importances_, index=FEATURES_19).sort_values(ascending=False)
on_val_19 = permutation_importance(forest_19, X_VAL_19, Y_VAL_19, n_repeats=5, random_state=0, n_jobs=-1)
perm_19 = pd.Series(on_val_19.importances_mean, index=FEATURES_19).sort_values(ascending=False)
rank_id_19 = list(mdi_19.index).index("row_id") + 1
ratio_19 = mdi_19["row_id"] / mdi_19["coin"]
second_mdi_19, second_perm_19 = mdi_19.index[1], perm_19.index[1]
on_train_19 = permutation_importance(forest_19, X_TRAIN_19, Y_TRAIN_19, n_repeats=5, random_state=0, n_jobs=-1)
id_train_19 = on_train_19.importances_mean[FEATURES_19.index("row_id")]
base_19 = forest_19.score(X_VAL_19, Y_VAL_19)
drops_19 = []
for r in range(5):
    order = np.random.default_rng(r).permutation(len(X_VAL_19))
    shuffled = X_VAL_19.copy()
    shuffled[["Latitude", "Longitude"]] = X_VAL_19[["Latitude", "Longitude"]].to_numpy()[order]
    drops_19.append(base_19 - forest_19.score(shuffled, Y_VAL_19))
joint_19 = float(np.mean(drops_19))
'''

CHECK_19 = r'''wb.check("14.19a", rank_id_19)
wb.check("14.19b", ratio_19)
wb.check("14.19c", second_mdi_19)
wb.check("14.19d", second_perm_19)
wb.check("14.19e", id_train_19)
wb.check("14.19f", joint_19)
with wb.attempt("14.19"):
    if not filled(mdi_19, perm_19):
        raise NotImplementedError("mdi_19, perm_19")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pd.Series(mdi_19).sort_values().plot.barh(ax=axes[0], title="MDI (training)")
    pd.Series(perm_19).sort_values().plot.barh(ax=axes[1], title="permutation (validation, drop of R²)")
    plt.tight_layout()
    plt.show()'''

PART_B = Part("B", "Bagging, forêts et ExtraTrees",
              "Les forêts de scikit-learn sur California (14.15), le nombre d'arbres face à des colonnes de bruit "
              "(14.16), ton `BaggingClassifier` (14.17), ta forêt et tes ExtraTrees (14.18), puis l'importance des "
              "features (14.19). La cellule suivante charge California et la découpe comme le défi 14.28 : 16 512 "
              "districts d'entraînement, 4 128 de test (`random_state=42`).",
              given=PART_B_GIVEN, exercises=[
    Ex("14.15", "📦", 2, 25, "Forêts scikit-learn sur California : OOB et parallélisme",
       "utiliser une forêt de régression, son score out-of-bag et son parallélisme, et savoir ce qu'ils valent.",
       "Ex 14.11 · ch. 13 (arbres de régression) · ch. 9 (R²) · fiche §14.4 et §14.5", thread="California",
       tracks="R, C",
       body=r"""`RandomForestRegressor` de scikit-learn sur 5 000 districts d'entraînement (`X_TRAIN_15`). En régression, `oob_score_` est un $R^2$ (ch. 9) calculé sur les exemples out-of-bag, et `score` rend aussi un $R^2$. Calcule :
a) `oob_15` : le `oob_score_` de `RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0, n_jobs=-1)` ;
b) `test_15` : son $R^2$ sur les 4 128 districts de test ;
c) `oob_curve_15` : la liste des `oob_score_` de la même forêt pour chaque nombre d'arbres de `NS_15` (10 à 200) ;
d) `first_close_15` : le plus petit nombre d'arbres de `NS_15` dont le `oob_score_` est à moins de 0,01 du $R^2$ de test **de la même forêt**.

Avec 10 arbres, scikit-learn prévient que quelques districts n'ont aucun arbre qui ne les a pas vus : c'est normal (tu peux masquer l'avertissement avec `warnings.catch_warnings()`). La vérification contrôle a) à d), puis entraîne la forêt sur un cœur (`n_jobs=1`) et sur tous (`n_jobs=-1`), compare les durées et les prédictions.

Dans tes notes : pourquoi le score out-of-bag est-il pessimiste avec peu d'arbres ? Combien d'arbres prédisent un district out-of-bag, en moyenne, dans une forêt de 10 arbres ? Le gain de `n_jobs=-1` dépend de quoi ? Les prédictions sont-elles identiques au bit près, et pourquoi pourraient-elles ne pas l'être ?""",
       given=GIVEN_15, todo=TODO_15, check=CHECK_15, solution=SOLUTION_15 + as_printed(CHECK_15, "14.15"),
       record=r'''wb.record("14.15a", oob_15, decimals=4,
          mistakes={"relis la question : on demande oob_score_, calculé sur les districts d'entraînement out-of-bag": test_15})
wb.record("14.15b", test_15, decimals=4,
          mistakes={"relis la question : on demande le R² (score) sur X_CAL_TEST, Y_CAL_TEST": oob_15})
wb.record("14.15c", oob_curve_15, decimals=4,
          mistakes={"relis la question : on demande les oob_score_ des forêts, pas leur score sur le test": test_curve_15})
wb.record("14.15d", first_close_15, choices=NS_15,
          fractional="un nombre non entier : est-ce un R² ? on demande un nombre d'arbres de NS_15",
          mistakes={f"{n} arbres : l'écart entre oob_score_ et le R² de test de la même forêt est-il sous 0,01 ? et est-ce le premier n de NS_15 qui y parvient ?": n
                    for n in NS_15 if n != first_close_15})''',
       note="Avec 100 arbres, le $R^2$ out-of-bag vaut 0,777 et celui de test 0,771 : l'estimation gratuite est "
            "juste. Avec 10 arbres, elle est très pessimiste (0,668 contre 0,748 en test) : chaque district n'est "
            "prédit que par les arbres qui ne l'ont pas vu, en moyenne $10 \\times 0{,}368 \\approx 3{,}7$ arbres, une "
            "forêt bien plus petite que celle qu'on évalue. L'écart passe sous 0,01 à partir de 50 arbres. Sur 2 "
            "cœurs, `n_jobs=-1` divise le temps par 1,5 à 2 ; le gain dépend du nombre de cœurs et de la taille du "
            "travail (au-delà de quelques centaines d'exemples par arbre). Les prédictions sont identiques à "
            "environ $10^{-15}$ près, pas forcément au bit près : en parallèle, les prédictions des arbres s'additionnent "
            "dans l'ordre où les cœurs finissent, et l'addition de nombres flottants dépend de l'ordre. C'est "
            "pourquoi un benchmark note le nombre de cœurs (🛠️ 14.27)."),

    Ex("14.16", "🔬", 2, 30, "Combien d'arbres ? Bagging, forêt et ExtraTrees face au bruit",
       "mesurer ce que le nombre d'arbres apporte à trois ensembles, et ce que des features de bruit leur coûtent.",
       "Ex 14.15 · ch. 7 (features de bruit) · fiche §14.4 à §14.6", thread="synthétique", tracks="C",
       body=r"""Deux demi-lunes bruitées (500 points d'entraînement, 3 000 de test), puis les mêmes avec **30 colonnes de bruit** (`moons_16(n_noise)`). Trois ensembles de scikit-learn (`ENSEMBLES_16`, chacun appelé avec son nombre d'arbres) : un bagging d'arbres sans limite, une forêt aléatoire (`max_features="sqrt"`) et des ExtraTrees, tous avec `random_state=0`. Écris `curves_16(n_noise)`, qui rend, pour chaque ensemble, la liste de ses accuracies de test pour chaque nombre d'arbres de `NS_16` (1 à 400), puis :
a) `noisy_16` : `curves_16(30)` (la vérification lit la courbe d'ExtraTrees) ; et `clean_16` : `curves_16(0)` ;
b) `worst_16` : l'ensemble qui a la plus petite accuracy avec 400 arbres et 30 colonnes de bruit : `"A"` bagging, `"B"` forêt, `"C"` ExtraTrees ;
c) `gain_extra_16` : pour ExtraTrees avec 30 colonnes de bruit, l'accuracy à 400 arbres moins celle à 25 arbres ;
d) `drop_16` : `True` si l'une des six courbes perd plus de 0,005 entre 100 et 400 arbres, `False` sinon.

Ta cellule entraîne 54 ensembles (compte une quinzaine de secondes) ; la vérification trace les deux familles de courbes.

Dans tes notes : avec 30 colonnes de bruit, combien de features utiles un nœud de la forêt voit-il, en moyenne, parmi les 5 qu'il tire ? Pourquoi les seuils tirés au hasard d'ExtraTrees souffrent-ils encore plus du bruit ? Pourquoi 2 arbres font-ils souvent moins bien qu'un seul ? Combien d'arbres mettrais-tu ?""",
       given=GIVEN_16, todo=TODO_16, check=CHECK_16, solution=SOLUTION_16 + as_printed(CHECK_16, "14.16"),
       record=r'''wb.record("14.16a", noisy_16["extra"], decimals=3,
          mistakes={"c'est la courbe sans colonne de bruit : on demande curves_16(30)": clean_16["extra"],
                    "c'est la courbe de la forêt : on demande celle d'ExtraTrees": noisy_16["forest"]})
wb.record("14.16b", worst_16,
          mistakes={"relis la dernière valeur de chaque courbe de noisy_16 : laquelle est la plus petite ?": "A",
                    "relis la dernière valeur de chaque courbe de noisy_16, ExtraTrees compris : laquelle est la plus petite ?": "B"})
wb.record("14.16c", gain_extra_16, decimals=3,
          mistakes={"c'est la valeur avec 400 arbres : on demande la différence avec 25 arbres": noisy_16["extra"][-1],
                    "le signe : 400 arbres moins 25 arbres": -gain_extra_16})
wb.record("14.16d", drop_16, mistakes={"compare, pour chacune des six courbes, la valeur à 100 arbres et celle à 400 : laquelle perd plus de 0,005 ?": True})''',
       note="Sans bruit, les trois ensembles plafonnent vers 0,86 dès 25 à 50 arbres. Avec 30 colonnes de bruit, le "
            "bagging (qui examine les 32 features à chaque nœud) garde 0,865 ; la forêt tire 5 features par nœud, "
            "dont $5 \\times 2/32 \\approx 0{,}3$ utile en moyenne : beaucoup de nœuds coupent sur du bruit, et elle "
            "descend à 0,852 ; ExtraTrees, dont les seuils sont aussi tirés au hasard, finit à 0,840 et a besoin de "
            "bien plus d'arbres (+0,028 entre 25 et 400). Aucune courbe ne perd plus de 0,004 entre 100 et 400 "
            "arbres : plus d'arbres ne fait pas surapprendre, mais au-delà d'une centaine on ne gagne plus rien. "
            "Avec 2 arbres, chaque désaccord donne une probabilité de 0,5 pour chaque classe, et l'égalité donne la "
            "première classe : la moitié des désaccords sont perdus, d'où un creux à 2 arbres. Avec d'autres "
            "tirages des données (graines 1 à 3 de `make_moons` et du bruit), ExtraTrees reste le dernier avec 30 colonnes de bruit (de 0,850 à 0,855 contre 0,859 "
            "à 0,871 pour les deux autres)."),

    Ex("14.17", "🔨", 3, 50, "Coder BaggingClassifier avec score out-of-bag",
       "écrire un bagging générique : des clones entraînés sur leurs bootstraps, un vote souple et un score out-of-bag.",
       "Ex 14.12 · Ex 13.25 · ch. 8 (`clone`) · fiche §14.4 (pseudo-code)", thread="Penguins", tracks="C",
       mylearn="ensemble.py",
       body=mylearn_howto() + r"""

Écris la classe `BaggingClassifier`. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__` range ses six arguments tels quels ; `fit(X, y)` vérifie, apprend `classes_` (les labels distincts triés), `estimators_`, `estimators_samples_` (et `oob_score_` si `oob_score=True`), puis renvoie `self` ; un second `fit` du même objet redonne le même ensemble ;
- `fit` crée **un** générateur à partir de `random_state` (`np.random.default_rng(random_state)`), d'où vient tout le hasard de l'ensemble ; chaque membre a son propre échantillon d'indices, de la taille que fixe `max_samples` (un bootstrap, avec `bootstrap_indices`, si `bootstrap=True` ; sinon des indices **distincts**), est un `clone(gabarit)` neuf (le gabarit est `estimator`, ou `DecisionTreeClassifier()` de **ton** `tree.py` si `estimator` vaut `None` ; il n'est jamais entraîné lui-même), reçoit, s'il a un attribut `random_state`, un **entier** tiré de ce générateur, et apprend sur `X[indices]`, `y[indices]` ;
- `max_samples` : un entier de 1 à $n$, ou un réel $f$ dans $]0, 1]$ (`max(1, int(f * n))` tirages) ; `ValueError` sinon, ainsi que pour `n_estimators < 1`, `oob_score=True` avec `bootstrap=False`, un `X` qui n'est pas 2-D ou contient `NaN`, ou `len(X) != len(y)` ;
- `predict_proba(X)` : la moyenne des `predict_proba` des membres, **alignés sur `classes_`** (un membre qui n'a vu que certaines classes donne une probabilité nulle aux autres) ; `predict(X)` : la classe de plus grande probabilité moyenne ; `score(X, y)` : l'accuracy, un `float` Python ; `RuntimeError` si `predict_proba`, `predict` ou `score` est appelée avant `fit` ;
- `oob_score_` : chaque exemple d'entraînement est prédit par la moyenne des probabilités des **seuls** membres qui ne l'ont pas vu ; les exemples vus par tous sont laissés de côté ; `ValueError` si aucun exemple n'est out-of-bag.

La vérification essaie l'exemple de la docstring, compare ton bagging (25 arbres, 5 graines) à celui de scikit-learn sur deux mesures des manchots, contrôle les échantillons, les clones et la reproductibilité, puis lance les tests.

Dans tes notes : pourquoi un clone neuf par membre, plutôt que le gabarit réentraîné 25 fois ? Pourquoi le vote souple doit-il aligner les colonnes sur `classes_` ? Que coûte le score out-of-bag ?""",
       check=RELOAD + CHECK_17, solution=ref_check(CHECK_17, "14.17"),
       note="Un modèle réentraîné oublie son entraînement précédent : 25 `fit` du même objet laisseraient 25 "
            "références au **même** arbre, le dernier. `clone` crée un objet neuf, non entraîné, aux mêmes "
            "hyperparamètres. Un membre dont le bootstrap ne contient aucun Chinstrap rend un `predict_proba` à deux "
            "colonnes : sans alignement sur `classes_`, la moyenne additionnerait les probabilités de deux espèces "
            "différentes. Le score out-of-bag ne demande aucun entraînement de plus, seulement une prédiction de "
            "chaque membre sur ses exemples out-of-bag (environ 37 % des exemples d'entraînement)."),

    Ex("14.18", "🔨", 3, 45, "Coder RandomForest et ExtraTrees",
       "construire une forêt aléatoire et des ExtraTrees sur ton bagging et tes arbres, avec leurs importances.",
       "Ex 14.17 · Ex 13.25 (`max_features`, `splitter`) · fiche §14.5 et §14.6", thread="synthétique", tracks="C",
       mylearn="ensemble.py",
       body=r"""> **mylearn** : dans `mon_travail/mylearn/ensemble.py`, mêmes règles qu'en 14.17. Enregistre, puis relance la cellule de vérification.

Écris les classes `RandomForestClassifier` et `ExtraTreesClassifier`. **Contrat** (les docstrings, plus ces précisions que vérifient les tests) :
- chaque arbre est un `DecisionTreeClassifier` de **ton** `tree.py`, avec les `max_depth`, `min_samples_split`, `min_samples_leaf` et `max_features` de l'ensemble, plus `splitter="random"` pour ExtraTrees ; il apprend sur un bootstrap de $n$ tirages (`bootstrap=True`) ou sur tout l'entraînement (`bootstrap=False`), et reçoit un `random_state` entier, différent d'un arbre à l'autre, tiré d'un générateur créé dans `fit` à partir du `random_state` de l'ensemble (comme les membres de ton bagging) ; `__init__` range ses arguments tels quels, et `fit` n'ajoute que des attributs appris (un nom qui finit par `_`) ou privés (un nom qui commence par `_`) ;
- après `fit` : `estimators_`, `classes_`, `feature_importances_` (et `oob_score_` pour la forêt si `oob_score=True`) ; `predict_proba`, `predict` et `score` comme le bagging (vote souple, `RuntimeError` avant `fit`) ; `ValueError` pour un hyperparamètre invalide, `NaN`, `len(X) != len(y)`, ou `oob_score=True` avec `bootstrap=False` ;
- `feature_importances_` : la moyenne des `feature_importances_` des arbres **qui coupent au moins une fois**, renormalisée pour sommer à 1 (des zéros si aucun arbre ne coupe), comme scikit-learn : un arbre réduit à une feuille ne donne que des zéros, qui ne doivent pas tirer la moyenne vers le bas ; c'est ce que demande l'exemple de la docstring (`array([1.])`), que la seule moyenne ne donne pas.

La vérification essaie les exemples des docstrings, compare tes deux ensembles (20 arbres, 3 graines) à ceux de scikit-learn sur des demi-lunes avec 3 colonnes de bruit (accuracy et importances), vérifie les hyperparamètres des arbres, puis lance les tests (les tests statistiques entraînent beaucoup de forêts : compte une à deux minutes si ton arbre est lent).

Dans tes notes : combien de lignes ta forêt a-t-elle demandé, une fois le bagging écrit ? Lequel de tes deux ensembles s'entraîne le plus vite, et pourquoi ?""",
       check=RELOAD + CHECK_18, solution=ref_check(CHECK_18, "14.18"),
       note="Une fois le bagging écrit, une forêt tient en quelques lignes : un gabarit `DecisionTreeClassifier("
            "max_features=..., ...)`, un `BaggingClassifier` de ce gabarit, puis la moyenne des importances. Ici, avec "
            "l'arbre de la référence, ExtraTrees s'entraîne plus lentement que la forêt (2 à 3 fois) : sans bootstrap "
            "et avec des seuils tirés, ses arbres sont bien plus profonds (environ 19 niveaux et 99 feuilles, contre "
            "11 et 33). Avec un arbre « naïf », qui essaie tous les seuils sans trier, un seuil tiré fait au contraire "
            "gagner beaucoup de temps (la fiche, §14.6, dit que cela dépend des données ; cela dépend aussi de l'arbre)."),

    Ex("14.19", "📦", 3, 35, "Importance des features : impureté contre permutation",
       "comparer l'importance MDI d'une forêt à l'importance par permutation, et voir ce qui trompe la première.",
       "Ex 14.15 · ch. 12 (sélection de features) · fiche §14.5 (importance des features)", thread="California",
       tracks="C",
       body=r"""California, avec deux colonnes **sans aucun lien** avec la cible : `row_id`, un numéro différent pour chaque district, et `coin`, 0 ou 1 au hasard. Une forêt de 100 arbres apprend sur 5 000 districts (`X_TRAIN_19`, un DataFrame) ; 2 000 autres districts d'entraînement servent de validation (`X_VAL_19`). Calcule `forest_19`, `mdi_19` (son `feature_importances_`) et `perm_19` (l'importance par permutation sur la validation, avec `permutation_importance(..., n_repeats=5, random_state=0, n_jobs=-1)`, attribut `importances_mean`), deux `pandas.Series` indexées par `FEATURES_19` et triées par ordre décroissant, puis :
a) `rank_id_19` : le rang de `row_id` dans `mdi_19` (1 pour la plus importante) ;
b) `ratio_19` : l'importance MDI de `row_id` divisée par celle de `coin` ;
c) `second_mdi_19` : le nom de la deuxième feature de `mdi_19` ;
d) `second_perm_19` : le nom de la deuxième feature de `perm_19` ;
e) `id_train_19` : l'importance par permutation de `row_id` mesurée sur les districts d'**entraînement** (le même appel sur `X_TRAIN_19`, `Y_TRAIN_19`) ;
f) `joint_19` : la baisse moyenne du $R^2$ de validation quand on permute **ensemble** `Latitude` et `Longitude` : pour `r` de 0 à 4, une permutation `np.random.default_rng(r).permutation(len(X_VAL_19))`, appliquée aux deux colonnes d'une copie de `X_VAL_19` ; la baisse est le $R^2$ de validation (`forest_19.score`) moins celui de la copie.

La vérification contrôle a) à f), puis trace les deux importances.

Dans tes notes : pourquoi `row_id` pèse-t-il tant plus que `coin` dans l'importance MDI ? Que dit e) sur ce que la forêt a appris de `row_id` ? Compare f) à la somme des importances par permutation de `Latitude` et de `Longitude` : pourquoi n'est-ce pas la même chose ?""",
       given=GIVEN_19, todo=TODO_19, check=CHECK_19, solution=SOLUTION_19 + as_printed(CHECK_19, "14.19"),
       record=r'''wb.record("14.19a", rank_id_19, choices=list(range(1, 11)),
          fractional="un nombre non entier : est-ce une importance ? on demande un rang, de 1 à 10",
          mistakes={"le rang se compte à partir de 1 (la plus importante) : as-tu pris la position Python, qui commence à 0 ?": rank_id_19 - 1,
                    "le rang se compte depuis la feature la plus importante : as-tu trié dans l'ordre décroissant ?": 11 - rank_id_19,
                    **{f"rang {k} : relis mdi_19, triée par ordre décroissant ; à quelle position y est row_id (1 pour la première) ?": k
                       for k in range(1, 11) if k not in (rank_id_19, rank_id_19 - 1, 11 - rank_id_19)}})
wb.record("14.19b", ratio_19, decimals=1,
          mistakes={"c'est le rapport inverse : importance de row_id divisée par celle de coin": 1 / ratio_19,
                    "relis la question : le rapport des importances MDI (mdi_19), pas des importances par permutation": float(perm_19["row_id"] / perm_19["coin"])})
wb.record("14.19c", second_mdi_19,
          mistakes={"c'est la feature la plus importante : on demande la deuxième de mdi_19": mdi_19.index[0],
                    "relis la question : on demande la deuxième feature de mdi_19 (l'importance MDI)": perm_19.index[1]})
wb.record("14.19d", second_perm_19,
          mistakes={"c'est la feature la plus importante : on demande la deuxième de perm_19": perm_19.index[0],
                    "relis la question : on demande la deuxième feature de perm_19 (l'importance par permutation)": mdi_19.index[1]})
id_val_19 = float(perm_19["row_id"])
wb.record("14.19e", id_train_19, decimals=3,
          mistakes={"relis la question : on demande l'importance par permutation mesurée sur les districts d'entraînement (X_TRAIN_19)": id_val_19})
def drop_19(shuffle):
    """Mean drop of the validation R² over r = 0..4, when shuffle(copy, r) permutes the two columns of the copy."""
    drops = []
    for r in range(5):
        shuffled = X_VAL_19.copy()
        shuffle(shuffled, r)
        drops.append(base_19 - forest_19.score(shuffled, Y_VAL_19))
    return float(np.mean(drops))


def independent_19(copy, r):
    copy["Latitude"] = X_VAL_19["Latitude"].to_numpy()[np.random.default_rng(r).permutation(len(copy))]
    copy["Longitude"] = X_VAL_19["Longitude"].to_numpy()[np.random.default_rng(100 + r).permutation(len(copy))]


def aligned_19(copy, r):                    # pandas puts the rows back in their place: nothing moves
    copy[["Latitude", "Longitude"]] = X_VAL_19[["Latitude", "Longitude"]].iloc[np.random.default_rng(r).permutation(len(copy))]


wb.record("14.19f", joint_19, decimals=2,
          mistakes={"c'est la somme des deux importances par permutation : on demande la baisse quand les deux colonnes sont permutées ensemble, par la même permutation": float(perm_19["Latitude"] + perm_19["Longitude"]),
                    "c'est le R² de la copie permutée : on demande la baisse, le R² de validation d'origine moins celui de la copie": base_19 - joint_19,
                    "les deux colonnes doivent recevoir la même permutation (le même order) : as-tu tiré une permutation par colonne ?": drop_19(independent_19),
                    "rien n'a bougé : en affectant un DataFrame, pandas aligne les lignes sur leur index ; prends les valeurs permutées avec .to_numpy()": drop_19(aligned_19)})''',
       note="L'importance MDI place `row_id` au 9ᵉ rang (sur 10), environ 8 fois au-dessus de `coin`, deux colonnes aussi "
            "inutiles l'une que l'autre : `row_id` offre des milliers de seuils, et un arbre profond finit toujours "
            "par en trouver un qui sépare quelques districts dans un petit nœud ; `coin` n'en offre qu'un. Les deux "
            "importances ne s'accordent pas sur l'ordre : la MDI met `AveOccup` en deuxième, la permutation "
            "`Latitude`. Sur l'entraînement, permuter `row_id` fait perdre 0,016 de $R^2$ : la forêt s'en sert "
            "pour retenir des districts par cœur ; sur la validation, presque rien (0,0015). Permuter `Latitude` et "
            "`Longitude` ensemble fait perdre 0,23 de $R^2$, **moins** que la somme de leurs deux importances "
            "(0,32 + 0,23 = 0,55) : permutées ensemble, les deux colonnes déplacent chaque district vers la position "
            "d'un **autre** district, un lieu qui existe ; permutée seule, une latitude prise ailleurs crée des "
            "positions impossibles (en mer, dans le désert), où les arbres prédisent n'importe quoi, et l'importance "
            "de chacune est gonflée : deux features complémentaires (l'une ne dit rien sans l'autre) paraissent "
            "trop importantes, comme deux features redondantes paraissent trop peu importantes (fiche §14.5). Dans "
            "les deux cas, on permute le groupe entier."),
])

# ---- Part C ----------------------------------------------------------------
PART_C_GIVEN = r'''# 3,000 training districts to fit the boostings, 2,000 others to validate them (taken from the training split)
X_FIT_C, Y_FIT_C = X_CAL_TRAIN[:3000], Y_CAL_TRAIN[:3000]
X_VAL_C, Y_VAL_C = X_CAL_TRAIN[3000:5000], Y_CAL_TRAIN[3000:5000]


def boosting_curves(**params):
    """Fit GradientBoostingRegressor(random_state=0, **params) on X_FIT_C; return the training RMSE and the validation
    RMSE after each tree (two arrays)."""
    model = GradientBoostingRegressor(random_state=0, **params).fit(X_FIT_C, Y_FIT_C)
    validation = np.array([rmse(Y_VAL_C, predicted) for predicted in model.staged_predict(X_VAL_C)])
    return np.sqrt(model.train_score_), validation'''

# ---- 14.20 -----------------------------------------------------------------
GIVEN_20 = r'''ORDER_20 = np.random.default_rng(1421).permutation([1, 3, 8])   # the depths of panels A, B and C, in a hidden order
CURVES_20 = {panel: boosting_curves(max_depth=int(depth), learning_rate=0.3, n_estimators=400)
             for panel, depth in zip("ABC", ORDER_20)}                 # panel -> (training RMSE, validation RMSE)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
for ax, (panel, (train, validation)) in zip(axes, CURVES_20.items()):
    ax.plot(np.arange(1, 401), train, label="training")
    ax.plot(np.arange(1, 401), validation, label="validation")
    ax.set_xlabel("number of trees")
    ax.set_title(f"panel {panel}")
axes[0].set_ylabel("RMSE")
axes[0].legend()
plt.show()
del ORDER_20                         # the answer is not kept in the notebook'''

TODO_20 = r'''overfit_20 = ...      # a) the panel ("A", "B" or "C") whose model learns its training set by heart
underfit_20 = ...     # b) the panel whose model still underfits after 400 trees
stop_20 = ...         # c) for the remaining panel (neither a) nor b)), where is the minimum of the validation curve?
                      #    "A" before 50 trees · "B" from 50 to 300 trees · "C" from 301 to 400 trees ·
                      #    "D" the validation curve still goes down at 400 trees
best_trees_20 = ...   # d) list: for panels A, B and C, the number of trees (1 to 400) at the minimum of the validation
                      #    RMSE, computed from CURVES_20
gap_20 = ...          # e) for the panel of c): the validation RMSE after 400 trees minus its minimum'''

CHECK_20 = r'''wb.check("14.20a", overfit_20)
wb.check("14.20b", underfit_20)
wb.check("14.20c", stop_20)
wb.check("14.20d", best_trees_20)
wb.check("14.20e", gap_20)'''

# ---- 14.21 -----------------------------------------------------------------
GIVEN_21 = r'''LRS_21 = [0.02, 0.05, 0.1, 0.5, 1.0]   # learning rates, all with trees of depth 3 and up to 1,000 trees'''

EXPERIMENT_21 = r'''curves_21 = {lr: boosting_curves(max_depth=3, learning_rate=lr, n_estimators=1000) for lr in LRS_21}   # ≈ 40 s
print(" learning rate · trees at the minimum · best validation RMSE · after 1,000 trees · training RMSE after 1,000 trees")
for lr, (train, validation) in curves_21.items():
    print(f"{lr:14.2f} · {int(np.argmin(validation)) + 1:19d} · {validation.min():24.4f} · {validation[-1]:17.4f} · "
          f"{train[-1]:.4f}")
fig, ax = plt.subplots(figsize=(8, 4.5))
for lr, (train, validation) in curves_21.items():
    ax.plot(np.arange(1, 1001), validation, label=f"learning rate {lr}")
ax.set_xscale("log")
ax.set_ylim(0.5, 0.9)
ax.set_xlabel("number of trees")
ax.set_ylabel("validation RMSE")
ax.legend()
plt.show()
wb.check("14.21a", ratio_21)
wb.check("14.21b", gain_21)
wb.check("14.21c", slow_beats_21)'''

# ---- 14.22 -----------------------------------------------------------------
GIVEN_22 = r'''X_22, Y_22 = wb.synth.make_moons(n=300, noise=0.25, seed=1422)
ROUNDS_22 = [1, 3, 10, 50]
GRID1_22, GRID2_22 = np.meshgrid(np.linspace(-1.6, 2.6, 160), np.linspace(-1.2, 1.7, 110))
GRID_22 = np.column_stack([GRID1_22.ravel(), GRID2_22.ravel()])   # 17,600 points that cover the moons'''

TODO_22 = r'''model_22 = ...   # AdaBoostClassifier(n_estimators=50, random_state=0) fitted on X_22, Y_22 (its members are stumps)


def stage_predictions_22(model, X, rounds):
    """Return a dict {m: the predictions of the fitted AdaBoost `model` on X when it keeps only its first m members},
    for each m of `rounds` (use model.staged_predict, which yields the predictions after 1, 2, 3... members)."""
    raise NotImplementedError("stage_predictions_22")


def draw_22(model, X, y, rounds):
    """Draw 1 x len(rounds) panels: in panel m, the regions predicted after m members on GRID_22 (reshaped like
    GRID1_22, e.g. with ax.contourf), the training points X coloured by y, and a title with m and the training
    accuracy after m members. Return the figure."""
    raise NotImplementedError("draw_22")


train_acc_22 = ...   # list: the training accuracy of model_22 after 1, 3, 10 and 50 members'''

SOLUTION_22 = r'''model_22 = AdaBoostClassifier(n_estimators=50, random_state=0).fit(X_22, Y_22)


def stage_predictions_22(model, X, rounds):
    """Return a dict {m: the predictions of the fitted AdaBoost `model` on X when it keeps only its first m members},
    for each m of `rounds` (use model.staged_predict, which yields the predictions after 1, 2, 3... members)."""
    wanted = set(rounds)
    return {m: predicted for m, predicted in enumerate(model.staged_predict(X), start=1) if m in wanted}


def draw_22(model, X, y, rounds):
    """Draw 1 x len(rounds) panels: in panel m, the regions predicted after m members on GRID_22 (reshaped like
    GRID1_22, e.g. with ax.contourf), the training points X coloured by y, and a title with m and the training
    accuracy after m members. Return the figure."""
    on_grid = stage_predictions_22(model, GRID_22, rounds)
    on_train = stage_predictions_22(model, X, rounds)
    fig, axes = plt.subplots(1, len(rounds), figsize=(4.5 * len(rounds), 3.8), sharey=True)
    for ax, m in zip(axes, rounds):
        ax.contourf(GRID1_22, GRID2_22, on_grid[m].reshape(GRID1_22.shape), levels=[-0.5, 0.5, 1.5], alpha=0.3,
                    colors=["tab:blue", "tab:orange"])
        ax.scatter(*X[y == 0].T, s=10, c="tab:blue")
        ax.scatter(*X[y == 1].T, s=10, c="tab:orange")
        ax.set_title(f"{m} stump(s) · training accuracy {np.mean(on_train[m] == y):.3f}")
    return fig


train_acc_22 = [float(np.mean(stage_predictions_22(model_22, X_22, ROUNDS_22)[m] == Y_22)) for m in ROUNDS_22]
'''

CHECK_22 = r'''wb.check("14.22a", train_acc_22)
with wb.attempt("14.22"):
    if not filled(model_22):
        raise NotImplementedError("model_22")
    stages_22 = stage_predictions_22(model_22, GRID_22, ROUNDS_22)
    if returned("14.22", "stage_predictions_22", stages_22) and not (isinstance(stages_22, dict) and set(ROUNDS_22) <= set(stages_22)):
        print("❌ Ex 14.22 : stage_predictions_22 doit rendre un dictionnaire {m: prédictions} avec une clé pour chaque m "
              f"de ROUNDS_22 ({ROUNDS_22}).")
    elif stages_22 is not None:
        first_22 = np.asarray(stages_22[1]).reshape(GRID1_22.shape)
        one_cut_22 = (first_22 == first_22[:1, :]).all() or (first_22 == first_22[:, :1]).all()
        verdict("14.22", one_cut_22 and np.array_equal(np.asarray(stages_22[50]), model_22.predict(GRID_22)),
                "après une souche, une seule coupure parallèle à un axe ; après 50, les prédictions de model_22.",
                "attendu : après 1 membre, des régions coupées par une seule droite verticale ou horizontale (une "
                "souche), et après 50 membres les prédictions de model_22.predict ; relis staged_predict.")
        figure_22 = draw_22(model_22, X_22, Y_22, ROUNDS_22)
        if returned("14.22", "draw_22", figure_22):
            plt.show()'''

# ---- 14.23 -----------------------------------------------------------------
GIVEN_23 = r'''X_23, Y_23 = wb.synth.make_moons(n=400, noise=0.3, seed=1423)   # two classes, 0 and 1


def predict_23(stumps, alphas, X):
    """Weighted vote of the stumps (correct, nothing to fix): the class whose stumps have the largest sum of alphas."""
    votes = np.zeros((len(X), 2))
    for stump, alpha in zip(stumps, alphas):
        votes[np.arange(len(X)), stump.predict(X)] += alpha
    return np.argmax(votes, axis=1)'''

TODO_23 = r'''# A colleague's AdaBoost (SAMME, two classes 0 and 1): its accuracy does not improve, whatever the number of rounds.
# Find and fix its bugs (there are three), then rerun the check cell.
def adaboost_23(X, y, n_rounds=40):
    """AdaBoost with scikit-learn stumps. Return (stumps, alphas, weights), where weights[m] is the vector of sample
    weights (summing to 1) with which stump m was fitted."""
    n = len(X)
    w = np.full(n, 1 / n)
    stumps, alphas, weights = [], [], []
    for m in range(n_rounds):
        weights.append(w.copy())
        stump = DecisionTreeClassifier(max_depth=1, random_state=m)
        stump.fit(X, y)
        wrong = stump.predict(X) != y
        err = np.sum(w[wrong])
        alpha = np.log((1 - err) / err)
        w = w * np.exp(alpha * (stump.predict(X) == y))
        stumps.append(stump)
        alphas.append(alpha)
    return stumps, alphas, weights'''

SOLUTION_23 = r'''def adaboost_23(X, y, n_rounds=40):
    """AdaBoost with scikit-learn stumps. Return (stumps, alphas, weights), where weights[m] is the vector of sample
    weights (summing to 1) with which stump m was fitted."""
    n = len(X)
    w = np.full(n, 1 / n)
    stumps, alphas, weights = [], [], []
    for m in range(n_rounds):
        w = w / w.sum()                                  # fix 2: normalise, so that err is a weighted error rate
        weights.append(w.copy())
        stump = DecisionTreeClassifier(max_depth=1, random_state=m)
        stump.fit(X, y, sample_weight=w)                 # fix 3: the stump must see the weights
        wrong = stump.predict(X) != y
        err = np.sum(w[wrong])
        alpha = np.log((1 - err) / err)
        w = w * np.exp(alpha * wrong)                    # fix 1: the MISclassified samples weigh more
        stumps.append(stump)
        alphas.append(alpha)
    return stumps, alphas, weights
'''

CHECK_23 = r'''with checked("14.23"):
    try:
        with np.errstate(all="ignore"):
            result_23 = adaboost_23(X_23, Y_23)
        crash_23 = None
    except ValueError as exc:                            # NaN weights refused by scikit-learn
        if "nan" not in str(exc).lower() and "infinity" not in str(exc).lower():
            raise
        result_23, crash_23 = None, exc
    if crash_23 is not None:
        print("❌ Ex 14.23 : point 1 : les poids sont devenus NaN ou infinis (scikit-learn refuse alors d'entraîner la "
              "souche) : l'erreur pondérée est tombée à 0 et α est devenu infini. Relis quels exemples la mise à jour "
              "doit alourdir.")
        print("⏳ Ex 14.23 : points 2 et 3, puis a) et b) : corrige d'abord le point 1.")
    elif returned("14.23", "adaboost_23", result_23):
        stumps_23, alphas_23, weights_23 = result_23
        weights_23 = [np.asarray(w, dtype=float) for w in weights_23]
        wrong0_23 = stumps_23[0].predict(X_23) != Y_23
        with np.errstate(all="ignore"):
            growth_23 = weights_23[1] / weights_23[0]        # how the update changed each weight after round 1
        cuts_23 = {(int(s.tree_.feature[0]), round(float(s.tree_.threshold[0]), 6)) for s in stumps_23}
        diagnostics_23 = [
            ("après le premier tour, la mise à jour alourdit les exemples que la souche a mal classés, par rapport aux "
             "autres", bool(np.mean(growth_23[wrong0_23]) > np.mean(growth_23[~wrong0_23]) * (1 + 1e-9))),
            ("les poids avec lesquels chaque souche est entraînée somment à 1",
             all(np.isfinite(w).all() and abs(w.sum() - 1) < 1e-9 for w in weights_23)),
            ("les souches changent d'un tour à l'autre", len(cuts_23) > 1),
        ]
        all_ok_23, failed_23 = True, None
        for k, (what, ok) in enumerate(diagnostics_23, start=1):
            if not all_ok_23:
                print(f"⏳ Ex 14.23 : point {k} ({what}) : corrige d'abord le point {failed_23}.")
                continue
            verdict("14.23", ok, f"point {k} : {what}.", f"point {k} : attendu : {what}.")
            if not ok:
                all_ok_23, failed_23 = False, k
        with np.errstate(all="ignore"):
            accuracy_23 = float(np.mean(predict_23(stumps_23, alphas_23, X_23) == Y_23))
        print(f"training accuracy of the 40 rounds: {accuracy_23:.3f} · of the first stump alone: "
              f"{float(np.mean(stumps_23[0].predict(X_23) == Y_23)):.3f}")
        if not all_ok_23:
            print(f"⏳ Ex 14.23a et 14.23b : corrige d'abord le point {failed_23}.")
        else:                                            # a) and b) only once the three points pass
            wb.check("14.23a", accuracy_23, computed=True)
            wb.check("14.23b", float(alphas_23[1]), computed=True)
            oracle_23 = AdaBoostClassifier(n_estimators=40, random_state=0).fit(X_23, Y_23)
            verdict("14.23", np.array_equal(predict_23(stumps_23, alphas_23, X_23), oracle_23.predict(X_23))
                    and np.allclose(alphas_23, oracle_23.estimator_weights_),
                    "mêmes poids des souches et mêmes prédictions que l'AdaBoostClassifier de scikit-learn.",
                    "tes poids de souches ou tes prédictions diffèrent de ceux de l'AdaBoostClassifier de scikit-learn.")'''

# ---- 14.24 -----------------------------------------------------------------
CHECK_24 = r'''with checked("14.24"):
    X_doc_24, y_doc_24 = np.arange(1.0, 8.0).reshape(-1, 1), np.array([0, 0, 0, 1, 1, 0, 1])
    doc_24 = fitted(mylearn.ensemble.AdaBoostClassifier(n_estimators=3), X_doc_24, y_doc_24)
    print("docstring example: errors", np.round(np.asarray(doc_24.estimator_errors_, dtype=float), 4).tolist(),
          "(expected [0.1429, 0.1667, 0.2]) · weights", np.round(np.asarray(doc_24.estimator_weights_, dtype=float), 4).tolist(),
          "(expected [1.7918, 1.6094, 1.3863])")
with checked("14.24"):
    X_24, y_24 = wb.synth.make_blobs(n=400, centers=3, std=4.0, seed=1424)   # three overlapping classes
    X_24 = X_24.astype(np.float32).astype(float)                              # continuous data, no ties
    X_train_24, y_train_24, X_test_24 = X_24[:300], y_24[:300], X_24[300:]
    for params in [{"n_estimators": 20}, {"n_estimators": 15, "learning_rate": 0.5}]:
        start_24 = time.perf_counter()
        mine_24 = fitted(mylearn.ensemble.AdaBoostClassifier(**params), X_train_24, y_train_24)
        seconds_24 = time.perf_counter() - start_24
        oracle_24 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=0, **params).fit(X_train_24, y_train_24)
        k_24 = len(mine_24.estimators_)
        same_weights_24 = (k_24 == len(oracle_24.estimators_)
                           and np.allclose(mine_24.estimator_weights_, oracle_24.estimator_weights_[:k_24])
                           and np.allclose(mine_24.estimator_errors_, oracle_24.estimator_errors_[:k_24]))
        same_stages_24 = all(np.array_equal(np.asarray(a), b) for a, b in
                             zip(mine_24.staged_predict(X_test_24), oracle_24.staged_predict(X_test_24)))
        verdict("14.24", same_weights_24 and same_stages_24,
                f"{params} : mêmes {k_24} poids, mêmes erreurs et mêmes prédictions après chaque tour que scikit-learn "
                f"(trois classes, en {fr(seconds_24, 2)} s).",
                f"{params} : scikit-learn garde {len(oracle_24.estimators_)} membres, de poids "
                f"{np.round(oracle_24.estimator_weights_[:3], 4).tolist()}… ; toi {k_24}, de poids "
                f"{np.round(np.asarray(mine_24.estimator_weights_[:3], dtype=float), 4).tolist()}… (ou d'autres "
                "prédictions) ; les tests ci-dessous disent où.")
    votes_24 = np.asarray(mine_24.decision_function(X_test_24), dtype=float)
    verdict("14.24", votes_24.shape == (100, 3) and np.allclose(votes_24.sum(axis=1), 1),
            "decision_function rend une ligne de 3 votes normalisés par exemple.",
            (f"decision_function : les lignes doivent sommer à 1 (divise par la somme des α), et non à "
             f"{np.round(votes_24.sum(axis=1)[:3], 4).tolist()}…" if votes_24.shape == (100, 3) else
             f"decision_function doit rendre un tableau (100, 3), une colonne par classe, pas {votes_24.shape}."))
if written(lambda: mylearn.ensemble.AdaBoostClassifier(n_estimators=1).fit(np.array([[0.0], [1.0]]), np.array([0, 1]))):
    run_mylearn_tests("ensemble", keyword="test_ada_boost_classifier_")
else:
    print("⏳ Ex 14.24 : les tests attendent le fit de ton AdaBoostClassifier.")'''

# ---- 14.25 -----------------------------------------------------------------
CHECK_25 = r'''with checked("14.25"):
    X_w_25, y_w_25 = wb.synth.make_moons(n=60, noise=0.3, seed=1425)
    X_w_25 = X_w_25.astype(np.float32).astype(float)
    t_w_25 = y_w_25 + X_w_25[:, 0]                                            # a real target
    weighted_25 = fitted(mylearn.tree.DecisionTreeRegressor(max_depth=4), X_w_25, t_w_25, sample_weight=np.full(60, 2.0))
    doubled_25 = fitted(mylearn.tree.DecisionTreeRegressor(max_depth=4), np.vstack([X_w_25, X_w_25]), np.concatenate([t_w_25, t_w_25]))
    first_25 = mylearn.tree.DecisionTreeRegressor(max_features=1, random_state=0)
    a_25 = fitted(first_25, X_w_25, t_w_25).predict(X_w_25)
    b_25 = fitted(first_25, X_w_25, t_w_25).predict(X_w_25)
    oracle_tree_25 = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X_w_25, t_w_25)
    mine_tree_25 = fitted(mylearn.tree.DecisionTreeRegressor(max_depth=4), X_w_25, t_w_25)
    verdict("14.25", np.allclose(weighted_25.predict(X_w_25), doubled_25.predict(X_w_25)) and np.array_equal(a_25, b_25)
            and np.allclose(mine_tree_25.feature_importances_, oracle_tree_25.feature_importances_),
            "ton DecisionTreeRegressor a les options du 13.25 : poids, max_features reproductible, importances.",
            "ton DecisionTreeRegressor doit accepter sample_weight (un poids 2 = un exemple répété), rendre "
            "max_features reproductible avec random_state et calculer feature_importances_ comme scikit-learn ; les "
            "tests du ch. 13 lancés plus bas disent où.")
with checked("14.25"):
    doc_25 = fitted(mylearn.ensemble.GradientBoostingRegressor(n_estimators=2, learning_rate=0.5, max_depth=1),
                    np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1.0, 1.2, 3.0, 3.4]))
    staged_doc_25 = [np.round(np.asarray(p, dtype=float), 4).tolist() for p in doc_25.staged_predict(np.array([[0.0], [5.0]]))]
    print("docstring example: init_", round(float(doc_25.init_), 4), "(expected 2.15) · staged", staged_doc_25,
          "(expected [[1.625, 2.675], [1.3625, 2.9375]]) · train_score_",
          np.round(np.asarray(doc_25.train_score_, dtype=float), 4).tolist(), "(expected [0.3006, 0.0939])")
with checked("14.25"):
    rows_25 = np.random.default_rng(1425).choice(len(X_CAL_TRAIN), 1400, replace=False)
    X_25 = X_CAL_TRAIN[rows_25].astype(np.float32).astype(float)               # continuous data, no ties
    y_25 = Y_CAL_TRAIN[rows_25]
    X_train_25, y_train_25, X_test_25, y_test_25 = X_25[:400], y_25[:400], X_25[400:], y_25[400:]
    params_25 = {"n_estimators": 20, "learning_rate": 0.3, "max_depth": 2, "min_samples_leaf": 10}   # no tiny leaves: no ties
    start_25 = time.perf_counter()
    mine_25 = fitted(mylearn.ensemble.GradientBoostingRegressor(**params_25), X_train_25, y_train_25)
    seconds_25 = time.perf_counter() - start_25
    oracle_25 = GradientBoostingRegressor(criterion="squared_error", random_state=0, **params_25).fit(X_train_25, y_train_25)
    stages_25 = list(mine_25.staged_predict(X_test_25))
    same_25 = (len(stages_25) == 20 and abs(float(mine_25.init_) - float(np.mean(y_train_25))) < 1e-12
               and all(np.allclose(a, b) for a, b in zip(stages_25, oracle_25.staged_predict(X_test_25)))
               and np.allclose(mine_25.train_score_, oracle_25.train_score_))
    verdict("14.25", same_25,
            f"mêmes prédictions après chacun des 20 arbres et même train_score_ que scikit-learn (en {fr(seconds_25, 2)} s) ; "
            f"RMSE de test {fr(rmse(y_test_25, stages_25[-1]))}.",
            "tes prédictions après chaque arbre, ton init_ ou ton train_score_ diffèrent de ceux de scikit-learn ; les "
            "tests ci-dessous disent où.")
    sub_a_25 = fitted(mylearn.ensemble.GradientBoostingRegressor(subsample=0.5, random_state=3, **params_25), X_train_25, y_train_25)
    sub_b_25 = fitted(mylearn.ensemble.GradientBoostingRegressor(subsample=0.5, random_state=3, **params_25), X_train_25, y_train_25)
    verdict("14.25", np.array_equal(sub_a_25.predict(X_test_25), sub_b_25.predict(X_test_25))
            and all(tree.root_.n_samples == 200 for tree in sub_a_25.estimators_),
            "subsample=0.5 : chaque arbre apprend sur 200 exemples tirés, et le même random_state redonne le même modèle.",
            "subsample=0.5 : chaque arbre doit apprendre sur max(1, int(0.5 × 400)) = 200 exemples tirés sans remise, "
            "avec un générateur créé dans fit (le même random_state redonne le même modèle).")
if written(lambda: mylearn.ensemble.GradientBoostingRegressor(n_estimators=1).fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))):
    run_mylearn_tests("ensemble", keyword="test_gradient_boosting_regressor_")
else:
    print("⏳ Ex 14.25 : les tests attendent le fit de ton GradientBoostingRegressor.")
if written(lambda: mylearn.tree.DecisionTreeRegressor().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))):
    run_mylearn_tests("tree", keyword="test_decision_tree_regressor_ensemble_")
else:
    print("⏳ Ex 14.25 : les tests du ch. 13 attendent ton DecisionTreeRegressor.")'''

PART_C = Part("C", "Le boosting",
              "Lire les courbes d'entraînement et de validation d'un gradient boosting (14.20), prévoir l'effet du "
              "learning rate (14.21), voir AdaBoost découper le plan souche après souche (14.22), réparer un AdaBoost "
              "qui n'apprend rien (14.23), puis tes `AdaBoostClassifier` (14.24) et `GradientBoostingRegressor` "
              "(14.25). La cellule suivante prend 3 000 districts d'entraînement pour les boostings et 2 000 autres "
              "pour les valider, et définit `boosting_curves`, qui rend les RMSE d'entraînement et de validation "
              "après chaque arbre.",
              given=PART_C_GIVEN, exercises=[
    Ex("14.20", "📈", 1, 15, "Lire les courbes train/validation d'un boosting",
       "diagnostiquer un gradient boosting à ses courbes d'entraînement et de validation, et choisir où s'arrêter.",
       "✏️ 14.7 · ch. 9 (overfitting, underfitting) · fiche, section sur le gradient boosting", thread="California",
       tracks="R, C",
       body=r"""La cellule fournie entraîne trois `GradientBoostingRegressor` sur 3 000 districts, avec 400 arbres et un learning rate de 0,3. Ils ne diffèrent que par la **profondeur** de leurs arbres : 1, 3 et 8, dans un ordre caché (panneaux A, B et C). Elle trace, pour chacun, la RMSE d'entraînement et celle de validation après chaque arbre. Réponds en lisant les courbes, puis vérifie d) et e) par le calcul :
a) `overfit_20` : le panneau dont le modèle apprend son entraînement **par cœur** ;
b) `underfit_20` : le panneau dont le modèle est encore en **underfitting** après 400 arbres ;
c) `stop_20` : pour le panneau restant (ni celui de a), ni celui de b)), où est le minimum de la courbe de validation : `"A"` avant 50 arbres ; `"B"` de 50 à 300 arbres ; `"C"` de 301 à 400 arbres ; `"D"` la courbe de validation descend encore à 400 arbres ;
d) `best_trees_20` : pour les panneaux A, B et C, le nombre d'arbres (de 1 à 400) au minimum de la RMSE de validation, calculé à partir de `CURVES_20` (un dictionnaire `panneau → (RMSE d'entraînement, RMSE de validation)`) ;
e) `gap_20` : pour le panneau de c), la RMSE de validation après 400 arbres moins son minimum.

Dans tes notes : attribue une profondeur à chaque panneau, et dis à quoi tu la reconnais. Combien d'arbres garderais-tu pour chacun ? Lequel des trois choisirais-tu, et que changerais-tu à son réglage ?""",
       given=GIVEN_20, todo=TODO_20, check=CHECK_20,
       solution=r'''order_20 = np.random.default_rng(1421).permutation([1, 3, 8])        # recomputed for the record only
panel_of_20 = {int(depth): panel for panel, depth in zip("ABC", order_20)}
overfit_20, underfit_20 = panel_of_20[8], panel_of_20[1]
best_trees_20 = [int(np.argmin(CURVES_20[panel][1])) + 1 for panel in "ABC"]
middle_20 = CURVES_20[panel_of_20[3]][1]
best_middle_20 = int(np.argmin(middle_20)) + 1
stop_20 = "A" if best_middle_20 < 50 else "B" if best_middle_20 <= 300 else "C" if best_middle_20 < 400 else "D"
gap_20 = float(middle_20[-1] - middle_20.min())
print("depths of panels A, B, C:", order_20.tolist(), "· trees at the validation minimum:", best_trees_20)
''',
       record=r'''wb.record("14.20a", overfit_20,
          mistakes={f"panneau {panel} : sa RMSE d'entraînement tombe-t-elle presque à 0 pendant que celle de validation reste bien plus haute ?": panel
                    for panel in "ABC" if panel != overfit_20})
wb.record("14.20b", underfit_20,
          mistakes={f"panneau {panel} : ses deux courbes descendent-elles encore ensemble, proches l'une de l'autre, à 400 arbres ?": panel
                    for panel in "ABC" if panel != underfit_20})
wb.record("14.20c", stop_20,
          mistakes={f"choix {choice} : repère le point le plus bas de la courbe de validation du panneau restant (ni celui de a), ni celui de b)), et lis son abscisse": choice
                    for choice in "ABCD" if choice != stop_20})
wb.record("14.20d", best_trees_20,
          mistakes={"np.argmin rend une position qui commence à 0 : un nombre d'arbres commence à 1": [b - 1 for b in best_trees_20],
                    "c'est le minimum de la RMSE d'entraînement : on demande celui de la validation": [int(np.argmin(CURVES_20[p][0])) + 1 for p in "ABC"]})
wb.record("14.20e", gap_20, decimals=3, mistakes={"le signe : la valeur après 400 arbres moins le minimum": -gap_20})''',
       note="Le panneau de profondeur 8 apprend l'entraînement par cœur (RMSE d'entraînement presque nulle, 0,02, "
            "après une centaine d'arbres), et sa validation est au plus bas dès 23 arbres, puis stagne vers 0,575 : "
            "chaque arbre de profondeur 8 (jusqu'à 256 feuilles, environ 200 pour les premiers) colle au bruit. Le panneau de profondeur 1 (des souches) est encore en "
            "underfitting : ses deux courbes descendent ensemble, à 0,535 et 0,584 après 400 arbres, et son minimum "
            "est presque au dernier arbre (397). Le panneau de profondeur 3 a le meilleur minimum (0,547 à 155 "
            "arbres), puis remonte doucement (0,562 à 400) : c'est l'overfitting progressif d'un boosting, que "
            "l'early stopping arrête. On garderait 155 arbres pour la profondeur 3, une vingtaine pour la "
            "profondeur 8 ; les souches demandent plus d'arbres, ou un learning rate plus grand. Le meilleur choix "
            "est la profondeur 3, avec un learning rate plus petit et plus d'arbres (🔮 14.21)."),

    Ex("14.21", "🔮", 1, 15, "Learning rate et nombre d'arbres : prédire les courbes",
       "prévoir, en nombres, ce que le learning rate change au nombre d'arbres utile et à l'erreur de validation.",
       "Ex 14.20 · fiche, section sur le gradient boosting (le learning rate et le nombre d'arbres)",
       thread="California", tracks="C", hypothesis=True,
       body=r"""Mêmes données que 14.20, des arbres de profondeur 3, jusqu'à 1 000 arbres, et cinq learning rates : 0,02 ; 0,05 ; 0,1 ; 0,5 et 1. La fiche te dit déjà le sens des effets (un petit learning rate demande plus d'arbres et généralise d'ordinaire mieux) ; prédis plutôt leur **taille**.

Prédis, **avant** d'exécuter quoi que ce soit :
a) `ratio_21` : en passant de 0,5 à 0,05 (un learning rate divisé par 10), le nombre d'arbres au minimum de la RMSE de validation est multiplié par environ : `"A"` 1 ; `"B"` 3 ; `"C"` 10 ; `"D"` 100 ;
b) `gain_21` : la meilleure RMSE de validation avec 0,05, comparée à celle avec 0,5 : `"A"` moins bonne ; `"B"` meilleure de moins de 0,005 ; `"C"` meilleure de 0,005 à 0,1 ; `"D"` meilleure de plus de 0,1 ;
c) `slow_beats_21` : avec 0,02, la meilleure RMSE de validation **en 1 000 arbres au plus** est-elle plus basse qu'avec 0,1 ? (`True` ou `False`)

Puis exécute l'expérience (une quarantaine de secondes) : elle affiche, pour chaque learning rate, le nombre d'arbres au minimum, la meilleure RMSE de validation, la RMSE après 1 000 arbres et la RMSE d'entraînement, et trace les courbes de validation.""",
       given=GIVEN_21,
       todo=r'''ratio_21 = ...        # a) "A", "B", "C" or "D"
gain_21 = ...         # b) "A", "B", "C" or "D"
slow_beats_21 = ...   # c) True or False''',
       solution=r'''ratio_21, gain_21, slow_beats_21 = "C", "C", False   # the answers, for the record''',
       after=[("code", guarded(EXPERIMENT_21, ["ratio_21", "gain_21", "slow_beats_21"],
                               "⏳ Ex 14.21 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : compare avec tes prédictions. Le produit « learning rate × nombre d'arbres au "
                     "minimum » est-il à peu près constant ? Que se passe-t-il à droite du minimum pour les grands "
                     "learning rates, et pour 0,02 ? Quel couple (learning rate, nombre d'arbres) choisirais-tu, avec "
                     "quel budget de temps ?")],
       record=r'''wb.record("14.21a", "C", mistakes={"le learning rate multiplie chaque arbre : pour la même correction totale, faut-il autant d'arbres avec des pas dix fois plus petits ?": "A",
                                     "chaque arbre ne corrige qu'une fraction η de ce qui reste : avec η dix fois plus petit, combien d'arbres pour la même correction ?": "B",
                                     "dix fois plus petit, est-ce cent fois plus d'arbres ? pense à une somme de pas de taille η": "D"})
wb.record("14.21b", "C", mistakes={"un petit learning rate corrige moins vite : est-ce qu'il finit par moins bien généraliser, si on lui laisse assez d'arbres ?": "A",
                                    "avec η = 0,5, chaque arbre colle fortement au bruit de l'entraînement : l'écart reste-t-il si petit ?": "B",
                                    "une RMSE de validation vers 0,5 : peut-elle gagner plus de 0,1 par le seul learning rate ?": "D"})
wb.record("14.21c", False, mistakes={"un learning rate de 0,02 avance cinq fois moins vite que 0,1 : 1 000 arbres suffisent-ils à l'amener au même niveau ?": True})''',
       note="De 0,5 à 0,05, le minimum passe de 96 à 964 arbres : 10 fois plus, comme le learning rate est 10 fois "
            "plus petit (le produit learning rate × arbres reste vers 48 à 65 pour 0,05, 0,1 et 0,5 ; 31 pour 1, qui colle au bruit dès les premiers arbres ; 20 pour 0,02, qui n'a pas fini). "
            "La meilleure RMSE de validation passe de 0,576 à 0,524 : 0,052 de mieux. Avec 0,02, la courbe est encore "
            "presque à son plus bas au 1 000ᵉ arbre (0,533 ; elle descendrait jusqu'à 0,522 vers 1 900 arbres) et "
            "reste au-dessus de celle de 0,1 (0,523 à 652 arbres) : un learning "
            "rate trop petit coûte des arbres, donc du temps, sans gain dans un budget fixé. Avec 1, le minimum "
            "arrive après 31 arbres (0,631), puis la validation remonte à 0,711 : le modèle colle au bruit. Sur huit "
            "autres tranches de 3 000 districts d'entraînement et 2 000 de validation prises dans `X_CAL_TRAIN` "
            "(de `[1000:6000]` à `[11500:16500]`), le rapport reste entre 7 et 16 (toujours plus proche de 10 que de "
            "3 ou de 100), le gain entre 0,010 et 0,044, et 0,02 reste derrière 0,1."),

    Ex("14.22", "🎨", 2, 25, "Frontières d'AdaBoost après 1, 3, 10 et 50 souches",
       "voir un ensemble de souches découper le plan, tour après tour.",
       "✏️ 14.6 · ch. 13 (souches, frontières) · fiche §14.7 (AdaBoost)", thread="synthétique", tracks="C",
       body=r"""Deux demi-lunes bruitées (300 points, `X_22`) et l'`AdaBoostClassifier` de scikit-learn, dont les membres par défaut sont des souches. Écris :
- `model_22` : `AdaBoostClassifier(n_estimators=50, random_state=0)` entraîné sur `X_22`, `Y_22` ;
- `stage_predictions_22(model, X, rounds)` : un dictionnaire qui associe à chaque $m$ de `rounds` les prédictions du modèle sur `X` quand il ne garde que ses $m$ premiers membres (`model.staged_predict(X)` donne celles après 1, 2, 3… membres) ;
- `draw_22(model, X, y, rounds)` : une figure de `len(rounds)` panneaux ; dans le panneau $m$, les régions prédites sur la grille `GRID_22` après $m$ membres (remises à la forme de `GRID1_22`, avec `ax.contourf`), les points d'entraînement colorés par classe, et un titre avec $m$ et l'accuracy d'entraînement après $m$ membres ; elle rend la figure ;
- `train_acc_22` : la liste des accuracies d'entraînement après 1, 3, 10 et 50 membres (`ROUNDS_22`).

La vérification contrôle `train_acc_22`, vérifie qu'après une souche les régions sont coupées par une seule droite parallèle à un axe, puis appelle `draw_22`.

Dans tes notes : décris la frontière après 3 souches, puis après 50. Pourquoi est-elle toujours faite de segments horizontaux et verticaux ? Où l'ensemble finit-il par dessiner de petits rectangles, et est-ce bon signe ?""",
       given=GIVEN_22, todo=TODO_22, check=CHECK_22, solution=SOLUTION_22 + as_printed(CHECK_22, "14.22"),
       record=r'''wb.record("14.22a", train_acc_22, decimals=3)''',
       note="Après une souche, une seule coupure horizontale ($x_2$ vers 0,06) : 0,820 d'accuracy d'entraînement. "
            "Après trois, un escalier grossier (0,923) ; après dix, l'escalier suit l'allure des lunes (0,927) ; "
            "après 50, la frontière contourne aussi des points isolés de l'autre classe par de petits rectangles "
            "(0,967 sur l'entraînement). "
            "Chaque souche pose une question sur **une** coordonnée, et la somme pondérée de leurs votes ne change "
            "qu'en franchissant l'un de leurs seuils : la frontière est faite de segments parallèles aux axes. Les "
            "petits rectangles autour de points isolés sont le début de l'overfitting : AdaBoost donne de plus en "
            "plus de poids aux points qu'il classe mal, souvent des points bruités. On le contrôle par le nombre de "
            "tours et le learning rate, choisis sur une validation."),

    Ex("14.23", "🐛", 2, 25, "L'AdaBoost qui n'apprend rien",
       "trouver et corriger les erreurs d'un AdaBoost écrit à la main, en s'aidant de ses propriétés.",
       "✏️ 14.6 · fiche §14.7 (pseudo-code d'AdaBoost)", thread="synthétique", tracks="C",
       body=r"""Un collègue a écrit `adaboost_23`, un AdaBoost (SAMME, deux classes 0 et 1) avec des souches de scikit-learn, et `predict_23`, son vote pondéré (correct). Son accuracy d'entraînement ne bouge pas, quel que soit le nombre de tours. Il y a **trois** erreurs dans `adaboost_23` : trouve-les et corrige-les (sans changer sa signature ni ce qu'elle rend), en t'aidant du pseudo-code de la fiche (§14.7).

La vérification lance ta version sur 400 points de deux demi-lunes, puis contrôle trois propriétés, dans l'ordre où les erreurs se masquent (un point ne se teste que si les précédents passent) : 1) après le premier tour, la mise à jour alourdit les exemples que la souche a mal classés, par rapport aux autres ; 2) les poids avec lesquels chaque souche est entraînée somment à 1 ; 3) les souches changent d'un tour à l'autre. Quand les trois points passent, elle vérifie a) l'accuracy d'entraînement des 40 tours et b) le poids $\alpha$ de la **deuxième** souche, puis compare ta version à l'`AdaBoostClassifier` de scikit-learn.

Dans tes notes : pour chaque erreur, le symptôme qu'elle provoque seule. Lesquelles des trois erreurs, seules, figent l'accuracy ? Une fois tout corrigé, pourquoi les exemples que la souche précédente classait mal pèsent-ils exactement la moitié du total après chaque mise à jour (à deux classes, avec $\alpha = \ln\frac{1 - \varepsilon}{\varepsilon}$) ?""",
       given=GIVEN_23, todo=TODO_23, check=CHECK_23,
       solution=SOLUTION_23 + CHECK_23.replace('with checked("14.23"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''def variant_23(n_rounds=40, weights_ok=True, normalise=True, update="wrong"):
    """The function of the statement with some of its bugs left (to compute the classic mistakes)."""
    w = np.full(len(X_23), 1 / len(X_23))
    stumps, alphas = [], []
    for m in range(n_rounds):
        if normalise:
            w = w / w.sum()
        stump = DecisionTreeClassifier(max_depth=1, random_state=m)
        stump.fit(X_23, Y_23, sample_weight=w if weights_ok else None)
        wrong = stump.predict(X_23) != Y_23
        err = np.sum(w[wrong])
        alpha = np.log((1 - err) / err)
        w = w * np.exp(alpha * (wrong if update == "wrong" else ~wrong))
        stumps.append(stump)
        alphas.append(alpha)
    return float(np.mean(predict_23(stumps, alphas, X_23) == Y_23)), float(alphas[1])


def safe_variant_23(**bugs):
    """variant_23, or (nan, nan) when the bugs make it crash (an error of 0 gives an infinite alpha)."""
    try:
        with np.errstate(all="ignore"):
            return variant_23(**bugs)
    except ValueError:
        return float("nan"), float("nan")


right_23 = variant_23()
classic_23 = {"c'est l'accuracy de la première souche seule : l'ensemble n'apprend rien de plus. Regarde si tes souches changent d'un tour à l'autre, et ce que valent les poids au début de chaque tour": safe_variant_23(weights_ok=False),
              "l'ensemble fait moins bien que sa première souche : ses α changent de signe d'un tour à l'autre et s'annulent ; regarde ce que valent les poids au début de chaque tour": safe_variant_23(weights_ok=False, normalise=False)}
wb.record("14.23a", right_23[0], decimals=4,
          mistakes={message: value[0] for message, value in classic_23.items()
                    if np.isfinite(value[0]) and round(value[0], 4) != round(right_23[0], 4)})
wb.record("14.23b", right_23[1], decimals=4,
          mistakes={"α nul : la deuxième souche se trompe sur la moitié exacte des poids, comme si c'était encore la première ; regarde ce que voit la souche quand elle s'entraîne": safe_variant_23(weights_ok=False)[1],
                    "un α négatif : l'erreur pondérée de la deuxième souche dépasse 0,5 ; regarde ce que valent les poids au début de chaque tour": safe_variant_23(weights_ok=False, normalise=False)[1],
                    "α faussé : err n'est pas une proportion ; regarde ce que valent les poids au début de chaque tour": safe_variant_23(normalise=False)[1],
                    "la formule de la fiche n'a pas de facteur 1/2 : α = ln((1 − ε)/ε) à deux classes": right_23[1] / 2})''',
       note="Les trois erreurs, dans l'ordre des points : la mise à jour alourdit les exemples **bien** classés "
            "(`== y` au lieu de `!= y`) ; les poids ne sont jamais renormalisés, si bien que `err` n'est plus une "
            "proportion (ici elle dépasse 0,5, et $\\alpha$ devient négatif) ; la souche est entraînée sans "
            "`sample_weight`, et toutes les souches sont alors la même. Deux d'entre elles, seules, figent l'accuracy "
            "à celle de la première souche (0,820) : l'oubli de `sample_weight` et celui de la normalisation ; la "
            "mauvaise mise à jour, seule, fait tomber l'erreur pondérée à 0, $\\alpha$ devient infini et les poids "
            "NaN. Une fois tout corrigé, les mal classés pesaient $\\varepsilon$ et sont multipliés par $e^\\alpha = "
            "(1 - \\varepsilon)/\\varepsilon$ : ils pèsent $1 - \\varepsilon$, autant que les bien classés ; après "
            "normalisation, chaque groupe vaut la moitié, la souche précédente fait exactement comme le hasard sur "
            "les nouveaux poids, et la suivante doit trouver autre chose. Corrigée, la fonction donne les mêmes poids "
            "et les mêmes prédictions que scikit-learn."),

    Ex("14.24", "🔨", 3, 75, "Coder AdaBoost (SAMME) avec des souches pondérées",
       "écrire AdaBoost multi-classe sur tes souches, identique à celui de scikit-learn.",
       "✏️ 14.6 · Ex 13.25 (`sample_weight`) · Ex 14.23 · fiche §14.7 (pseudo-code)", thread="synthétique", tracks="C",
       mylearn="ensemble.py",
       body=r"""> **mylearn** : dans `mon_travail/mylearn/ensemble.py`, mêmes règles qu'en 14.17. Enregistre, puis relance la cellule de vérification.

Écris la classe `AdaBoostClassifier`. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__` range ses quatre arguments tels quels ; le gabarit est `estimator`, ou `DecisionTreeClassifier(max_depth=1)` de **ton** `tree.py` si `estimator` vaut `None` ; il n'est jamais entraîné lui-même : chaque membre est un `clone`, qui reçoit, s'il a un `random_state`, un entier tiré d'un générateur `np.random.default_rng(random_state)` créé dans `fit` (comme le bagging) ;
- `fit` vérifie **d'abord** que le `fit` du gabarit a un paramètre `sample_weight`, avant tout entraînement ; puis les poids partent de $1/n$ ; à chaque tour : normaliser les poids (somme 1), entraîner le clone avec `sample_weight`, calculer `err` (la somme des poids des mal classés) ; si `err == 0`, garder le membre avec le poids 1,0 et l'erreur 0,0, et s'arrêter ; si `err >= 1 - 1/K`, rejeter le membre et s'arrêter (`ValueError` si c'était le tout premier : l'ensemble serait vide) ; sinon $\alpha = \eta\,(\ln\frac{1 - \text{err}}{\text{err}} + \ln(K - 1))$, garder le membre et multiplier par $e^\alpha$ le poids des mal classés ;
- attributs : `classes_` (labels triés, au moins 2 : `ValueError` sinon), `estimators_`, `estimator_weights_` et `estimator_errors_` (des tableaux NumPy, un élément par membre gardé) ; `ValueError` aussi pour `n_estimators < 1`, `learning_rate <= 0`, `NaN` ou `len(X) != len(y)` ;
- `decision_function(X)` rend un tableau `(n, K)` : pour chaque classe, la somme des $\alpha_m$ des membres qui la prédisent, divisée par la somme de tous les $\alpha_m$ ; `predict(X)` la classe de plus grand vote (à égalité, la première de `classes_`) ; `staged_predict(X)` est une **fonction génératrice** qui donne les prédictions après 1, 2, … membres ; `score(X, y)` l'accuracy, un `float` Python ; `RuntimeError` avant `fit`.

La vérification essaie l'exemple de la docstring, compare ton AdaBoost à celui de scikit-learn sur trois classes (poids, erreurs, prédictions après chaque tour, avec deux réglages), contrôle `decision_function`, puis lance les tests.

Dans tes notes : pourquoi vérifier la signature avant tout entraînement ? À deux classes, que vaut $\ln(K - 1)$ ? À trois classes, quel $\alpha$ la formule sans ce terme donnerait-elle à un membre qui se trompe sur 55 % des poids ?""",
       check=RELOAD + CHECK_24, solution=ref_check(CHECK_24, "14.24"),
       note="Vérifier la signature d'abord évite un entraînement inutile, et surtout un message obscur : un gabarit "
            "sans `sample_weight` lèverait un `TypeError` au premier tour, sans dire pourquoi AdaBoost a besoin de "
            "poids. À deux classes, $\\ln(K - 1) = \\ln 1 = 0$ : SAMME redonne l'AdaBoost à deux classes. Sans ce terme, un "
            "membre à trois classes qui se trompe sur 55 % des poids aurait $\\alpha = \\ln(0{,}45/0{,}55) \\approx "
            "-0{,}20$, négatif, alors qu'il fait mieux que le hasard (66,7 % d'erreur) ; avec SAMME, il reçoit "
            "$\\ln(0{,}45/0{,}55) + \\ln 2 \\approx 0{,}49$ et compte dans le vote."),

    Ex("14.25", "🔨", 3, 60, "Coder un gradient boosting de régression",
       "écrire le gradient boosting de la fiche sur tes arbres de régression, identique à celui de scikit-learn.",
       "✏️ 14.7 · Ex 13.26 · ch. 9 (MSE, R²) · fiche, section sur le gradient boosting (pseudo-code)",
       thread="California", tracks="M, C", mylearn="ensemble.py` et `tree.py",
       body=r"""> **mylearn** : dans `mon_travail/mylearn/ensemble.py` (et `tree.py`), mêmes règles qu'en 14.17. Enregistre, puis relance la cellule de vérification.

**D'abord, finis ton arbre de régression** (`tree.py`). Le 13.26 l'a laissé sans les options des ensembles : comme ton `DecisionTreeClassifier` au 13.25, ton `DecisionTreeRegressor` accepte `sample_weight` (un poids 2 équivaut à un exemple répété), `max_features` et `splitter="random"` (avec un générateur créé dans `fit` à partir de `random_state`), et calcule `feature_importances_` (la baisse de variance pondérée, normalisée). Ce sont les options d'une forêt de régression, comme celle de scikit-learn du 14.15.

**Puis écris la classe `GradientBoostingRegressor`** (`ensemble.py`). **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__` range ses six arguments tels quels ; `fit(X, y)` : `init_ = float(np.mean(y))` et $F = $ `init_` pour tous les exemples ; à chaque étape, les lignes de l'étape (toutes, ou, si `subsample < 1`, `max(1, int(subsample * n))` lignes tirées sans remise par **un** générateur créé dans `fit` à partir de `random_state`), un `clone` d'un `DecisionTreeRegressor(max_depth=..., min_samples_leaf=...)` de **ton** `tree.py` ajusté aux résidus $y - F$ de ces lignes, puis $F \leftarrow F + \eta \times$ `tree.predict(X)` pour **tous** les exemples ;
- `train_score_[m]` est la MSE après l'étape $m$ sur les lignes de l'étape (toutes si `subsample=1.0`) ; `estimators_` est la liste des arbres ; `ValueError` pour `n_estimators < 1`, `learning_rate <= 0`, `subsample` hors de $]0, 1]$, `NaN` ou `len(X) != len(y)` ;
- `predict(X)` rend `init_` plus $\eta$ fois la somme des prédictions des arbres, un tableau `float64` ; `staged_predict(X)` est une **fonction génératrice** qui donne les prédictions après 1, 2, … arbres ; `score(X, y)` rend le $R^2$, un `float` Python ; `RuntimeError` avant `fit`.

La vérification contrôle d'abord ton arbre de régression (poids, `max_features`, importances), essaie l'exemple de la docstring, compare ton boosting à celui de scikit-learn sur 400 districts (prédictions après chaque arbre, `init_`, `train_score_`), vérifie `subsample`, puis lance les tests du boosting et ceux de ton arbre de régression qui étaient réservés à ce chapitre (`test_decision_tree_regressor_ensemble_` du ch. 13).

Dans tes notes : pourquoi `predict` peut-il se réduire à parcourir `staged_predict` ? Que mesure `train_score_` avec `subsample=0.5`, et pourquoi n'est-ce plus la MSE d'entraînement ?""",
       check=RELOAD + CHECK_25, solution=ref_check(CHECK_25, "14.25"),
       note="Les prédictions après $m$ arbres s'obtiennent en ajoutant un arbre à celles après $m - 1$ : la dernière "
            "valeur de `staged_predict` est `predict`, et une seule fonction porte la boucle (la référence fait "
            "ainsi). Avec `subsample=0.5`, chaque arbre n'apprend que sur la moitié des exemples, tirée à chaque "
            "étape ; `train_score_` mesure la MSE sur ces seuls exemples, ceux que l'étape a vus : une estimation "
            "bruitée de l'erreur d'entraînement, qui monte et descend d'une étape à l'autre. scikit-learn calcule "
            "en plus, sur les exemples laissés de côté, `oob_improvement_`, une estimation de ce qu'apporte chaque "
            "arbre."),
])

# ---- Part D ----------------------------------------------------------------
PART_D_GIVEN = r'''# The gradient boosting libraries: XGBoost and LightGBM (installed on Colab and by requirements.txt)
import platform
import sklearn

warnings.filterwarnings("ignore", message="X does not have valid feature names")   # LightGBM names its columns itself

try:
    import lightgbm
    import xgboost
except ImportError as exc:                                # wb.ensure installs a missing package at its pinned version
    print(f"⏳ {exc.name} absent : wb.ensure(\"{exc.name}\") l'installe à la version du workbook.")
print("xgboost", getattr(globals().get("xgboost"), "__version__", "absent"), "· lightgbm",
      getattr(globals().get("lightgbm"), "__version__", "absent"))'''

# ---- 14.26 -----------------------------------------------------------------
GIVEN_26 = r'''X_FIT_26, X_VAL_26, Y_FIT_26, Y_VAL_26 = train_test_split(X_CAL_TRAIN, Y_CAL_TRAIN, test_size=0.2, random_state=1426)
print(len(X_FIT_26), "districts to fit,", len(X_VAL_26), "to validate (early stopping),", len(X_CAL_TEST), "to test")'''

TODO_26 = r'''def best_iteration_26(curve, patience=20):
    """`curve` holds the validation RMSE after 1, 2, 3... trees. Scan it from the start and stop as soon as `patience`
    trees in a row have not improved on the best value so far (improve = strictly lower). Return the number of trees
    (counted from 1) at that best value."""
    raise NotImplementedError("best_iteration_26")


hgb_26 = ...          # HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.1, early_stopping=False,
                      #   random_state=0) fitted on X_FIT_26, Y_FIT_26
hgb_curve_26 = ...    # the validation RMSE of hgb_26 after each tree (staged_predict on X_VAL_26)
hgb_trees_26 = ...    # a) best_iteration_26(hgb_curve_26)
hgb_rmse_26 = ...     # b) the test RMSE (X_CAL_TEST) of hgb_26 when it keeps its first hgb_trees_26 trees
xgb_26 = ...          # xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.1, early_stopping_rounds=20,
                      #   random_state=0), fitted on X_FIT_26, Y_FIT_26 with eval_set=[(X_VAL_26, Y_VAL_26)], verbose=False
xgb_trees_26 = ...    # c) the number of trees XGBoost keeps (best_iteration counts from 0)
xgb_rmse_26 = ...     # d) the test RMSE of xgb_26 (its predict uses the best iteration)
lgbm_26 = ...         # lightgbm.LGBMRegressor(n_estimators=1000, learning_rate=0.1, random_state=0, verbose=-1), fitted on
                      #   X_FIT_26, Y_FIT_26 with eval_set=[(X_VAL_26, Y_VAL_26)],
                      #   callbacks=[lightgbm.early_stopping(20, verbose=False)]
lgbm_trees_26 = ...   # e) the number of trees LightGBM keeps (best_iteration_ counts from 1)
lgbm_rmse_26 = ...    # f) the test RMSE of lgbm_26
seconds_26 = ...      # dict {"hgb": ..., "xgb": ..., "lgbm": ...}: the fit time of each model, in seconds'''

SOLUTION_26 = r'''def best_iteration_26(curve, patience=20):
    """`curve` holds the validation RMSE after 1, 2, 3... trees. Scan it from the start and stop as soon as `patience`
    trees in a row have not improved on the best value so far (improve = strictly lower). Return the number of trees
    (counted from 1) at that best value."""
    best_value, best_index = np.inf, -1
    for index, value in enumerate(curve):
        if value < best_value:
            best_value, best_index = value, index
        elif index - best_index >= patience:
            break
    return best_index + 1


seconds_26 = {}
start = time.perf_counter()
hgb_26 = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.1, early_stopping=False, random_state=0)
hgb_26.fit(X_FIT_26, Y_FIT_26)
seconds_26["hgb"] = time.perf_counter() - start
hgb_curve_26 = [rmse(Y_VAL_26, predicted) for predicted in hgb_26.staged_predict(X_VAL_26)]
hgb_trees_26 = best_iteration_26(hgb_curve_26)
hgb_rmse_26 = next(rmse(Y_CAL_TEST, predicted) for m, predicted in enumerate(hgb_26.staged_predict(X_CAL_TEST), start=1)
                   if m == hgb_trees_26)
start = time.perf_counter()
xgb_26 = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.1, early_stopping_rounds=20, random_state=0)
xgb_26.fit(X_FIT_26, Y_FIT_26, eval_set=[(X_VAL_26, Y_VAL_26)], verbose=False)
seconds_26["xgb"] = time.perf_counter() - start
xgb_trees_26 = xgb_26.best_iteration + 1
xgb_rmse_26 = rmse(Y_CAL_TEST, xgb_26.predict(X_CAL_TEST))
start = time.perf_counter()
lgbm_26 = lightgbm.LGBMRegressor(n_estimators=1000, learning_rate=0.1, random_state=0, verbose=-1)
lgbm_26.fit(X_FIT_26, Y_FIT_26, eval_set=[(X_VAL_26, Y_VAL_26)], callbacks=[lightgbm.early_stopping(20, verbose=False)])
seconds_26["lgbm"] = time.perf_counter() - start
lgbm_trees_26 = lgbm_26.best_iteration_
lgbm_rmse_26 = rmse(Y_CAL_TEST, lgbm_26.predict(X_CAL_TEST))
'''

CHECK_26 = r'''wb.check("14.26a", hgb_trees_26)
wb.check("14.26b", hgb_rmse_26)
with wb.attempt("14.26"):
    if not filled(xgb_26, xgb_trees_26, xgb_rmse_26, lgbm_26, lgbm_trees_26, lgbm_rmse_26):
        raise NotImplementedError("xgb_26, xgb_trees_26, xgb_rmse_26, lgbm_26, lgbm_trees_26, lgbm_rmse_26")
    params_ok_26 = (xgb_26.get_params()["learning_rate"] == 0.1 and xgb_26.get_params()["n_estimators"] == 1000
                    and xgb_26.get_params()["early_stopping_rounds"] == 20 and lgbm_26.get_params()["learning_rate"] == 0.1
                    and lgbm_26.get_params()["n_estimators"] == 1000 and getattr(lgbm_26, "best_iteration_", 0) >= 1)
    verdict("14.26", params_ok_26,
            "les deux modèles ont les réglages de l'énoncé (learning rate 0,1, au plus 1 000 arbres) et un early stopping.",
            "relis les réglages : learning_rate=0.1 et n_estimators=1000 pour les deux, early_stopping_rounds=20 pour "
            "XGBoost, et pour LightGBM eval_set avec callbacks=[lightgbm.early_stopping(20, verbose=False)] "
            "(sans early stopping, best_iteration_ vaut 0).")
    on_test_26 = (abs(float(xgb_rmse_26) - rmse(Y_CAL_TEST, xgb_26.predict(X_CAL_TEST))) < 1e-9
                  and abs(float(lgbm_rmse_26) - rmse(Y_CAL_TEST, lgbm_26.predict(X_CAL_TEST))) < 1e-9)
    verdict("14.26", on_test_26, "d) et f) sont les RMSE de test des deux modèles.",
            "d) et f) sont les RMSE sur le test (X_CAL_TEST, Y_CAL_TEST) des modèles entraînés, pas sur la validation.")
    if not (params_ok_26 and on_test_26):
        raise NotImplementedError("c) à f) : corrige d'abord les points signalés ci-dessus")
    verdict("14.26", xgb_trees_26 == xgb_26.best_iteration + 1,
            f"c) XGBoost garde {xgb_trees_26} arbres (best_iteration = {xgb_26.best_iteration}, compté à partir de 0).",
            f"c) XGBoost garde best_iteration + 1 arbres, soit {xgb_26.best_iteration + 1} : best_iteration compte "
            "à partir de 0.")
    wb.check("14.26d", xgb_rmse_26)
    verdict("14.26", lgbm_trees_26 == lgbm_26.best_iteration_,
            f"e) LightGBM garde {lgbm_trees_26} arbres (best_iteration_, compté à partir de 1).",
            f"e) LightGBM garde best_iteration_ arbres, soit {lgbm_26.best_iteration_} (compté à partir de 1).")
    wb.check("14.26f", lgbm_rmse_26)
    if filled(seconds_26):
        print("fit times (s):", {name: round(float(seconds), 2) for name, seconds in dict(seconds_26).items()})'''

# ---- 14.27 -----------------------------------------------------------------
GIVEN_27 = r'''X_TRAIN_27, Y_TRAIN_27 = X_CAL_TRAIN[:5000], Y_CAL_TRAIN[:5000]
MODELS_27 = {"ridge": lambda seed: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
             "forest": lambda seed: RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1),
             "hgb": lambda seed: HistGradientBoostingRegressor(random_state=seed)}
SEEDS_27 = [0, 1, 2]'''

TODO_27 = r'''def versions_27():
    """Return a dict {name: version} for "python", "numpy", "scikit-learn", "xgboost" and "lightgbm", plus "cores":
    the number of CPU cores (os.cpu_count())."""
    raise NotImplementedError("versions_27")


def benchmark_27(models, X_train, y_train, X_test, y_test, seeds):
    """`models` maps a name to a function seed -> NEW unfitted regressor. For every name and every seed: build the
    model, fit it, measure the fit time (time.perf_counter) and the test RMSE. Return a pandas DataFrame with one row
    per (model, seed) and the columns "model", "seed", "rmse", "fit_seconds"."""
    raise NotImplementedError("benchmark_27")


def summary_27(results):
    """One row per model, sorted by increasing mean RMSE: the columns "rmse_mean", "rmse_std" (ddof=1) and
    "fit_seconds_mean" (a DataFrame indexed by the model name)."""
    raise NotImplementedError("summary_27")


def markdown_27(summary, versions):
    """A Markdown table of `summary` (a header line, a |---| line, one line per model, the numbers with 4 decimals
    for the RMSE and 2 for the seconds), followed by an empty line and one line that lists the versions."""
    raise NotImplementedError("markdown_27")'''

SOLUTION_27 = r'''def versions_27():
    """Return a dict {name: version} for "python", "numpy", "scikit-learn", "xgboost" and "lightgbm", plus "cores":
    the number of CPU cores (os.cpu_count())."""
    return {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "cores": os.cpu_count()}


def benchmark_27(models, X_train, y_train, X_test, y_test, seeds):
    """`models` maps a name to a function seed -> NEW unfitted regressor. For every name and every seed: build the
    model, fit it, measure the fit time (time.perf_counter) and the test RMSE. Return a pandas DataFrame with one row
    per (model, seed) and the columns "model", "seed", "rmse", "fit_seconds"."""
    rows = []
    for name, make in models.items():
        for seed in seeds:
            model = make(seed)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            seconds = time.perf_counter() - start
            rows.append({"model": name, "seed": seed, "rmse": rmse(y_test, model.predict(X_test)),
                         "fit_seconds": seconds})
    return pd.DataFrame(rows, columns=["model", "seed", "rmse", "fit_seconds"])


def summary_27(results):
    """One row per model, sorted by increasing mean RMSE: the columns "rmse_mean", "rmse_std" (ddof=1) and
    "fit_seconds_mean" (a DataFrame indexed by the model name)."""
    grouped = results.groupby("model")
    return pd.DataFrame({"rmse_mean": grouped["rmse"].mean(), "rmse_std": grouped["rmse"].std(ddof=1),
                         "fit_seconds_mean": grouped["fit_seconds"].mean()}).sort_values("rmse_mean")


def markdown_27(summary, versions):
    """A Markdown table of `summary` (a header line, a |---| line, one line per model, the numbers with 4 decimals
    for the RMSE and 2 for the seconds), followed by an empty line and one line that lists the versions."""
    lines = ["| model | RMSE (mean) | RMSE (std) | fit (s) |", "|---|---|---|---|"]
    for name, row in summary.iterrows():
        lines.append(f"| {name} | {row['rmse_mean']:.4f} | {row['rmse_std']:.4f} | {row['fit_seconds_mean']:.2f} |")
    return "\n".join(lines) + "\n\n" + " · ".join(f"{name} {version}" for name, version in versions.items())
'''

CHECK_27 = r'''with checked("14.27"):
    versions_found_27 = versions_27()
    if returned("14.27", "versions_27", versions_found_27):
        expected_27 = {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                       "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "cores": os.cpu_count()}
        verdict("14.27", {k: str(v) for k, v in dict(versions_found_27).items()} == {k: str(v) for k, v in expected_27.items()},
                "versions_27 rend les six informations attendues.",
                f"versions_27 doit rendre exactement les clés {list(expected_27)} et les versions installées.")
with checked("14.27"):
    results_27 = benchmark_27(MODELS_27, X_TRAIN_27, Y_TRAIN_27, X_CAL_TEST, Y_CAL_TEST, SEEDS_27)
    if returned("14.27", "benchmark_27", results_27):
        shape_ok_27 = (isinstance(results_27, pd.DataFrame) and list(results_27.columns) == ["model", "seed", "rmse", "fit_seconds"]
                       and len(results_27) == 9)
        verdict("14.27", shape_ok_27, "benchmark_27 rend 9 lignes (3 modèles × 3 graines) et les 4 colonnes attendues.",
                "benchmark_27 doit rendre un DataFrame de 9 lignes, aux colonnes model, seed, rmse, fit_seconds.")
        if shape_ok_27:
            again_27 = benchmark_27(MODELS_27, X_TRAIN_27, Y_TRAIN_27, X_CAL_TEST, Y_CAL_TEST, SEEDS_27)
            verdict("14.27", np.allclose(results_27["rmse"], again_27["rmse"], rtol=0, atol=1e-9),
                    "relancé, le benchmark redonne les mêmes RMSE : il est reproductible.",
                    "relancé, le benchmark donne d'autres RMSE : chaque modèle reçoit-il bien sa graine, et un modèle "
                    "neuf à chaque appel ?")
            spread_27 = results_27.groupby("model")["rmse"].std(ddof=1)
            verdict("14.27", spread_27["ridge"] < 1e-12 and spread_27["forest"] > 1e-6,
                    "la régression ridge ne dépend pas de la graine, la forêt si.",
                    "attendu : un écart nul entre graines pour ridge (aucun hasard), non nul pour la forêt.")
            table_27 = summary_27(results_27)
            text_27 = markdown_27(table_27, versions_found_27)
            lines_27 = str(text_27).splitlines()
            verdict("14.27", list(table_27.columns) == ["rmse_mean", "rmse_std", "fit_seconds_mean"]
                    and list(table_27["rmse_mean"]) == sorted(table_27["rmse_mean"])
                    and len(lines_27) >= 6 and lines_27[0].startswith("|") and lines_27[1].replace(" ", "").startswith("|---")
                    and "scikit-learn" in lines_27[-1],
                    "summary_27 trie les modèles par RMSE moyenne, et markdown_27 rend un tableau suivi des versions.",
                    "attendu : summary_27 aux colonnes rmse_mean, rmse_std, fit_seconds_mean, trié par rmse_mean ; "
                    "markdown_27 : une ligne d'en-tête, une ligne |---|, une ligne par modèle, puis les versions.")
            print(text_27)'''

# ---- 14.28 -----------------------------------------------------------------
TODO_28 = r'''def build_model_28():
    """Return a NEW, unfitted regressor (scikit-learn, XGBoost or LightGBM, possibly a pipeline) that receives the 8 raw
    features of California (shape (n, 8), in the order of CAL_FEATURES) and predicts MedHouseVal."""
    raise NotImplementedError("build_model_28")'''

SOLUTION_28 = r'''from sklearn.preprocessing import FunctionTransformer


def add_location_28(X):
    """The 8 features, plus rotated coordinates (a tree cuts one axis at a time: oblique coasts and valleys need
    several cuts) and the income per room times the occupancy."""
    X = np.asarray(X, dtype=float)
    latitude, longitude = X[:, 6], X[:, 7]
    rotated = [np.cos(a) * latitude + np.sin(a) * longitude for a in (np.pi / 6, np.pi / 4, np.pi / 3)]
    rotated += [-np.sin(a) * latitude + np.cos(a) * longitude for a in (np.pi / 6, np.pi / 4, np.pi / 3)]
    return np.column_stack([X, *rotated, X[:, 0] * X[:, 5] / X[:, 2]])


def build_model_28():
    """Return a NEW, unfitted regressor (scikit-learn, XGBoost or LightGBM, possibly a pipeline) that receives the 8 raw
    features of California (shape (n, 8), in the order of CAL_FEATURES) and predicts MedHouseVal."""
    return make_pipeline(FunctionTransformer(add_location_28),
                         HistGradientBoostingRegressor(max_iter=3000, learning_rate=0.05, max_leaf_nodes=63,
                                                       min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
                                                       validation_fraction=0.15, n_iter_no_change=50, random_state=0))
'''

CHECK_28 = r'''from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


def is_fitted_28(model):
    """True when check_is_fitted says the model was already fitted."""
    try:
        check_is_fitted(model)
    except NotFittedError:
        return False
    except Exception:                                   # not a scikit-learn-like model: the fit below will say
        return False
    return True


with checked("14.28"):
    if globals().get("REVEALED_28") is None:
        scores_28 = {}
        first_28, second_28 = build_model_28(), build_model_28()
        accepted_28 = returned("14.28", "build_model_28", first_28)
        if accepted_28 and (isinstance(first_28, type) or is_fitted_28(first_28)):
            print("❌ Ex 14.28 : build_model_28 doit rendre un modèle (une instance) neuf, non entraîné.")
            accepted_28 = False
        elif accepted_28 and first_28 is second_28:
            print("❌ Ex 14.28 : build_model_28 rend deux fois le même objet : il doit construire un modèle NEUF à chaque appel.")
            accepted_28 = False
        for split_seed in ([42, 1, 2, 3] if accepted_28 else []):   # the challenge, then three other splits
            X_tr, X_te, y_tr, y_te = train_test_split(X_CAL, Y_CAL, test_size=0.2, random_state=split_seed)
            model_28 = build_model_28()
            start_28 = time.perf_counter()
            model_28.fit(X_tr, y_tr)
            scores_28[split_seed] = (rmse(y_te, model_28.predict(X_te)), time.perf_counter() - start_28)
        if len(scores_28) == 4:                         # a control: the same method on shuffled targets must learn nothing
            X_tr, X_te, y_tr, y_te = train_test_split(X_CAL, Y_CAL, test_size=0.2, random_state=42)
            control_28 = build_model_28()
            control_28.fit(X_tr, np.random.default_rng(1428).permutation(y_tr))
            scores_28["shuffled"] = (rmse(y_te, control_28.predict(X_te)), 0.0)
            REVEALED_28 = scores_28                     # revealed once per session: kept if the cell runs again
    if globals().get("REVEALED_28") is not None:
        main_28 = REVEALED_28[42][0]
        others_28 = float(np.mean([REVEALED_28[s][0] for s in (1, 2, 3)]))
        print(f"test RMSE, split random_state=42: {main_28:.4f} (fit in {REVEALED_28[42][1]:.1f} s) · mean over the "
              f"splits 1, 2, 3: {others_28:.4f} ({', '.join(f'{REVEALED_28[s][0]:.4f}' for s in (1, 2, 3))})")
        shuffled_28 = REVEALED_28["shuffled"][0]
        if min(main_28, others_28) < 0.35 or shuffled_28 < 0.9:
            print(f"⚠️ Ex 14.28 : le défi n'est pas validé : {'une RMSE sous 0,35 est trop belle pour être honnête' if min(main_28, others_28) < 0.35 else ''}"
                  f"{' ; ' if min(main_28, others_28) < 0.35 and shuffled_28 < 0.9 else ''}"
                  f"{f'entraîné sur des cibles mélangées au hasard, ton modèle fait encore {fr(shuffled_28)} (un modèle honnête ne fait pas mieux que 1,15 environ)' if shuffled_28 < 0.9 else ''}"
                  " : ton modèle voit-il les lignes de test (des variables globales, un fit qui ignore ses arguments) ?")
        else:
            verdict("14.28", main_28 <= 0.45 and others_28 <= 0.45,
                    f"défi réussi : {fr(main_28, 4)} sur le test du défi et {fr(others_28, 4)} en moyenne sur trois "
                    "autres découpages." + (" 🌟 Palier atteint : les deux sous 0,43." if max(main_28, others_28) <= 0.43 else ""),
                    f"{fr(main_28, 4)} sur le test du défi, {fr(others_28, 4)} en moyenne ailleurs : les deux doivent "
                    "être au plus 0,45 ; règle ta méthode par validation croisée sur l'entraînement, jamais sur ces "
                    "tests (redémarre le noyau pour une nouvelle tentative).")'''

PART_D = Part("D", "Le gradient boosting en pratique",
              "Les trois bibliothèques de gradient boosting à histogrammes avec early stopping (14.26), un benchmark "
              "reproductible (14.27), puis le défi sur California (14.28). La cellule suivante importe XGBoost et "
              "LightGBM.",
              given=PART_D_GIVEN, exercises=[
    Ex("14.26", "📦", 3, 45, "HistGradientBoosting, XGBoost et LightGBM sur California",
       "entraîner trois gradient boostings à histogrammes avec early stopping, et lire leurs nombres d'arbres.",
       "Ex 14.20 · ch. 8 (jeu de validation) · ch. 9 (early stopping) · fiche, section sur le gradient boosting",
       thread="California", tracks="R, C",
       body=r"""Les 16 512 districts d'entraînement, coupés en 13 209 pour entraîner (`X_FIT_26`) et 3 303 pour valider (`X_VAL_26`) ; le test (`X_CAL_TEST`) ne sert qu'à la mesure finale. Les trois modèles ont le même learning rate, 0,1, et au plus 1 000 arbres ; l'early stopping s'arrête après 20 arbres sans amélioration de la RMSE de validation.
- **HistGradientBoosting** (dans scikit-learn 1.6, son `fit` n'accepte pas de jeu de validation fourni par toi) : entraîne-le **sans** early stopping, calcule sa courbe de validation avec `staged_predict`, puis fais l'early stopping toi-même avec `best_iteration_26(curve, patience=20)`, que tu écris : parcourir la courbe depuis le début, s'arrêter dès que `patience` arbres de suite n'ont pas battu (strictement) la meilleure valeur, et rendre le nombre d'arbres (compté à partir de 1) de cette meilleure valeur.
- **XGBoost** et **LightGBM** font l'early stopping eux-mêmes sur `eval_set` ; leurs `predict` utilisent ensuite le meilleur nombre d'arbres. Attention : `best_iteration` de XGBoost compte à partir de 0, `best_iteration_` de LightGBM à partir de 1.

Calcule les variables de la cellule, dont :
a) `hgb_trees_26` : le nombre d'arbres que garde ton early stopping ;
b) `hgb_rmse_26` : la RMSE de test de HistGradientBoosting réduit à ces arbres (`staged_predict` sur le test) ;
c) `xgb_trees_26` : le nombre d'arbres que garde XGBoost ; d) `xgb_rmse_26` : sa RMSE de test ;
e) `lgbm_trees_26` : le nombre d'arbres que garde LightGBM ; f) `lgbm_rmse_26` : sa RMSE de test ;
et `seconds_26`, les durées d'entraînement.

Les RMSE de XGBoost et de LightGBM se vérifient à 2 décimales seulement : leur calcul peut changer à la marge d'une machine à l'autre. Dans tes notes : classe les trois en RMSE et en temps. Pourquoi les trois ne gardent-ils pas le même nombre d'arbres, avec le même learning rate ? Pourquoi l'early stopping se règle-t-il sur la validation et pas sur le test ? À combien de la barre de 0,45 du défi (🏆 14.28) es-tu ?""",
       given=GIVEN_26, todo=TODO_26, check=CHECK_26, solution=SOLUTION_26 + as_printed(CHECK_26, "14.26"),
       record=r'''hgb_argmin_26 = int(np.argmin(hgb_curve_26)) + 1
test_curve_26 = [rmse(Y_CAL_TEST, predicted) for predicted in hgb_26.staged_predict(X_CAL_TEST)]
wb.record("14.26a", hgb_trees_26,
          mistakes={"c'est le minimum de toute la courbe : l'early stopping s'arrête dès que 20 arbres de suite n'ont rien amélioré": hgb_argmin_26,
                    "le nombre d'arbres se compte à partir de 1 : as-tu rendu la position Python ?": hgb_trees_26 - 1,
                    "c'est le dernier arbre parcouru : on demande celui de la meilleure valeur": hgb_trees_26 + 20})
wb.record("14.26b", hgb_rmse_26, decimals=3,
          mistakes={"c'est la RMSE avec les 1 000 arbres : garde seulement les hgb_trees_26 premiers": test_curve_26[-1],
                    "c'est la RMSE de validation : on demande celle du test": hgb_curve_26[hgb_trees_26 - 1],
                    "c'est une MSE : prends sa racine": hgb_rmse_26 ** 2})
wb.record("14.26d", xgb_rmse_26, decimals=2, mistakes={"c'est une MSE : prends sa racine": xgb_rmse_26 ** 2})
wb.record("14.26f", lgbm_rmse_26, decimals=2, mistakes={"c'est une MSE : prends sa racine": lgbm_rmse_26 ** 2})''',
       note="Ton early stopping garde 349 arbres de HistGradientBoosting (le minimum de toute la courbe est à 426 "
            "arbres, à peine plus bas) : RMSE de test 0,447. XGBoost garde 217 arbres (`best_iteration` = 216) : "
            "0,458 ; LightGBM 312 arbres : 0,447. Les trois partent du même learning rate, mais leurs arbres "
            "diffèrent : XGBoost pousse ses arbres en profondeur (`max_depth=6`, jusqu'à 64 feuilles, et une "
            "régularisation L2 des feuilles), LightGBM et HistGradientBoosting feuille par feuille, avec 31 "
            "feuilles au plus ; chaque arbre corrige donc plus ou moins, et la validation cesse de s'améliorer à "
            "des moments différents. Régler l'early stopping sur le test serait une fuite : le nombre d'arbres "
            "serait choisi pour ce test-là, et sa RMSE ne serait plus une estimation honnête. Les trois sont en "
            "dessous ou tout près de 0,45 sur ce découpage ; le défi en demande autant sur trois autres."),

    Ex("14.27", "🛠️", 2, 25, "Un benchmark reproductible : graines, versions et tableau de résultats",
       "écrire un petit banc d'essai qu'un collègue pourra relancer : graines, plusieurs exécutions, versions notées.",
       "Ex 14.26 · ch. 2 (moyenne, écart-type) · fiche, encadré « fixer les graines d'un ensemble »",
       thread="California", tracks="C",
       body=r"""Un tableau de résultats ne vaut que si on peut le refaire. Écris quatre fonctions :
- `versions_27()` : un dictionnaire des versions de `"python"` (`platform.python_version()`), `"numpy"`, `"scikit-learn"`, `"xgboost"` et `"lightgbm"` (leur `__version__`), plus `"cores"`, le nombre de cœurs (`os.cpu_count()`) ;
- `benchmark_27(models, X_train, y_train, X_test, y_test, seeds)` : `models` associe un nom à une fonction `graine → modèle neuf` (`MODELS_27` : une régression ridge, une forêt de 50 arbres, un HistGradientBoosting) ; pour chaque modèle et chaque graine de `seeds`, construire le modèle, l'entraîner, mesurer la durée de `fit` (`time.perf_counter`) et la RMSE de test ; rendre un `DataFrame` d'une ligne par couple (modèle, graine), aux colonnes `"model"`, `"seed"`, `"rmse"` et `"fit_seconds"` ;
- `summary_27(results)` : une ligne par modèle, indexée par son nom et triée par RMSE moyenne croissante, aux colonnes `"rmse_mean"`, `"rmse_std"` (écart-type avec `ddof=1`) et `"fit_seconds_mean"` ;
- `markdown_27(summary, versions)` : le texte d'un tableau Markdown du résumé (une ligne d'en-tête, une ligne `|---|`, une ligne par modèle, la RMSE avec 4 décimales et les secondes avec 2), suivi d'une ligne vide et d'une ligne qui liste les versions (`DataFrame.to_markdown` demande le paquet `tabulate`, absent : écris les lignes toi-même).

La vérification contrôle les versions, la forme du tableau, la reproductibilité (le benchmark relancé redonne les mêmes RMSE), l'effet des graines, le résumé et le texte, qu'elle affiche. Cet exercice se corrige avec `05_solutions.md`.

Dans tes notes : qu'est-ce qui, dans ce tableau, changerait sur une autre machine, et qu'est-ce qui ne changerait pas ? Pourquoi trois graines plutôt qu'une ? Où rangerais-tu ce tableau dans un dépôt, et avec quel commit ?""",
       given=GIVEN_27, todo=TODO_27, check=CHECK_27, solution=SOLUTION_27 + CHECK_27.replace('with checked("14.27"):\n', "if True:\n"),
       note="Sur une autre machine, les durées changent (processeur, nombre de cœurs, charge), et parfois les "
            "derniers chiffres d'une RMSE (une autre version de scikit-learn, de NumPy ou d'une bibliothèque "
            "d'algèbre linéaire, un autre nombre de cœurs pour une somme faite en parallèle) ; les graines, les "
            "données et le code ne changent pas. Une seule graine peut faire prendre la chance pour un progrès : la "
            "forêt de 50 arbres varie de quelques millièmes de RMSE d'une graine à l'autre, autant que l'écart entre "
            "deux réglages voisins. Le tableau et les versions vont dans le dépôt (par exemple `results/benchmark.md`), "
            "écrits par le code et commités avec lui (« benchmark: forest vs HGB on California, 3 seeds »), pour qu'on "
            "sache quel code a produit quels chiffres."),

    Ex("14.28", "🏆", 4, 100, "Défi : RMSE ≤ 0,45 sur California",
       "construire et régler un modèle tabulaire qui passe sous 0,45 de RMSE, sur le test et sur d'autres découpages.",
       "Ex 14.26 · ch. 8 (validation croisée) · ch. 12 (préparation des features) · fiche, section sur le gradient boosting",
       thread="California", tracks="C",
       body=r"""**Objectif** : une RMSE de test d'au plus **0,45** sur le découpage du défi (20 % de test, `random_state=42`, celui des parties B à D), **et** au plus 0,45 en moyenne sur trois autres découpages (`random_state` = 1, 2 et 3), ta méthode étant réentraînée sur chacun. Palier 🌟 : les deux sous 0,43. Écris `build_model_28()`, qui rend un modèle **neuf**, non entraîné (scikit-learn, XGBoost ou LightGBM, éventuellement un pipeline), qui reçoit les 8 features brutes de California (forme `(n, 8)`, dans l'ordre de `CAL_FEATURES`) et prédit `MedHouseVal`.

Règles : règle ta méthode par validation croisée (ou early stopping) sur les données d'entraînement, dans une cellule à toi ; les quatre jeux de test ne servent qu'à la mesure finale, révélée **une fois par session** (relancer la cellule redonne le même résultat ; redémarre le noyau pour une nouvelle tentative). Pour te situer : `HistGradientBoostingRegressor(random_state=0)`, avec ses réglages par défaut, fait 0,462 sur le test du défi et 0,466 en moyenne sur les trois autres. La vérification refuse un modèle déjà entraîné, ou le même objet rendu deux fois, signale une RMSE trop belle pour être honnête (sous 0,35) et réentraîne ta méthode une fois sur des cibles mélangées au hasard, sur lesquelles elle ne doit rien apprendre (sinon, ton modèle voit-il les lignes de test ?) ; le reste est une règle d'honneur : ne règle rien sur les découpages de la vérification, et ne fais pas oublier à la cellule ce qu'elle a révélé. Ce défi se corrige avec `05_solutions.md`.

Dans tes notes : les modèles et les réglages essayés, avec leurs scores de validation croisée ; ce qui a le plus aidé (les hyperparamètres, le learning rate et l'early stopping, de nouvelles features) ; le temps d'entraînement de ton modèle.""",
       todo=TODO_28, check=CHECK_28, solution=SOLUTION_28 + CHECK_28.replace('with checked("14.28"):\n', "if True:\n"),
       note="Le corrigé passe le défi et le palier : 0,423 sur le test du défi et 0,424 en moyenne ailleurs, en "
            "trois secondes environ par découpage. Deux choses l'expliquent, mesurées en validation croisée à 5 folds "
            "sur l'entraînement du défi (RMSE moyenne). Les **features** : un arbre ne coupe qu'un axe à la fois, et les "
            "prix de la Californie suivent des lignes obliques (la côte, les vallées) ; des coordonnées tournées de "
            "30°, 45° et 60° (et le revenu par pièce multiplié par l'occupation) les rendent faciles à découper : "
            "0,472 → 0,453 avec les réglages par défaut. Les **réglages** : un learning rate de 0,05, des arbres de "
            "63 feuilles au plus, une régularisation L2 et un early stopping patient (50 itérations) : 0,472 → "
            "0,457 sans les features. Les deux ensemble : 0,438. "
            "`05_solutions.md` donne les essais et la validation croisée qui ont mené à ces choix."),
])
PARTS = [PART_A, PART_B, PART_C, PART_D]
NEXT_SESSION = [
    ("A", "14.11 à 14.14", "faire voter trois classifieurs du ch. 13, le vote et le bootstrap dans "
                           "mylearn.ensemble, un jury de votants indépendants puis corrélés, voting et stacking"),
    ("B", "14.15 à 14.19", "forêts de scikit-learn sur California, nombre d'arbres face au bruit, BaggingClassifier, "
                           "RandomForestClassifier et ExtraTreesClassifier dans mylearn.ensemble, importance des "
                           "features"),
    ("C", "14.20 à 14.25", "courbes d'un boosting, learning rate et nombre d'arbres, frontières et débogage "
                           "d'AdaBoost, AdaBoostClassifier et GradientBoostingRegressor dans mylearn.ensemble"),
    ("D", "14.26 à 14.28", "HistGradientBoosting, XGBoost et LightGBM sur California, un benchmark reproductible, "
                           "le défi RMSE ≤ 0,45"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 14.Q1 à 14.Q11, 14.R1 à 14.R3, 14.1 à 14.3, 14.5 à 14.7 | vérifier tes réponses courtes | 🧠 🔁 ✏️ ∂ | "
            "★ à ★★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# 14 · Ensembles — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est "
               "pas fait affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch14_ensembles/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 14`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 14 · Ensembles — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Calculer à la main un vote pondéré, la probabilité qu'une majorité se trompe, la part des exemples "
               "oubliés par un bootstrap, une itération d'AdaBoost et deux itérations de gradient boosting.\n"
               "- Écrire et vérifier contre scikit-learn les ensembles de ta librairie, construits sur tes arbres du "
               "ch. 13 : `plurality_vote`, `bootstrap_indices`, `BaggingClassifier`, `RandomForestClassifier`, "
               "`ExtraTreesClassifier`, `AdaBoostClassifier` et `GradientBoostingRegressor`.\n"
               "- Utiliser les forêts de scikit-learn, HistGradientBoosting, XGBoost et LightGBM avec early "
               "stopping, et interpréter l'importance des features.\n"
               "- Choisir entre bagging et boosting, et rendre un benchmark reproductible.\n\n"
               "**Rappel express.** Vote pondéré : $\\hat y = \\arg\\max_k \\sum_m w_m\\,\\mathbb{1}[h_m(\\mathbf{x}) "
               "= k]$ ; majorité de $M$ votants indépendants, chacun juste avec la probabilité $q$ : $\\sum_{j > M/2} \\binom{M}{j} q^j (1 - q)^{M - j}$ "
               "; jamais tiré par un bootstrap : $(1 - 1/n)^n \\to e^{-1} \\approx 0{,}368$ ; variance d'une moyenne "
               "corrélée : $\\rho\\sigma^2 + (1 - \\rho)\\sigma^2/M$ ; AdaBoost (SAMME) : $\\varepsilon_m = \\sum_i "
               "w_i \\mathbb{1}[h_m(\\mathbf{x}_i) \\ne y_i] / \\sum_i w_i$, $\\alpha_m = \\eta\\big(\\ln\\frac{1 - "
               "\\varepsilon_m}{\\varepsilon_m} + \\ln(K - 1)\\big)$, $w_i \\leftarrow w_i e^{\\alpha_m}$ pour les mal "
               "classés ; gradient boosting : $F_0 = \\bar y$, $r_i = y_i - F_{m-1}(\\mathbf{x}_i)$, $F_m = F_{m-1} "
               "+ \\eta\\, h_m$. En scikit-learn : `sklearn.ensemble` (`BaggingClassifier`, "
               "`RandomForestClassifier`, `ExtraTreesClassifier`, `AdaBoostClassifier`, "
               "`GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `VotingClassifier`, "
               "`StackingClassifier`), `sklearn.inspection.permutation_importance` ; `xgboost.XGBRegressor`, "
               "`lightgbm.LGBMRegressor`.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 14.11 à 14.28 (parties A à D, ta librairie `ensemble`) seront ajoutés à la "
             "prochaine session de génération ; relance alors `python tools/start_chapter.py 14` pour obtenir le "
             "notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dire quand un ensemble bat ses membres, et pourquoi il ne corrige pas un biais commun ?\n"
               "2. Sais-tu dérouler à la main un vote pondéré, un tour d'AdaBoost et une étape de gradient "
               "boosting ?\n"
               "3. Sais-tu choisir entre une forêt et un gradient boosting, et régler le nombre d'arbres et le "
               "learning rate ?\n\n"
               "**Pour aller plus loin** : les articles de Breiman (1996, 2001), de Freund et Schapire (1997) et de "
               "Friedman (2001), et le guide utilisateur « Ensembles » de scikit-learn, cités dans la fiche. La "
               "suite : le ch. 15 (scikit-learn), qui mettra ces ensembles dans des pipelines et des recherches "
               "d'hyperparamètres." + later)]


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
