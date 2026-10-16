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
Parts A to D (exercises 14.11 to 14.28) come with the second generation session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Paper, badge, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

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
# Parts A to D (exercises 14.11 to 14.28): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
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
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch14_ensembles/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 14`) : ce fichier-ci est mis à jour par Claude.")
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
               "= k]$ ; majorité de $M$ votants indépendants : $\\sum_{j > M/2} \\binom{M}{j} p^j (1 - p)^{M - j}$ "
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
