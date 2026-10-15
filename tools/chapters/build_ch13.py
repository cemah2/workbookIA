#!/usr/bin/env python
"""Build the two notebooks of chapter 13 from a single source (used by Claude).

    python tools/chapters/build_ch13.py
    python tools/run_all_notebooks.py chapitres/ch13_classifieurs/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch13_classifieurs/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper
exercises 13.1 to 13.7 and of the 🧮 estimate 13.9 (02_exercices.md). The proof ∂ 13.8, the reflection
(🗣️ 13.10, ⚖️ 13.11) and the interview are corrected with 05_solutions.md.
Parts A to F (exercises 13.12 to 13.36) come with the second generation session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Paper, badge, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "13"
FOLDER = "chapitres/ch13_classifieurs"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np
from scipy.stats import norm


def gini(counts):
    p = np.asarray(counts, dtype=float) / np.sum(counts)
    return float(1 - np.sum(p ** 2))


def entropy_bits(probs):
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def posterior_first(l_first, l_second):
    """P(first class | x) from two joint log-probabilities."""
    return float(1 / (1 + math.exp(l_second - l_first)))


# R1: three penguins (flipper in mm, mass in g) and the training standard deviations
A_R1, B_R1, C_R1 = np.array([190.0, 3700.0]), np.array([210.0, 3750.0]), np.array([192.0, 3800.0])
SD_R1 = np.array([14.0, 800.0])

# 13.1: seven training samples, the query (0, 0)
X_131 = np.array([[1, 1], [-1.5, 0], [0, 2], [2, -1.5], [-2, -2], [3, 0], [0, -3.5]])
Y_131 = np.array([1, 0, 0, 1, 2, 2, 2])
D_131 = np.linalg.norm(X_131, axis=1)          # euclidean distances to (0, 0)
M_131 = np.abs(X_131).sum(axis=1)              # manhattan distances
ORDER_131 = np.argsort(D_131, kind="stable")


def vote_131(k, weighted=False):
    """Class shares of the k nearest samples (euclidean), uniform or 1/d votes."""
    idx = ORDER_131[:k]
    w = 1 / D_131[idx] if weighted else np.ones(k)
    share = np.array([w[Y_131[idx] == c].sum() for c in range(3)])
    return share / share.sum()


# 13.3: ten flippers, Adelie (0) and Gentoo (1)
FL_133 = np.array([181, 186, 190, 193, 197, 210, 205, 215, 217, 228])
SP_133 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
VALUES_133 = np.unique(FL_133)
THRESHOLDS_133 = (VALUES_133[:-1] + VALUES_133[1:]) / 2


def gain_133(s, impurity="gini"):
    left, right = SP_133[FL_133 <= s], SP_133[FL_133 > s]
    if impurity == "gini":
        f = lambda y: gini(np.bincount(y, minlength=2))
    else:
        f = lambda y: entropy_bits(np.bincount(y, minlength=2) / len(y))
    return f(SP_133) - len(left) / 10 * f(left) - len(right) / 10 * f(right)


BEST_133 = float(max(THRESHOLDS_133, key=gain_133))

# 13.4: six points, w = (0.5, 0.5), b = -2
P_134 = np.array([[3, 3], [5, 4], [4, 5], [1, 1], [0, 1], [2, 0]], dtype=float)
Y_134 = np.array([1, 1, 1, -1, -1, -1])
W_134, B_134 = np.array([0.5, 0.5]), -2.0
F_134 = P_134 @ W_134 + B_134
NORM_134 = float(np.linalg.norm(W_134))

# 13.5: x = (1, 2), z = (3, 1)
X_135, Z_135 = np.array([1.0, 2.0]), np.array([3.0, 1.0])
PHI = lambda v: np.array([v[0] ** 2, math.sqrt(2) * v[0] * v[1], v[1] ** 2])
K_135 = lambda a, b: float(a @ b) ** 2

# 13.6: letter counts e, h, q, w
EN_136, FR_136 = np.array([30, 12, 1, 7]), np.array([38, 4, 8, 0])
TH_EN = (EN_136 + 1) / (EN_136.sum() + 4)
TH_FR = (FR_136 + 1) / (FR_136.sum() + 4)
WHERE, QUE = np.array([2, 1, 0, 1]), np.array([1, 0, 1, 0])
L_EN = math.log(0.6) + WHERE @ np.log(TH_EN)
L_FR = math.log(0.4) + WHERE @ np.log(TH_FR)

# 13.7: Gaussian naive Bayes, flipper and bill depth
G_137 = np.array([210.0, 215.0, 220.0, 225.0])
LA_FL, LP_FL = norm.logpdf(205, 190, 6.5), norm.logpdf(205, 217, 6.5)
LA_BD, LP_BD = norm.logpdf(16.5, 18.3, 1.2), norm.logpdf(16.5, 15.0, 1.0)
JA_137 = math.log(0.55) + LA_FL + LA_BD
JP_137 = math.log(0.45) + LP_FL + LP_BD

# 13.9: MNIST, 60 000 training and 10 000 test images of 784 pixels
N_TRAIN, N_TEST, P_PIX = 60_000, 10_000, 784'''

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("13.Q1", "Pourquoi essayer des classifieurs classiques d'abord ?", [
        ("a", "True or False", "True",
         r'''mistakes={"relis le §13.1 : à quoi sert un score obtenu en quelques secondes, quand on envisage un modèle coûteux ?": False}'''),
        ("b", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"sur 2 000 lignes, que t'apprend un modèle aussi coûteux, et que coûte-t-il ?": "A",
          "attendre d'autres données n'apprend rien sur celles qu'on a : que peut-on mesurer dès maintenant ?": "C",
          "un réglage fixé au hasard et aucune évaluation : comment saurais-tu ce que vaut le modèle ?": "D"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"quelque chose garantit-il qu'un modèle plus complexe fasse mieux sur des données de cette taille ?": True}'''),
        ("d", "True or False", "True",
         r'''mistakes={"qu'apprends-tu si un kNN bat nettement une régression logistique sur les mêmes données ?": False}'''),
    ]),
    Paper("13.Q2", "Paramétrique ou non paramétrique ?", [
        ("a", 'six letters P or N, in the order of the items, e.g. "XYXYXY"', '"PNPNPN"',
         r'''mistakes={"élément 6 (le SVM RBF) : relis la formule de prédiction du §13.4 ; de quoi est faite la somme, et ce nombre dépend-il de n ?": "PNPNPP",
          "élément 3 (Naïve Bayes) : combien de nombres garde-t-il, et ce nombre dépend-il de n ?": "PNNNPN",
          "élément 4 (l'arbre sans limite) : que devient son nombre de feuilles quand les données bruitées doublent ?": "PNPPPN",
          "élément 5 (le SVM linéaire) : que garde-t-il après l'entraînement, à part w et b ?": "PNPNNN",
          "élément 1 (la régression logistique) : combien de nombres garde-t-elle ?": "NNPNPN",
          "tu as inversé les deux familles : relis la définition du §13.2": "NPNPNP"}'''),
        ("b", "the letter of your choice", '"B"',
         r'''mistakes={"que garde ce modèle après l'entraînement : des poids, ou les exemples ?": "A",
          "que garde ce modèle après l'entraînement : des statistiques par classe, ou les exemples ?": "C",
          "que garde ce modèle après l'entraînement : un hyperplan, ou les exemples ?": "D"}'''),
        ("c", "True or False", "True",
         r'''mistakes={"relis la définition d'un modèle paramétrique : son nombre de paramètres dépend-il de n ?": False}'''),
        ("d", "True or False", "False",
         r'''mistakes={"un arbre ou un SVM à noyau n'apprend-il vraiment aucun nombre ? Relis la définition d'un classifieur non paramétrique (§13.2)": True}'''),
    ]),
    Paper("13.Q3", "kNN n'est pas k-means", [
        ("a", "the letter of your choice", '"A"',
         r'''mistakes={"un algorithme non supervisé regarde-t-il les labels ?": "B",
          "l'un des deux est non supervisé : lequel ?": "C",
          "que fait voter le kNN ?": "D"}'''),
        ("b", "the letter of your choice", '"B"',
         r'''mistakes={"c'est le k du k-means ; relis le pseudo-code du kNN": "A",
          "le nombre de features est p ; relis le pseudo-code du kNN": "C",
          "le nombre de classes ne se choisit pas ; relis le pseudo-code du kNN": "D"}'''),
        ("c", "True or False", "True",
         r'''mistakes={"relis ce que fait fit dans le pseudo-code du kNN : quel calcul y vois-tu ?": False}'''),
        ("d", "True or False", "True",
         r'''mistakes={"un algorithme non supervisé connaît-il le nom des classes ?": False}'''),
    ]),
    Paper("13.Q4", "Petit k, grand k", [
        ("a", "a number", "1.0",
         r'''decimals=2, mistakes={"sans doublon, quel est le plus proche voisin d'un point d'entraînement, cherché dans l'entraînement ?": 0.0,
          "une accuracy s'écrit ici entre 0 et 1, pas en pourcentage": 100.0,
          "sans doublon, sur quel point d'entraînement un 1-NN pourrait-il se tromper ?": 0.99}'''),
        ("b", "the letter of your choice", '"C"',
         r'''mistakes={"avec k = 1, un point du groupe a-t-il un voisin du groupe ? Qui gagne le vote ?": "A",
          "avec k = 3, combien de voisins d'un point du groupe peuvent venir du groupe, sur 6 points ?": "B"}'''),
        ("c", "the letter of your choice", '"B"',
         r'''mistakes={"avec k = n, combien d'exemples votent, et le résultat dépend-il encore du point ?": "A",
          "le vote est déterministe : qui vote quand k = n ?": "C",
          "k = n est permis : qui vote alors ?": "D"}'''),
        ("d", "True or False", "False",
         r'''mistakes={"quelle valeur de k maximise toujours l'accuracy d'entraînement, et que dit-elle de la généralisation ?": True}'''),
        ("e", "True or False", "True",
         r'''mistakes={"avec deux classes, un nombre impair de votes peut-il se partager en deux moitiés égales ?": False}'''),
    ]),
    Paper("13.Q5", "Marge, vecteurs de support et C", [
        ("a", "True or False", "True",
         r'''mistakes={"relis la définition des vecteurs de support : quels points fixent la solution ?": False}'''),
        ("b", "the letter of your choice", '"B"',
         r'''mistakes={"C multiplie les violations dans la loss : quand elles coûtent plus cher, l'optimisation en tolère-t-elle plus ?": "A",
          "C est un poids de la loss : peut-il ne rien changer ?": "C"}'''),
        ("c", "the letter of your choice", '"B"',
         r'''mistakes={"avec une marge souple, que deviennent les points dans la bande ou du mauvais côté ? Relis le §13.4": "A",
          "un point loin de la frontière, du bon côté, influence-t-il la solution ?": "C"}'''),
        ("d", "True or False", "True",
         r'''mistakes={"le noyau RBF dépend d'une distance : une feature en grands nombres la dominerait-elle ?": False}'''),
    ]),
    Paper("13.Q6", "Ce que l'astuce du noyau évite de calculer", [
        ("a", "the letter of your choice", '"B"',
         r'''mistakes={"c'est justement ce que l'astuce évite de construire : relis l'encadré 🧮 sur l'astuce du noyau": "A",
          "le noyau calcule une ressemblance dans l'espace relevé, pas seulement dans l'espace de départ": "C",
          "le noyau ne choisit rien : relis l'encadré ⚠️ sur l'astuce du noyau": "D"}'''),
        ("b", "a whole number", "36",
         r'''mistakes={"relis la définition du noyau polynomial : où est le degré 2 ?": 6}'''),
        ("c", "a number", "1",
         r'''decimals=2, mistakes={"quelle est la distance d'un point à lui-même ? Remplace-la dans la formule": 0}'''),
        ("d", "True or False", "True",
         r'''mistakes={"avec un grand γ, la ressemblance chute-t-elle vite ou lentement avec la distance ? Relis la figure du noyau RBF": False}'''),
        ("e", "True or False", "False",
         r'''mistakes={"qui choisit le noyau, C et γ ? Relis l'encadré ⚠️ sur l'astuce du noyau": True}'''),
    ]),
    Paper("13.Q7", "Racine, feuilles, profondeur : le vocabulaire des arbres", [
        ("a", "a whole number", "3",
         r'''mistakes={"compte aussi la feuille de la branche « non » de la racine": 2,
          "une feuille n'a pas d'enfant : la racine et le second nœud n'en sont pas": 5}'''),
        ("b", "a whole number", "2",
         r'''mistakes={"la profondeur compte les arêtes, pas les nœuds, du chemin le plus long": 3,
          "la profondeur se lit sur le chemin le plus long, pas le plus court": 1}'''),
        ("c", "a whole number", "2",
         r'''mistakes={"un nœud interne pose une question ; les feuilles n'en posent pas": 5,
          "compte aussi la racine": 1}'''),
        ("d", "a whole number", "8",
         r'''mistakes={"combien de feuilles au plus à la profondeur 1, puis 2 ? Que fait chaque niveau de plus ?": 6,
          "à la profondeur 3, combien de niveaux d'arêtes y a-t-il sous la racine ?": 16}'''),
        ("e", "True or False", "True",
         r'''mistakes={"combien d'arêtes sépare la racine d'elle-même ?": False}'''),
    ]),
    Paper("13.Q8", "Pourquoi un arbre surapprend-il ?", [
        ("a", "a number", "1.0",
         r'''decimals=2, mistakes={"un arbre sans limite s'arrête-t-il avant d'avoir des feuilles pures ?": 0.5,
          "une accuracy s'écrit ici entre 0 et 1, pas en pourcentage": 100.0,
          "sur quel exemple d'entraînement un arbre aux feuilles pures pourrait-il se tromper ?": 0.99}'''),
        ("b", "True or False", "False",
         r'''mistakes={"une question « x_j ≤ s ? » dépend-elle de l'unité de x_j, ou seulement de l'ordre de ses valeurs ?": True}'''),
        ("c", "the letter of your choice", '"B"',
         r'''mistakes={"relis l'expérience du livre sur deux tirages de 70 % de données bruitées (§13.5.1)": "A",
          "si la première question change, que devient le reste de l'arbre ?": "C"}'''),
        ("d", "the letter of your choice", '"B"',
         r'''mistakes={"une question ne regarde qu'une feature : peut-elle tracer une droite oblique ?": "A",
          "une question ne regarde qu'une feature : peut-elle tracer une courbe ?": "C",
          "une seule droite, c'est un arbre de profondeur 1 ; et ensuite ?": "D"}'''),
    ]),
    Paper("13.Q9", "Gini ou gain d'information ?", [
        ("a", "a number", "gini([1, 1])",
         r'''decimals=2, mistakes={"relis la formule de Gini : 1 − Σ p²": 1}'''),
        ("b", "a number", "entropy_bits([0.5, 0.5])",
         r'''decimals=2, mistakes={"relis la formule de l'entropie : −Σ p log₂ p": 0.5, "en bits, le logarithme est en base 2": 0.69}'''),
        ("c", "a number", "gini([1, 0])",
         r'''decimals=2, mistakes={"dans un nœud pur, que vaut la proportion de sa classe, et donc Σ p² ?": 1}'''),
        ("d", "the letter of your choice", '"B"',
         r'''mistakes={"sans pondération, un enfant d'un seul exemple compterait autant que l'autre : relis la formule du gain": "A",
          "relis la formule du gain : quelle fraction multiplie I(gauche) ?": "C",
          "relis la formule du gain : le nombre de classes y apparaît-il ?": "D"}'''),
        ("e", "True or False", "True",
         r'''mistakes={"relis le paragraphe « Gini ou entropie ? » et compare les courbes de la figure des impuretés": False}'''),
    ]),
    Paper("13.Q10", "Limiter la profondeur ou élaguer ?", [
        ("a", 'five letters P or E, in the order of the items, e.g. "XYXYX"', '"PPEPE"',
         r'''mistakes={"élément 3 (ccp_alpha) : agit-il pendant la croissance, ou sur l'arbre complet ? Relis le §13.5.3": "PPPPE",
          "élément 5 : on retire des sous-arbres d'un arbre déjà construit ; pendant ou après la croissance ?": "PPEPP",
          "élément 2 (min_samples_leaf) : interdit-il une coupe pendant la croissance, ou retire-t-il une branche après ?": "PEEPE",
          "élément 4 (min_samples_split) : interdit-il une coupe pendant la croissance, ou retire-t-il une branche après ?": "PPEEE",
          "élément 1 (max_depth) : interdit-il une coupe pendant la croissance, ou retire-t-il une branche après ?": "EPEPE",
          "tu as inversé les deux familles : relis les deux paragraphes du §13.5.3": "EEPEP"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"dans R_α(T) = R(T) + α|T|, chaque feuille coûte α : quand α grandit, un arbre à beaucoup de feuilles devient-il plus intéressant ?": False}'''),
        ("c", "the letter of your choice", '"B"',
         r'''mistakes={"un réglage choisi sur le test rend le score de test optimiste (ch. 8)": "A",
          "le plus profond, c'est l'arbre sans limite : relis le §13.5.1": "C",
          "quelle profondeur maximise toujours l'accuracy d'entraînement ?": "D"}'''),
    ]),
    Paper("13.Q11", "Ce qui est vraiment « naïf » dans Naïve Bayes", [
        ("a", "the letter of your choice", '"B"',
         r'''mistakes={"c'est l'explication du livre ; relis l'encadré ⚠️ sur ce qui est « naïf »": "A",
          "la forme gaussienne est un choix de modèle, il y a d'autres variantes : quelle hypothèse leur est commune ?": "C",
          "les priors sont les fréquences des classes : quelle hypothèse permet le produit des P(x_j | c) ?": "D"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"d'où viennent les moyennes et les variances des gaussiennes ?": False}'''),
        ("c", "a whole number", "2 * 4 * 3",
         r'''mistakes={"tu as compté une loi par classe et par feature : combien de nombres faut-il pour une loi normale ?": 12,
          "compte une loi par classe et par feature, pas par feature seulement": 8,
          "relis la question : quels nombres compte-t-on, et lesquels sont exclus ?": 27}'''),
        ("d", "True or False", "False",
         r'''mistakes={"des features corrélées comptent plusieurs fois la même information : vers où cela pousse-t-il les probabilités ?": True}'''),
        ("e", "True or False", "True",
         r'''mistakes={"que vaut le logarithme de 0, et un produit qui contient un facteur nul ?": False}'''),
    ]),
    Paper("13.Q12", "Quel classifieur pour quelle situation ?", [
        ("a", 'five letters A, N, K, L, S, in the order of the situations, e.g. "AAAAA"', '"ANKLS"',
         r'''mistakes={"situations 3 et 5 : lequel intègre un exemple sans réentraînement, et lequel a deux hyperparamètres à régler pour une frontière non linéaire ?": "ANSLK",
          "situations 1 et 4 : lequel donne des règles « si … alors … », et lequel un poids par feature et des probabilités fiables ?": "LNKAS",
          "situations 2 et 4 : lequel est la baseline de quelques secondes sur des comptes de mots ?": "ALKNS",
          "situations 4 et 5 : lequel donne un poids par feature et des probabilités fiables, et lequel a deux hyperparamètres pour une frontière non linéaire ?": "ANKSL"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"avec des folds différents, l'écart entre deux scores mesure-t-il seulement la différence entre les modèles ?": False}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("13.R1", "Pourquoi standardiser avant de mesurer des distances", [
        ("a", "a number with 2 decimals", "float(np.linalg.norm(C_R1 - A_R1))",
         r'''decimals=2, mistakes={"tu as la distance au carré : prends sa racine": 10004,
          "c'est la distance de Manhattan ; l'euclidienne prend la racine de la somme des carrés": 102}'''),
        ("b", "a number with 2 decimals", "float(np.linalg.norm(C_R1 - B_R1))",
         r'''decimals=2, mistakes={"tu as la distance au carré : prends sa racine": 2824,
          "c'est la distance de Manhattan ; l'euclidienne prend la racine de la somme des carrés": 68}'''),
        ("c", 'the letter "A" or "B"', '"B"',
         r'''mistakes={"compare les deux distances brutes que tu viens de calculer": "A"}'''),
        ("d", "a number with 3 decimals", "float(np.linalg.norm((C_R1 - A_R1) / SD_R1))",
         r'''decimals=3, mistakes={"tu as la distance au carré : prends sa racine": 0.036,
          "divise chaque différence par l'écart-type de sa propre feature": 7.143,
          "vérifie quel manchot tu as pris : la question porte sur C et A": 1.287}'''),
        ("e", 'the letter "A" or "B"', '"A"',
         r'''mistakes={"calcule aussi la distance standardisée entre C et B, puis compare": "B"}'''),
        ("f", "True or False", "True",
         r'''mistakes={"relis la règle d'or du ch. 12 : où s'apprennent les paramètres d'un scaler ?": False}'''),
    ]),
    Paper("13.R2", "L'équation de la frontière d'un perceptron", [
        ("a", "0 or 1", "1",
         r'''mistakes={"calcule w·x + b pour cette entrée : quel est son signe, et que sort alors le perceptron ?": 0, "la sortie d'un perceptron vaut 0 ou 1": 2}'''),
        ("b", "0 or 1", "0",
         r'''mistakes={"calcule w·x + b pour cette entrée : quel est son signe, et que sort alors le perceptron ?": 1, "la sortie d'un perceptron vaut 0 ou 1": -1}'''),
        ("c", "the letter of your choice", '"A"',
         r'''mistakes={"le OU sort 1 dès qu'une entrée vaut 1 : est-ce le cas pour (0, 1) ?": "B",
          "le OU exclusif n'est pas linéairement séparable (ch. 10) : un perceptron peut-il le réaliser ?": "C",
          "le NON-ET sort 1 pour (0, 0) : est-ce le cas ici ?": "D"}'''),
        ("d", "a number", "-1",
         r'''decimals=2, mistakes={"isole x₂ dans x₁ + x₂ − 1,5 = 0 : quel coefficient multiplie x₁ ?": 1}'''),
        ("e", "a number", "1.5",
         r'''decimals=2, mistakes={"isole x₂ dans x₁ + x₂ − 1,5 = 0 : quel est le signe de l'ordonnée à l'origine ?": -1.5}'''),
        ("f", "a number with 3 decimals", "0.5 / math.sqrt(2)",
         r'''decimals=3, mistakes={"as-tu divisé par la norme de w ?": 0.5, "recalcule la norme de (1, 1)": 0.25}'''),
        ("g", "True or False", "True",
         r'''mistakes={"pour quelle valeur de z a-t-on σ(z) = 0,5 ?": False}'''),
    ]),
    Paper("13.R3", "Entropie de deux distributions", [
        ("a", "a number", "entropy_bits([0.5, 0.25, 0.25])",
         r'''decimals=2, mistakes={"en bits, log₂ 0,25 vaut −2 : vérifie ce terme": 1, "le logarithme est en base 2, pas népérien": 1.04}'''),
        ("b", "a number with 3 decimals", "entropy_bits([0.8, 0.2])",
         r'''decimals=3, mistakes={"le logarithme est en base 2, pas népérien": 0.5, "le signe moins rend l'entropie positive": -0.722}'''),
        ("c", "a number", "2.0",
         r'''decimals=2, mistakes={"c'est le nombre de valeurs ; l'entropie maximale est son logarithme en base 2": 4.0, "en bits, le logarithme est en base 2": 1.39}'''),
        ("d", "a number with 3 decimals", "float(-0.5 * math.log2(0.8) - 0.5 * math.log2(0.2))",
         r'''decimals=3, mistakes={"dans H(p, q), quels nombres sont les poids, et lesquels sont dans le log ?": 0.722,
          "dans H(p, q), les logarithmes sont-ils ceux des p_i ou des q_i ?": 1,
          "le logarithme est en base 2, pas népérien": 0.916}'''),
        ("e", "True or False", "True",
         r'''mistakes={"que vaut H(p, q) − H(p), et quel est son signe (ch. 6) ?": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("13.1", "Vote des k plus proches voisins à la main", [
        ("a", "an index from 0 to 6", "int(ORDER_131[0])",
         r'''mistakes={"as-tu calculé les distances euclidiennes, et numéroté les exemples à partir de 0 ?": 1}'''),
        ("b", "a class: 0, 1 or 2", "int(Y_131[ORDER_131[0]])",
         r'''mistakes={"refais le calcul en distance euclidienne": 0, "lis la classe du plus proche voisin dans le tableau": 2}'''),
        ("c", "a class: 0, 1 or 2", "int(np.argmax(vote_131(3)))",
         r'''mistakes={"compte les classes des trois plus proches, pas seulement celle du premier": 1}'''),
        ("d", "a number with 3 decimals", "float(vote_131(3)[0])",
         r'''decimals=3, mistakes={"c'est la part de l'autre classe représentée parmi les trois voisins": 0.333}'''),
        ("e", "a class: 0, 1 or 2", "int(np.argmax(vote_131(5)))",
         r'''mistakes={"compte les votes des cinq plus proches : y a-t-il une égalité, et quelle classe la gagne dans scikit-learn ?": 1,
          "la classe 2 n'a qu'un vote parmi les cinq plus proches : compte à nouveau": 2}'''),
        ("f", "a class: 0, 1 or 2", "int(np.argmax(vote_131(7)))",
         r'''mistakes={"avec k = 7, tous les exemples votent : compte les classes de tout l'entraînement": 0,
          "avec k = 7, tous les exemples votent : compte à nouveau les classes de tout l'entraînement": 1}'''),
        ("g", "a number with 3 decimals", "float(vote_131(3, weighted=True)[1])",
         r'''decimals=3, mistakes={"c'est la part des votes uniformes ; pondère chaque voisin par 1/d": 0.333,
          "divise par la somme des poids, pas par k": 0.236,
          "tu as la part de l'autre classe": 0.623}'''),
        ("h", "a class: 0, 1 or 2", "int(np.argmax(vote_131(7, weighted=True)))",
         r'''mistakes={"as-tu pondéré chaque vote par 1/d, puis additionné les poids de chaque classe ?": 2,
          "additionne aussi le poids du second exemple de cette classe, puis compare les sommes": 1}'''),
        ("i", "a list [index, class]", "[int(np.argsort(M_131, kind='stable')[0]), int(Y_131[np.argsort(M_131, kind='stable')[0]])]",
         r'''mistakes={"en Manhattan, additionne |x₁| + |x₂| pour chaque exemple": [0, 1]}'''),
    ]),
    Paper("13.2", "Gini et entropie d'une feuille", [
        ("a", "a number", "gini([10, 90])",
         r'''decimals=2, mistakes={"c'est la somme des carrés ; Gini, c'est 1 moins cette somme": 0.82,
          "relis la formule de Gini : 1 − Σ p²": 0.1}'''),
        ("b", "a number with 3 decimals", "entropy_bits([0.1, 0.9])",
         r'''decimals=3, mistakes={"le logarithme est en base 2, pas népérien": 0.325}'''),
        ("c", "a number", "0.1",
         r'''decimals=2, mistakes={"en répondant toujours B, sur quels exemples se trompe-t-on ?": 0.9}'''),
        ("d", "a number", "0.9",
         r'''decimals=2, mistakes={"en répondant toujours A, sur quels exemples se trompe-t-on ? (Le livre se trompe ici, p. 527.)": 0.1}'''),
        ("e", "a number with 3 decimals", "0.1 * 0.9 + 0.9 * 0.1",
         r'''decimals=3, mistakes={"on se trompe de deux façons (vrai A et tiré B, vrai B et tiré A) : additionne-les": 0.09,
          "c'est la probabilité de tomber juste ; on demande celle de se tromper": 0.82}'''),
        ("f", "a number", "gini([50, 30, 20])",
         r'''decimals=2, mistakes={"c'est la somme des carrés ; Gini, c'est 1 moins cette somme": 0.38}'''),
        ("g", "a number with 3 decimals", "entropy_bits([0.5, 0.3, 0.2])",
         r'''decimals=3, mistakes={"le logarithme est en base 2, pas népérien": 1.03}'''),
        ("h", "a number with 3 decimals", "gini([1, 1, 1])",
         r'''decimals=3, mistakes={"c'est le maximum à deux classes ; que vaut chaque proportion avec trois classes à égalité ?": 0.5}'''),
        ("i", "a number with 3 decimals", "math.log2(3)",
         r'''decimals=3, mistakes={"le logarithme est en base 2, pas népérien": 1.099, "c'est le maximum à deux classes": 1}'''),
    ]),
    Paper("13.3", "Choisir le meilleur seuil sur la longueur des nageoires", [
        ("a", "a whole number", "len(THRESHOLDS_133)",
         r'''mistakes={"compte les intervalles entre valeurs consécutives, pas les valeurs": 10}'''),
        ("b", "a number", "gini([6, 4])",
         r'''decimals=2, mistakes={"c'est la somme des carrés ; Gini, c'est 1 moins cette somme": 0.52}'''),
        ("c", "a number", "gain_133(201)",
         r'''decimals=2, mistakes={"pondère l'impureté de chaque enfant par sa part des exemples (5 sur 10)": 0.16}'''),
        ("d", "a number with 3 decimals", "gain_133(212.5)",
         r'''decimals=3, mistakes={"pondère l'impureté de chaque enfant par sa part des exemples": 0.235}'''),
        ("e", "a number with 3 decimals", "gain_133(207.5)",
         r'''decimals=3, mistakes={"pondère l'impureté de chaque enfant par sa part des exemples": -0.173}'''),
        ("f", "a threshold in mm", "BEST_133",
         r'''decimals=1, mistakes={"calcule les gains au lieu de prendre le seuil « au milieu » des deux espèces": 207.5,
          "as-tu calculé le gain de tous les seuils candidats ?": 212.5}'''),
        ("g", "a number with 3 decimals", "gain_133(201, impurity='entropy')",
         r'''decimals=3, mistakes={"pondère l'entropie de chaque enfant par sa part des exemples": 0.249,
          "le logarithme est en base 2, pas népérien": 0.423}'''),
        ("h", "a whole number", "1",
         r'''mistakes={"chaque feuille prédit sa classe majoritaire : à droite du seuil, combien de manchots ne l'ont pas ?": 0}'''),
        ("i", "0 (Adélie) or 1 (Papou)", "1",
         r'''mistakes={"seul compte le côté du seuil : 203 est-il au plus égal au seuil ?": 0, "réponds 0 (Adélie) ou 1 (Papou)": 2, "réponds 0 (Adélie) ou 1 (Papou) ; quelle classe prédit la feuille où tombe 203 mm ?": -1}'''),
        ("j", "a whole number", "int(sum(5 <= (FL_133 <= s).sum() <= 5 for s in THRESHOLDS_133))",
         r'''mistakes={"chaque enfant doit recevoir au moins 5 manchots : sur 10, quel découpage reste possible ?": 0,
          "vérifie aussi l'enfant de droite": 5}'''),
    ]),
    Paper("13.4", "Distance à une droite et marge d'un SVM", [
        ("a", "a number with 4 decimals", "NORM_134",
         r'''decimals=4, mistakes={"c'est la norme au carré : prends sa racine": 0.5, "c'est la norme de Manhattan ; prends la racine de la somme des carrés": 1}'''),
        ("b", "a number", "float(F_134[1])",
         r'''decimals=2, mistakes={"n'oublie pas b = −2": 4.5}'''),
        ("c", "a number with 3 decimals", "float(abs(F_134[1]) / NORM_134)",
         r'''decimals=3, mistakes={"as-tu divisé par la norme de w ?": 2.5, "divise par la norme de w, pas par son carré": 5}'''),
        ("d", "a number", "float(np.min(Y_134 * F_134))",
         r'''decimals=2, mistakes={"multiplie f par y : pour les points de la classe −1, le signe change": -1.5}'''),
        ("e", "a list of indices, in increasing order", "[int(i) for i in np.flatnonzero(np.isclose(Y_134 * F_134, 1))]",
         r'''mistakes={"le point 4 vérifie y f = 1,5 : est-il sur le bord de la bande ?": [0, 3, 4, 5],
          "calcule y f pour les points de la classe −1 aussi": [0]}'''),
        ("f", "a number with 3 decimals", "2 / NORM_134",
         r'''decimals=3, mistakes={"c'est la demi-largeur 1/‖w‖ ; la bande va de f = −1 à f = +1": 1.414}'''),
        ("g", "+1 or -1", "1",
         r'''mistakes={"calcule f(2, 3) et regarde son signe": -1, "réponds +1 ou −1 : le signe de f": 0}'''),
        ("h", "True or False", "True",
         r'''mistakes={"compare |f(2, 3)| à 1": False}'''),
        ("i", "True or False", "False",
         r'''mistakes={"le numérateur |f| et le dénominateur ‖w‖ sont multipliés par le même nombre : que devient le rapport ?": True}'''),
        ("j", "True or False", "True",
         r'''mistakes={"le point 1 est-il sur le bord de la bande ?": False}'''),
    ]),
    Paper("13.5", "L'astuce du noyau vérifiée à la main", [
        ("a", "a number", "float(X_135 @ Z_135)",
         r'''decimals=2, mistakes={"multiplie les coordonnées de même rang : la première avec la première, la seconde avec la seconde": 7}'''),
        ("b", "a number", "K_135(X_135, Z_135)",
         r'''decimals=2, mistakes={"relis la définition du noyau polynomial : où est le degré 2 ?": 5}'''),
        ("c", "a number with 3 decimals", "float(PHI(X_135)[1])",
         r'''decimals=3, mistakes={"n'oublie pas le facteur √2": 2}'''),
        ("d", "a number", "float(PHI(X_135) @ PHI(Z_135))",
         r'''decimals=2, mistakes={"n'oublie pas le facteur √2 dans la deuxième coordonnée de chaque vecteur": 19}'''),
        ("e", "a number", "K_135(X_135, X_135) - 2 * K_135(X_135, Z_135) + K_135(Z_135, Z_135)",
         r'''decimals=2, mistakes={"c'est la distance au carré dans l'espace de départ ; on la demande dans l'espace relevé": 5,
          "n'oublie pas le facteur 2 devant K(x, z)": 100}'''),
        ("f", "a number with 4 decimals", "math.exp(-0.5 * float((X_135 - Z_135) @ (X_135 - Z_135)))",
         r'''decimals=4, mistakes={"prends le carré de la distance dans l'exposant": 0.3269, "prends le carré de la distance dans l'exposant, pas la distance": 0.327,
          "n'oublie pas le signe moins dans l'exposant": 12.1825}'''),
        ("g", "a number with 2 decimals", "math.exp(-1.5 * float((X_135 - Z_135) @ (X_135 - Z_135))) * 1e5",
         r'''decimals=2, mistakes={"multiplie le résultat par 100 000": 0.0}'''),
        ("h", "a whole number", "1000 + 1000 * 999 // 2",
         r'''mistakes={"compte chaque paire j < l une seule fois": 1_000_000,
          "ajoute les 1 000 carrés x_j²": 499_500,
          "il n'y a pas de paire j = l parmi les produits croisés": 501_500}'''),
    ]),
    Paper("13.6", "Anglais ou français ? Naïve Bayes multinomial sur des lettres", [
        ("a", "a number", "0.0",
         r'''decimals=2, mistakes={"sans lissage, compte de w en français divisé par le total français : combien de w en français ?": 0.02}'''),
        ("b", "a number with 3 decimals", "float(TH_EN[3])",
         r'''decimals=3, mistakes={"c'est la valeur sans lissage": 0.14,
          "le total gagne α p = 4, pas 1": 0.157,
          "le compte gagne aussi α = 1": 0.13}'''),
        ("c", "a number with 4 decimals", "float(TH_FR[3])",
         r'''decimals=4, mistakes={"la question est posée avec lissage : le compte gagne-t-il α ?": 0, "le total gagne α p = 4, pas 1": 0.0196}'''),
        ("d", "a number with 4 decimals", "float(TH_FR[0])",
         r'''decimals=4, mistakes={"c'est la valeur sans lissage": 0.76, "le total gagne α p = 4, pas 1": 0.7647}'''),
        ("e", "a number with 3 decimals", "float(L_EN)",
         r'''decimals=3, mistakes={"multiplie log θ_e par le compte de e, 2": -4.399,
          "n'oublie pas le log du prior": -4.443,
          "les logarithmes sont népériens": -2.152}'''),
        ("f", "a number with 3 decimals", "float(L_FR)",
         r'''decimals=3, mistakes={"multiplie log θ_e par le compte de e, 2": -7.61,
          "n'oublie pas le log du prior": -7.019,
          "les logarithmes sont népériens": -3.447}'''),
        ("g", "a number with 3 decimals", "posterior_first(L_EN, L_FR)",
         r'''decimals=3, mistakes={"tu as le posterior de l'autre langue": 0.048,
          "normalise les probabilités, pas les log-probabilités": 0.384}'''),
        ("h", "a number", "0.0",
         r'''decimals=2, mistakes={"quelle langue la question vise-t-elle ? Reprends ta réponse a) dans la vraisemblance de cette langue": 1.0,
          "la question est posée sans lissage : que vaut P(w | FR) ?": 0.05}'''),
        ("i", "a number with 3 decimals", "posterior_first(math.log(0.6) + QUE @ np.log(TH_EN), math.log(0.4) + QUE @ np.log(TH_FR))",
         r'''decimals=3, mistakes={"tu as le posterior de l'autre langue": 0.791,
          "n'oublie pas les priors 0,6 et 0,4": 0.15}'''),
    ]),
    Paper("13.7", "Posterior d'un Naïve Bayes gaussien", [
        ("a", "a number", "float(G_137.mean())",
         r'''decimals=2, mistakes={}'''),
        ("b", "a number", "float(G_137.var())",
         r'''decimals=2, mistakes={"Naïve Bayes divise par n, pas par n − 1": 41.67, "c'est l'écart-type ; on demande la variance": 5.59}'''),
        ("c", "a number with 3 decimals", "float(LA_FL)",
         r'''decimals=3, mistakes={"dans 2πσ², σ² est la variance 6,5², pas l'écart-type": -4.518,
          "n'oublie pas le facteur ½ devant le logarithme": -8.244,
          "n'oublie pas le terme −½ log(2πσ²)": -2.663}'''),
        ("d", "a number with 3 decimals", "float(LP_FL)",
         r'''decimals=3, mistakes={"dans 2πσ², σ² est la variance 6,5², pas l'écart-type": -3.559,
          "n'oublie pas le facteur ½ devant le logarithme": -7.286,
          "n'oublie pas le terme −½ log(2πσ²)": -1.704}'''),
        ("e", "a number with 3 decimals", "float(LA_BD)",
         r'''decimals=3, mistakes={"dans 2πσ², σ² est la variance 1,2², pas l'écart-type": -2.135,
          "n'oublie pas le terme −½ log(2πσ²)": -1.125}'''),
        ("f", "a number with 3 decimals", "float(LP_BD)",
         r'''decimals=3, mistakes={"n'oublie pas le facteur ½ devant le logarithme": -2.963,
          "n'oublie pas le terme −½ log(2π)": -1.125}'''),
        ("g", "a number with 3 decimals", "float(JA_137)",
         r'''decimals=3, mistakes={"n'oublie pas le log du prior": -7.68,
          "additionne le log du prior, ne le multiplie pas": 4.591}'''),
        ("h", "a number with 3 decimals", "float(JP_137)",
         r'''decimals=3, mistakes={"n'oublie pas le log du prior": -6.539,
          "additionne le log du prior, ne le multiplie pas": 5.221}'''),
        ("i", "a number with 3 decimals", "posterior_first(JP_137, JA_137)",
         r'''decimals=3, mistakes={"tu as le posterior de l'autre espèce": 0.281,
          "n'oublie pas les priors 0,55 et 0,45": 0.758}'''),
        ("j", "a number with 3 decimals", "posterior_first(math.log(0.45) + LP_FL, math.log(0.55) + LA_FL)",
         r'''decimals=3, mistakes={"tu as le posterior de l'autre espèce": 0.319,
          "n'oublie pas les priors 0,55 et 0,45": 0.723}'''),
        ("k", "True or False", "False",
         r'''mistakes={"que suppose le produit des deux densités ? Relis l'encadré sur l'indépendance conditionnelle (§13.6)": True}'''),
    ]),
    Paper("13.9", "Mémoire et temps d'un kNN sur MNIST complet", [
        ("a", "a whole number", "N_TRAIN * P_PIX",
         r'''mistakes={"le kNN garde l'entraînement, pas le test": 7_840_000,
          "tu as compté l'entraînement et le test ; le kNN ne garde que l'entraînement": 54_880_000}'''),
        ("b", "a number with 2 decimals (MB)", "N_TRAIN * P_PIX * 8 / 1e6",
         r'''decimals=2, mistakes={"l'énoncé définit le Mo comme 10⁶ octets, pas 2²⁰": 358.89,
          "combien d'octets occupe un float64 ?": 188.16, "combien d'octets occupe un float64 ? (pas un seul)": 47.04}'''),
        ("c", "a number with 2 decimals (MB)", "N_TRAIN * P_PIX / 1e6",
         r'''decimals=2, mistakes={"l'énoncé définit le Mo comme 10⁶ octets, pas 2²⁰": 44.86, "combien d'octets occupe un uint8 ?": 376.32}'''),
        ("d", "a whole number", "3 * N_TRAIN * P_PIX",
         r'''mistakes={"combien d'opérations par valeur ?": 47_040_000}'''),
        ("e", "a number with 2 decimals (seconds)", "3 * N_TRAIN * P_PIX * N_TEST / 1e10",
         r'''decimals=2, mistakes={"c'est le temps d'une seule requête ; il y en a 10 000": 0.01,
          "combien d'opérations par valeur ?": 47.04}'''),
        ("f", "a number (GB)", "N_TEST * N_TRAIN * 8 / 1e9",
         r'''decimals=2, mistakes={"combien d'octets occupe un float64 ?": 0.6, "en quelle unité la question demande-t-elle la taille ?": 4800}'''),
        ("g", "a number (MB)", "500 * N_TRAIN * 8 / 1e6",
         r'''decimals=2, mistakes={"combien d'octets occupe un float64 ?": 30, "un batch compte combien d'images de test ?": 4800}'''),
        ("h", "a whole number", "10 * (P_PIX + 1)",
         r'''mistakes={"n'oublie pas les 10 biais": 7840, "un jeu de poids par classe : 10 classes": 785}'''),
        ("i", "a whole number (rounded)", "round(N_TRAIN * P_PIX * 8 / (10 * (P_PIX + 1) * 8))",
         r'''mistakes={"n'oublie pas les 10 biais dans la régression logistique": 6000,
          "compare des octets à des octets : 8 octets par paramètre aussi": 47939}'''),
        ("j", "a number (MB)", "N_TRAIN * 50 * 8 / 1e6",
         r'''decimals=2, mistakes={"combien d'octets occupe un float64 ?": 3, "combien de valeurs garde chaque image après la PCA ?": 376.32}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 13.1 à 13.7 et 🧮 13.9)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "une suite de codes : les lettres dans l'ordre (`\"XYZ\"`) ; plusieurs nombres : une liste "
               "(`[4, 7]`). Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas "
               "encore remplies affichent ⏳. Les questions « dans ta copie », la preuve ∂ 13.8, la réflexion "
               "(🗣️ 13.10, ⚖️ 13.11) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to F (exercises 13.12 to 13.36): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "13.12 à 13.15", "prédire les frontières de cinq classifieurs, choisir k, KNeighborsClassifier dans "
                           "mylearn.neighbors, kNN et malédiction de la dimension"),
    ("B", "13.16 à 13.18", "SVM linéaire et paramètre C, relever des cercles en 3D, noyau RBF réglé par "
                           "validation croisée"),
    ("C", "13.19 à 13.26", "instabilité d'un arbre, impuretés, meilleur split et arbres dans mylearn.tree, "
                           "une colonne qui triche, plot_tree et élagage, arbre de régression"),
    ("D", "13.27 à 13.30", "Naïve Bayes sur des blobs puis des lunes, GaussianNB et MultinomialNB dans "
                           "mylearn.naive_bayes, phrases de Holmes en sac de mots et TF-IDF"),
    ("E", "13.31 et 13.32", "régression logistique de scikit-learn, puis la tienne dans mylearn.logistic"),
    ("F", "13.33 à 13.36", "associer des frontières, accélérer son kNN, comparatif mylearn contre "
                           "scikit-learn, défi MNIST"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 13.Q1 à 13.Q12, 13.R1 à 13.R3, 13.1 à 13.7, 13.9 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 | "
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
        title = "# 13 · Classifieurs — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch13_classifieurs/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 13`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 13 · Classifieurs — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Calculer à la main un vote de k plus proches voisins, une marge, une impureté, un gain de split et "
               "un posterior de Naïve Bayes.\n"
               "- Écrire et vérifier contre scikit-learn les classifieurs de ta librairie : `KNeighborsClassifier`, "
               "les arbres de `mylearn.tree` (prêts pour les ensembles du ch. 14), `GaussianNB`, `MultinomialNB` "
               "et `LogisticRegression`.\n"
               "- Régler $k$, C, $\\gamma$ et la profondeur d'un arbre par validation croisée, et limiter "
               "l'overfitting d'un arbre.\n"
               "- Comparer les classifieurs (accuracy, temps, mémoire, explicabilité) et choisir une baseline.\n\n"
               "**Rappel express.** Distances : $\\sqrt{\\sum_j (x_j - z_j)^2}$ et $\\sum_j \\lvert x_j - z_j "
               "\\rvert$ ; distance à un hyperplan : $\\lvert \\mathbf{w} \\cdot \\mathbf{x} + b \\rvert / \\lVert "
               "\\mathbf{w} \\rVert$, largeur de la bande $2/\\lVert \\mathbf{w} \\rVert$ ; noyau RBF : "
               "$e^{-\\gamma \\lVert \\mathbf{x} - \\mathbf{z} \\rVert^2}$ ; Gini : $1 - \\sum_c p_c^2$ ; entropie : "
               "$-\\sum_c p_c \\log_2 p_c$ ; gain : $I(\\text{parent}) - \\frac{n_G}{n} I(G) - \\frac{n_D}{n} I(D)$ "
               "; Naïve Bayes : $\\log P(c) + \\sum_j \\log P(x_j \\mid c)$, normalisé par logsumexp ; lissage : "
               "$\\frac{N_{cj} + \\alpha}{N_c + \\alpha d}$ ; régression logistique : $\\hat p = \\sigma(\\mathbf{w} "
               "\\cdot \\mathbf{x} + b)$, gradient $\\frac1n \\mathbf{X}^\\top (\\hat{\\mathbf{p}} - \\mathbf{y})$. En "
               "Python : `sklearn.neighbors.KNeighborsClassifier`, `sklearn.svm.SVC`, `sklearn.tree` "
               "(`DecisionTreeClassifier`, `plot_tree`, `export_text`), `sklearn.naive_bayes` (`GaussianNB`, "
               "`MultinomialNB`), `sklearn.linear_model.LogisticRegression`, `sklearn.model_selection.GridSearchCV`.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 13.12 à 13.36 (parties A à F, ta librairie `neighbors`, `tree`, "
             "`naive_bayes` et `logistic`) seront ajoutés à la prochaine session de génération ; relance alors "
             "`python tools/start_chapter.py 13` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dire, pour chaque classifieur du chapitre, ce qu'il garde en mémoire, où il passe son "
               "temps et ce qu'il suppose des données ?\n"
               "2. Sais-tu calculer à la main un vote kNN, une marge, un gain de split et un posterior de Naïve "
               "Bayes ?\n"
               "3. Sais-tu choisir une baseline et régler ses hyperparamètres par validation croisée ?\n\n"
               "**Pour aller plus loin** : les guides utilisateur de scikit-learn sur les plus proches voisins, les "
               "SVM, les arbres et Naïve Bayes, et l'article de Domingos et Pazzani, cités dans la fiche. La suite : "
               "le ch. 14 (ensembles), qui construira forêts aléatoires, AdaBoost et gradient boosting avec tes "
               "arbres de `mylearn.tree`." + later)]


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
