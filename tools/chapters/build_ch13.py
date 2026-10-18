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

from chapter_kit import STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

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
    Paper("13.R1", "Ch. 12 : pourquoi standardiser avant de mesurer des distances", [
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
    Paper("13.R2", "Ch. 10 : l'équation de la frontière d'un perceptron", [
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
    Paper("13.R3", "Ch. 6 : entropie de deux distributions", [
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
# Part A: k nearest neighbours (13.12 to 13.15)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools for the notebook exercises (parts A to F), and the penguins
import contextlib
import io
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


TEST_FILES = {module: f"tests/test_ch13_{module}.py" for module in ["neighbors", "tree", "naive_bayes", "logistic"]}


def run_mylearn_tests(module, keyword, impl="learner"):
    """Run the tests of mylearn.<module> selected by `keyword` (on YOUR code by default)."""
    test_file = TEST_FILES[module]
    command = [sys.executable, "-m", "pytest", test_file, "-k", keyword, "-q", "-p", "no:cacheprovider",
               "--color=no", "-rf", "--tb=no"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "1000"})   # long lines: the reason of each failure
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

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="13")   # reload your saved file\n'


def mylearn_howto(file: str) -> str:
    return ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre "
            f"`mon_travail/mylearn/{file}` (créé par `python tools/start_chapter.py 13`), lis la docstring de chaque "
            "fonction et de chaque méthode, remplace les `raise NotImplementedError(...)` par ton code et "
            "**enregistre**. NumPy seulement : scikit-learn et SciPy sont les **oracles** des tests. La cellule de "
            "vérification recharge ta librairie, compare quelques résultats à scikit-learn, puis lance les tests de "
            f"l'exercice ; `python -m pytest tests/test_ch13_{file.removesuffix('.py')}.py -q` les lance tous dans un "
            "terminal.")


def ref_check(check: str, ex_id: str) -> str:
    """The solutions version of a 🔨 check cell: no wb.attempt, and the tests run on the reference."""
    return (check.replace(f'with wb.attempt("{ex_id}"):\n', "if True:\n")
            .replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", module="'))


def as_printed(check: str, ex_id: str) -> str:
    """The solutions version of a check cell that reads the learner's values: print them instead of checking."""
    return check.replace(f'with wb.attempt("{ex_id}"):\n', "if True:\n").replace("wb.check(", "print_answer(")


GIVEN_12 = r'''X_12, Y_12 = wb.synth.make_xor(n=600, noise=0.25, seed=1312)   # 4 noisy corners, class = x1 XOR x2
X_TRAIN_12, X_TEST_12, Y_TRAIN_12, Y_TEST_12 = train_test_split(X_12, Y_12, test_size=0.3, random_state=1312,
                                                                stratify=Y_12)
print(len(X_TRAIN_12), "training points,", len(X_TEST_12), "test points")'''

EXPERIMENT_12 = r'''models_12 = {"A": KNeighborsClassifier(n_neighbors=15), "B": SVC(), "C": DecisionTreeClassifier(random_state=0),
             "D": GaussianNB(), "E": LogisticRegression()}
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (letter, model) in zip(axes, models_12.items()):
    model.fit(X_TRAIN_12, Y_TRAIN_12)
    leaves = f" · {model.get_n_leaves()} leaves" if letter == "C" else ""
    print(f"{letter} {type(model).__name__:24s} training {model.score(X_TRAIN_12, Y_TRAIN_12):.3f} · "
          f"test {model.score(X_TEST_12, Y_TEST_12):.3f}{leaves}")
    wb.plot.plot_decision_boundary(model, X_TEST_12, Y_TEST_12, ax=ax, title=f"{letter} · {type(model).__name__}")
plt.show()
wb.check("13.12a", chance_12)
wb.check("13.12b", leaves_12)
wb.check("13.12c", tree_beats_knn_12)'''

GIVEN_13 = r'''X_MOONS_13, Y_MOONS_13 = wb.synth.make_moons(n=600, noise=0.25, seed=1313)
ISLAND_13 = np.array([-0.2, 1.0])                  # the centre of a small region of class 1 in the upper moon (class 0)
island_points_13 = np.random.default_rng(1313).normal(ISLAND_13, 0.08, (15, 2))
X_13 = np.vstack([X_MOONS_13, island_points_13])   # 615 points
Y_13 = np.concatenate([Y_MOONS_13, np.ones(15, dtype=int)])
KS_13 = [1, 3, 5, 9, 15, 25, 41, 61, 101, 151]
CV_13 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1313)'''

TODO_13 = r'''train_k1_13 = ...   # a) accuracy of KNeighborsClassifier(n_neighbors=1) fitted on X_13, Y_13, measured on X_13, Y_13
cv_13 = ...         # b) list: for each k of KS_13, the mean accuracy of a kNN in 5-fold cross-validation (CV_13)
best_k_13 = ...     # c) the k of KS_13 with the best mean accuracy in b)
erased_13 = ...     # d) the smallest k of KS_13 whose kNN fitted on X_13, Y_13 predicts class 0 at ISLAND_13'''

SOLUTION_13 = r'''train_k1_13 = KNeighborsClassifier(n_neighbors=1).fit(X_13, Y_13).score(X_13, Y_13)
cv_13 = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_13, Y_13, cv=CV_13).mean() for k in KS_13]
best_k_13 = KS_13[int(np.argmax(cv_13))]
erased_13 = next(k for k in KS_13 if KNeighborsClassifier(n_neighbors=k).fit(X_13, Y_13).predict([ISLAND_13])[0] == 0)
'''

CHECK_13 = r'''wb.check("13.13a", train_k1_13)
wb.check("13.13b", cv_13)
wb.check("13.13c", best_k_13)
wb.check("13.13d", erased_13)
with wb.attempt("13.13"):
    if not filled(cv_13, erased_13):
        raise NotImplementedError("cv_13, erased_13")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, k in zip(axes, [1, 15, 101]):
        model = KNeighborsClassifier(n_neighbors=k).fit(X_13, Y_13)
        wb.plot.plot_decision_boundary(model, X_13, Y_13, ax=ax, title=f"k = {k}")
        ax.add_patch(plt.Circle(ISLAND_13, 0.25, fill=False, color="black", ls="--"))
    plt.show()'''

CHECK_14 = r'''with checked("13.14"):
    D_doc_14 = mylearn.neighbors.pairwise_distances(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[3.0, 4.0]]))
    if returned("13.14", "pairwise_distances", D_doc_14):
        print("docstring example:", np.round(np.asarray(D_doc_14, dtype=float), 4).ravel().tolist(), "(expected [5.0, 3.6056])")
with checked("13.14"):
    scaler_14 = StandardScaler().fit(P_TRAIN[MEASURES])
    Z_train_14, Z_test_14 = scaler_14.transform(P_TRAIN[MEASURES]), scaler_14.transform(P_TEST[MEASURES])
    y_train_14, y_test_14 = P_TRAIN["species"].to_numpy(), P_TEST["species"].to_numpy()
    for k, weights, metric in [(5, "uniform", "euclidean"), (6, "distance", "manhattan")]:
        setting_14 = f"k = {k}, {weights}, {metric}"
        mine_14 = fitted(mylearn.neighbors.KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric),
                         Z_train_14, y_train_14)
        oracle_14 = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, algorithm="brute")
        oracle_14.fit(Z_train_14, y_train_14)
        proba_14 = mine_14.predict_proba(Z_test_14)
        if not returned("13.14", "predict_proba", proba_14):
            continue
        predicted_14 = mine_14.predict(Z_test_14)
        if not returned("13.14", "predict", predicted_14):
            continue
        proba_14, predicted_14 = np.asarray(proba_14), np.asarray(predicted_14)
        if proba_14.shape != (len(Z_test_14), 3):
            print(f"❌ Ex 13.14 : {setting_14} : predict_proba devrait rendre un tableau (100, 3), une colonne par "
                  f"espèce de classes_, pas un tableau de forme {proba_14.shape}.")
        elif not np.allclose(proba_14, oracle_14.predict_proba(Z_test_14)):
            print(f"❌ Ex 13.14 : {setting_14} : tes probabilités diffèrent de celles de scikit-learn ; les tests "
                  "ci-dessous disent où.")
        elif not np.array_equal(predicted_14, oracle_14.predict(Z_test_14)):
            print(f"❌ Ex 13.14 : {setting_14} : tes probabilités sont justes, mais predict rend autre chose que les "
                  f"labels de scikit-learn (par exemple {predicted_14[:3].tolist()} au lieu de "
                  f"{oracle_14.predict(Z_test_14)[:3].tolist()}).")
        else:
            print(f"✅ Ex 13.14 : {setting_14} : mêmes probabilités et mêmes prédictions que scikit-learn sur les 100 "
                  f"manchots de test (accuracy {fr(oracle_14.score(Z_test_14, y_test_14))}).")
if written(lambda: mylearn.neighbors.pairwise_distances([[0.0]], [[1.0]])):
    run_mylearn_tests("neighbors", keyword="test_pairwise_distances_ or test_kneighbors_classifier_")
else:
    print("⏳ Ex 13.14 : les tests attendent ton pairwise_distances.")'''

GIVEN_15 = r'''SCALER_15 = StandardScaler().fit(P_TRAIN[MEASURES])
Z_TRAIN_15, Z_TEST_15 = SCALER_15.transform(P_TRAIN[MEASURES]), SCALER_15.transform(P_TEST[MEASURES])
Y_TRAIN_15, Y_TEST_15 = P_TRAIN["species"].to_numpy(), P_TEST["species"].to_numpy()
NOISE_15 = [0, 2, 5, 10, 20, 50, 100, 200]   # numbers of noise columns
SEEDS_15 = range(10)                          # 10 draws of the noise for each number
P_15 = [2, 10, 100, 1000]                     # dimensions of the uniform cube'''

TODO_15 = r'''def noisy_accuracy_15(n_noise, seed):
    """Test accuracy of YOUR mylearn.neighbors.KNeighborsClassifier(n_neighbors=5), fitted on the 4 standardized
    measures Z_TRAIN_15 plus n_noise columns of noise, then scored on Z_TEST_15 plus n_noise columns of noise. The noise
    is standard normal, drawn with rng = np.random.default_rng(seed) in two calls: rng.normal(size=(len(Z_TRAIN_15),
    n_noise)) for the training part, then rng.normal(size=(len(Z_TEST_15), n_noise)) for the test part."""
    raise NotImplementedError("noisy_accuracy_15")


def distance_ratio_15(p, seed=1315):
    """Draw 500 points uniformly in [0, 1]^p (np.random.default_rng(seed).uniform(size=(500, p))). For each of the
    first 50, the distance to its nearest OTHER point divided by the distance to its farthest point (YOUR
    mylearn.neighbors.pairwise_distances); return the mean of these 50 ratios."""
    raise NotImplementedError("distance_ratio_15")


accuracies_15 = ...   # a) list: for each n_noise of NOISE_15, the mean of noisy_accuracy_15(n_noise, seed) over SEEDS_15
first_below_15 = ...  # b) the smallest n_noise of NOISE_15 whose mean accuracy in a) is below 0.9
ratios_15 = ...       # c) list: distance_ratio_15(p) for each p of P_15'''

SOLUTION_15 = r'''def noisy_accuracy_15(n_noise, seed):
    """Test accuracy of YOUR mylearn.neighbors.KNeighborsClassifier(n_neighbors=5), fitted on the 4 standardized
    measures Z_TRAIN_15 plus n_noise columns of noise, then scored on Z_TEST_15 plus n_noise columns of noise. The noise
    is standard normal, drawn with rng = np.random.default_rng(seed) in two calls: rng.normal(size=(len(Z_TRAIN_15),
    n_noise)) for the training part, then rng.normal(size=(len(Z_TEST_15), n_noise)) for the test part."""
    rng = np.random.default_rng(seed)
    train = np.hstack([Z_TRAIN_15, rng.normal(size=(len(Z_TRAIN_15), n_noise))])
    test = np.hstack([Z_TEST_15, rng.normal(size=(len(Z_TEST_15), n_noise))])
    model = mylearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    model.fit(train, Y_TRAIN_15)
    return model.score(test, Y_TEST_15)


def distance_ratio_15(p, seed=1315):
    """Draw 500 points uniformly in [0, 1]^p (np.random.default_rng(seed).uniform(size=(500, p))). For each of the
    first 50, the distance to its nearest OTHER point divided by the distance to its farthest point (YOUR
    mylearn.neighbors.pairwise_distances); return the mean of these 50 ratios."""
    points = np.random.default_rng(seed).uniform(size=(500, p))
    D = mylearn.neighbors.pairwise_distances(points[:50], points)
    D[np.arange(50), np.arange(50)] = np.inf          # a point is not its own neighbour
    nearest = D.min(axis=1)
    D[np.arange(50), np.arange(50)] = 0.0
    return float(np.mean(nearest / D.max(axis=1)))


accuracies_15 = [float(np.mean([noisy_accuracy_15(n, seed) for seed in SEEDS_15])) for n in NOISE_15]
first_below_15 = next(n for n, accuracy in zip(NOISE_15, accuracies_15) if accuracy < 0.9)
ratios_15 = [distance_ratio_15(p) for p in P_15]
'''

CHECK_15 = r'''wb.check("13.15a", accuracies_15)
wb.check("13.15b", first_below_15)
wb.check("13.15c", ratios_15)
with wb.attempt("13.15"):
    if not filled(accuracies_15, ratios_15):
        raise NotImplementedError("accuracies_15, ratios_15")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(NOISE_15, accuracies_15, marker="o")
    axes[0].set_xscale("symlog", linthresh=2)
    axes[0].set_xlabel("noise columns added to the 4 measures")
    axes[0].set_ylabel("mean test accuracy (k = 5)")
    axes[1].plot(P_15, ratios_15, marker="o")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("dimension p of the cube")
    axes[1].set_ylabel("nearest / farthest distance")
    plt.show()'''

PART_A = Part("A", "Les k plus proches voisins",
              "Cinq classifieurs sur un « ou exclusif » bruité, pour fixer les idées (13.12) ; puis le kNN : le choix "
              "de $k$ (13.13), ta classe `KNeighborsClassifier` (13.14) et ce que deviennent les distances quand la "
              "dimension augmente (13.15). La cellule suivante importe les classifieurs de scikit-learn, définit les "
              "outils des parties A à F et découpe les 333 manchots complets (233 pour l'entraînement, 100 pour le "
              "test), dont la partie 0 n'a utilisé que quelques valeurs. Ne recopie jamais un nombre à la main : chaque "
              "réponse est calculée par ton code, et pas arrondie.",
              given=PART_A_GIVEN, exercises=[
    Ex("13.12", "🔮", 1, 15, "Prédire la frontière de cinq classifieurs avant de l'afficher",
       "prévoir, avant de les voir, les scores de cinq classifieurs sur un problème qu'une droite ne sépare pas.",
       "ch. 7 (classification) · fiche §13.1, §13.2 et §13.7 (tableau « frontière »)", thread="synthétique",
       tracks="R, C", hypothesis=True,
       body=r"""`X_12` : 600 points bruités autour des quatre coins du carré unité ; la classe vaut 1 quand une seule des deux coordonnées est proche de 1 (un « ou exclusif »), 0 sinon. On entraîne sur 70 % des points, avec leurs réglages par défaut, cinq classifieurs : A `KNeighborsClassifier(n_neighbors=15)` ; B `SVC()` (noyau RBF) ; C `DecisionTreeClassifier()`, sans limite ; D `GaussianNB()` ; E `LogisticRegression()`. La fiche (tableau du §13.7) te dit déjà la **forme** de leurs frontières ; prédis plutôt des nombres.

Prédis, **avant** d'exécuter quoi que ce soit :
a) `chance_12` : les lettres des classifieurs dont l'accuracy de test reste **sous 65 %**, dans n'importe quel ordre (`""` si aucun) ;
b) `leaves_12` : le nombre de feuilles de l'arbre C : `"A"` moins de 10 ; `"B"` de 10 à 50 ; `"C"` de 50 à 150 ; `"D"` plus de 150 (une feuille par point d'entraînement en ferait 420) ;
c) `tree_beats_knn_12` : l'arbre C a-t-il une meilleure accuracy de test que le kNN A ? (`True` ou `False`)

Puis exécute l'expérience : elle affiche les accuracies d'entraînement et de test, et les cinq frontières sur les points de test.""",
       given=GIVEN_12,
       todo=r'''chance_12 = ...          # a) the letters, e.g. "AB", or "" if none
leaves_12 = ...          # b) "A", "B", "C" or "D"
tree_beats_knn_12 = ...  # c) True or False''',
       solution=r'''chance_12, leaves_12, tree_beats_knn_12 = "DE", "B", False   # the answers, for the record''',
       after=[("code", guarded(EXPERIMENT_12, ["chance_12", "leaves_12", "tree_beats_knn_12"],
                               "⏳ Ex 13.12 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : compare avec tes prédictions. Pour chaque classifieur dont l'accuracy t'a "
                     "surpris, cherche ce qu'il sait représenter et ce qu'il ne sait pas. Où l'arbre place-t-il ses "
                     "premières questions (`export_text`) ? Explique son nombre de feuilles.")],
       record=r'''wb.record("13.12a", "DE", ordered=False,
          mistakes={"une droite peut-elle mettre (0, 0) et (1, 1) d'un côté, (0, 1) et (1, 0) de l'autre ? et un modèle qui décrit chaque classe par une moyenne et une variance par coordonnée ?": "",
                    "relis le tableau du §13.7 : quelle frontière a la régression logistique, et peut-elle séparer les quatre coins ?": "D",
                    "Naïve Bayes gaussien décrit chaque classe par une moyenne et une variance par coordonnée : que valent-elles pour les deux classes d'un ou exclusif ?": "E",
                    "un arbre sans limite a-t-il besoin que les classes soient séparables par une droite ?": "CDE"})
wb.record("13.12b", "B", mistakes={"quatre coins demandent au moins quatre rectangles, mais le bruit mêle les classes près du centre : l'arbre s'arrête-t-il à quatre ?": "A",
                                    "le bruit ne touche que les points proches des frontières x = 0,5 et y = 0,5 : combien de petits rectangles demandent-ils ?": "C",
                                    "un arbre sans limite isole les points mal placés, pas chaque point : combien de points d'entraînement sont du mauvais côté du centre ?": "D"})
wb.record("13.12c", False, mistakes={"un arbre sans limite apprend aussi le bruit de l'entraînement : est-ce un avantage sur le test, face à un vote de 15 voisins ?": True})''',
       note="Naïve Bayes gaussien et la régression logistique restent près du hasard (environ 0,51 et 0,53 en test) : "
            "une droite ne sépare pas les quatre coins, et, pour Naïve Bayes, les deux classes ont la **même** moyenne "
            "(0,5) et à peu près la même variance sur chaque coordonnée : prises une par une, les coordonnées ne disent "
            "rien de la classe ; seule leur **combinaison** compte, et l'hypothèse naïve l'ignore. Le kNN et le SVM "
            "RBF atteignent 0,96 ; l'arbre sans limite, parfait sur l'entraînement, fait 0,92 en test avec 29 "
            "feuilles. Sur un « ou exclusif », aucune question seule autour de 0,5 ne fait baisser l'impureté : "
            "l'arbre glouton commence par une coupe presque sans gain au bord du nuage ($x_1 \\le -0{,}14$, gain "
            "0,004), puis coupe $x_2$ vers 0,6 ; deux questions autour de 0,5 suffiraient, mais la recherche "
            "gloutonne ne les trouve pas, et les points bruités proches du centre ajoutent leurs petits rectangles. "
            "Avec d'autres graines (1, 2 et 3, pour les données et pour le découpage), les mêmes conclusions "
            "tiennent : Naïve Bayes et la régression logistique de 0,46 à 0,63, l'arbre de 24 à 28 feuilles, et "
            "toujours derrière le kNN."),

    Ex("13.13", "🔬", 2, 30, "Choisir k : de la frontière en dents de scie à l'érosion",
       "choisir $k$ par validation croisée, et voir ce qu'un grand $k$ efface.",
       "Ex 13.12 · ch. 8 (validation croisée) · fiche §13.3", thread="synthétique", tracks="R, C",
       body=r"""`X_13` : 600 points de deux demi-lunes bruitées, plus une petite **île** de 15 points de la classe 1 autour de `ISLAND_13` = (−0,2 ; 1,0), au milieu de la lune du haut (classe 0). Avec le `KNeighborsClassifier` de scikit-learn (ta classe arrive en 13.14), calcule :
a) `train_k1_13` : l'accuracy d'un kNN à $k = 1$ entraîné sur `X_13` et mesurée sur ces **mêmes** points ;
b) `cv_13` : la liste des accuracies moyennes en validation croisée à 5 folds (`CV_13`), une par $k$ de `KS_13` ;
c) `best_k_13` : le $k$ de `KS_13` qui a la meilleure accuracy en b) ;
d) `erased_13` : le plus petit $k$ de `KS_13` pour lequel un kNN entraîné sur tous les points prédit la classe 0 en `ISLAND_13` (l'île a disparu).

La vérification contrôle a) à d), puis trace les frontières pour $k$ = 1, 15 et 101, l'île entourée.

Dans tes notes : pourquoi a) ne dit-il rien de la qualité du modèle ? L'île est-elle un vrai phénomène ou du bruit, et la validation croisée peut-elle le dire ? Que gagne-t-on, et que perd-on, en passant de $k = 1$ à $k = 101$ ?""",
       given=GIVEN_13, todo=TODO_13, check=CHECK_13, solution=SOLUTION_13 + as_printed(CHECK_13, "13.13"),
       record=r'''wb.record("13.13a", train_k1_13, decimals=4,
          mistakes={"c'est une accuracy de validation croisée : la question la mesure sur les points mêmes qui ont servi au fit": cv_13[0]})
train_curve_13 = [KNeighborsClassifier(n_neighbors=k).fit(X_13, Y_13).score(X_13, Y_13) for k in KS_13]
wb.record("13.13b", cv_13, decimals=4,
          mistakes={"ce sont les accuracies sur les points d'entraînement : chaque point y est son propre voisin ; utilise cross_val_score avec CV_13": train_curve_13})
others_13c = {f"k = {k} n'a pas la meilleure accuracy moyenne de b) : rends le k de la plus grande valeur de cv_13": k
              for k in KS_13 if k not in (1, best_k_13, int(np.argmax(cv_13)))}
wb.record("13.13c", best_k_13, choices=KS_13,
          fractional="un nombre non entier : est-ce une accuracy ? on demande la valeur de k, un élément de KS_13",
          mistakes={"sur l'entraînement, k = 1 gagne toujours : choisis k sur la validation croisée": 1,
                    "est-ce la position de k dans KS_13 ? rends le k lui-même, celui de la plus grande accuracy moyenne de b)": int(np.argmax(cv_13)),
                    **others_13c})
island_13 = [int(KNeighborsClassifier(n_neighbors=k).fit(X_13, Y_13).predict([ISLAND_13])[0]) for k in KS_13]
others_13d = {f"k = {k} : est-ce le PREMIER k de KS_13 dont la prédiction en ISLAND_13 vaut 0 ? parcours KS_13 dans l'ordre": k
              for k in KS_13 if k not in (KS_13[0], erased_13, KS_13[KS_13.index(erased_13) - 1])}
wb.record("13.13d", erased_13, choices=KS_13,
          fractional="un nombre non entier : on demande la valeur de k, un élément de KS_13",
          mistakes={"c'est le dernier k qui prédit encore la classe 1 sur l'île : on demande le premier qui prédit la classe 0": KS_13[KS_13.index(erased_13) - 1],
                    "relis la question : le premier k qui prédit la classe 0 en ISLAND_13 (la classe de la lune autour)": KS_13[0],
                    **others_13d})''',
       note="À $k = 1$, chaque point d'entraînement est son propre plus proche voisin : l'accuracy sur l'entraînement "
            "vaut 1 par construction et ne mesure rien (annexe des erreurs fréquentes). En validation croisée, "
            "$k = 1$ ne fait que 0,889 : la frontière en dents de scie suit le bruit. L'accuracy monte jusqu'à "
            "$k = 9$ (0,925), reste presque plate jusqu'à 25, puis redescend (0,889 à $k = 151$). L'île tient tant que "
            "ses 15 points dominent le vote : jusqu'à $k = 25$ ; dès 41 voisins, la lune autour l'emporte, et l'île "
            "disparaît. La validation croisée ne dit pas si l'île est un vrai phénomène : 15 points sur 615 pèsent "
            "peu dans l'accuracy moyenne, et les $k$ de 9 à 25 la gardent avec des scores très proches de ceux qui "
            "l'effacent. Seule une connaissance du domaine (ou plus de données dans cette zone) tranche."),

    Ex("13.14", "🔨", 3, 45, "Coder pairwise_distances et KNeighborsClassifier",
       "écrire un kNN en force brute, vectorisé, qui donne exactement les votes de scikit-learn.",
       "✏️ 13.1 · ch. 12 (standardisation) · 0A (classes Python) · fiche §13.3", thread="Penguins",
       tracks="R, C", mylearn="neighbors.py",
       body=mylearn_howto("neighbors.py") + r"""

Écris `pairwise_distances` et la classe `KNeighborsClassifier` de `neighbors.py`. **Contrat** (les docstrings, plus ces précisions que vérifient les tests) :
- `pairwise_distances(A, B, metric="euclidean")` reçoit deux tableaux 2-D (ou des listes) de mêmes nombres de colonnes et rend la matrice `float64` de forme `(len(A), len(B))` des distances `"euclidean"` ou `"manhattan"`, **sans boucle Python sur les exemples** ; la distance d'un point à lui-même vaut 0, à $10^{-9}$ près, même pour des points loin de l'origine (les tests le vérifient) ; elle ne modifie pas ses arguments ; `ValueError` si `A` ou `B` n'est pas 2-D, si leurs nombres de colonnes diffèrent, si la métrique est inconnue ou si une valeur est `NaN` (comme scikit-learn) ;
- `__init__(n_neighbors=5, weights="uniform", metric="euclidean")` range ses trois arguments tels quels, sans rien vérifier ni calculer ;
- `fit(X, y)` vérifie, puis **range des copies** de `X` et `y` dans des attributs dont le nom finit par `_` (ou commence par `_`), avec `classes_` (les labels distincts triés), `n_features_in_` et `n_samples_fit_`, et renvoie `self` ; un second `fit` oublie le premier ; `ValueError` si `n_neighbors < 1` ou `n_neighbors > len(X)` (dès `fit`), si `weights` ou `metric` est inconnu, si `X` n'est pas 2-D ou contient `NaN`, ou si `len(X) != len(y)` ;
- `kneighbors(X, n_neighbors=None, return_distance=True)` rend `(dist, ind)` (ou `ind` seul), de forme `(len(X), k)`, du plus proche au plus lointain ; à distances égales, le plus petit indice d'entraînement d'abord (un tri **stable**) ; `RuntimeError` avant `fit` ; `ValueError` pour une requête qui n'est pas 2-D (une seule requête s'écrit `X[None, :]`), un mauvais nombre de colonnes, une valeur `NaN`, ou un $k$ hors de 1 à `n_samples_fit_` ;
- `predict_proba(X)` rend la part (pondérée) des votes de chaque classe, colonne $j$ pour `classes_[j]` ; avec `weights="distance"`, un voisin à distance $d$ pèse $1/d$, et des voisins à distance 0 se partagent tout le poids ; `predict(X)` rend la classe de plus grande part (à égalité, la première de `classes_`) ; `score(X, y)` rend l'accuracy, un `float` Python.

La vérification essaie l'exemple de la docstring, compare tes probabilités et tes prédictions à celles de scikit-learn sur les manchots standardisés (deux réglages), puis lance les tests.

Dans tes notes : que coûte un appel à `predict` sur 100 manchots, en nombre de distances ? Pourquoi un tri stable, et pourquoi conseille-t-on souvent un $k$ impair à deux classes ?""",
       check=RELOAD + CHECK_14, solution=ref_check(CHECK_14, "13.14"),
       note="Une prédiction calcule les distances de chaque manchot de test aux 233 manchots d'entraînement : 23 300 "
            "distances de 4 coordonnées pour 100 manchots, et rien n'a été calculé pendant `fit`. Le tri stable rend "
            "le résultat indépendant de l'algorithme de tri, et le même que celui de scikit-learn en force brute ; à "
            "deux classes, un $k$ impair évite les votes à égalité (avec des poids uniformes)."),

    Ex("13.15", "🔬", 3, 35, "kNN et malédiction de la dimension",
       "mesurer comment des features inutiles noient les distances d'un kNN, et pourquoi.",
       "Ex 13.14 · ch. 7 (malédiction de la dimension) · ch. 12 (standardisation) · fiche §13.3",
       thread="Penguins", tracks="C",
       body=r"""Deux expériences avec **ta** librairie (`mylearn.neighbors` ; si 13.14 n'est pas fait, la référence prend le relais). Écris :
- `noisy_accuracy_15(n_noise, seed)` : l'accuracy de test de ton `KNeighborsClassifier(n_neighbors=5)` entraîné sur les 4 mesures standardisées des manchots (`Z_TRAIN_15`) **plus** `n_noise` colonnes de bruit, puis évalué sur `Z_TEST_15` plus `n_noise` colonnes de bruit ; le bruit suit une loi normale standard, tiré par `rng = np.random.default_rng(seed)` en deux appels : `rng.normal(size=(len(Z_TRAIN_15), n_noise))` pour l'entraînement, puis `rng.normal(size=(len(Z_TEST_15), n_noise))` pour le test ;
- `distance_ratio_15(p, seed=1315)` : 500 points tirés uniformément dans le cube $[0, 1]^p$ ; pour chacun des 50 premiers, la distance à son plus proche **autre** point divisée par la distance à son point le plus lointain (ton `pairwise_distances`) ; la fonction rend la moyenne de ces 50 rapports ;
- a) `accuracies_15` : pour chaque nombre de colonnes de `NOISE_15`, la moyenne de `noisy_accuracy_15` sur les 10 graines de `SEEDS_15` ;
- b) `first_below_15` : le plus petit nombre de colonnes de `NOISE_15` dont l'accuracy moyenne passe sous 0,9 ;
- c) `ratios_15` : `distance_ratio_15(p)` pour chaque dimension de `P_15`.

La vérification contrôle a) à c), puis trace les deux courbes.

Dans tes notes : les colonnes de bruit ont la même échelle que les mesures standardisées ; pourquoi finissent-elles par décider des voisins ? Que dit c) sur le sens de « plus proche voisin » en grande dimension ? Que ferais-tu sur de vraies données qui ont beaucoup de features ?""",
       given=GIVEN_15, todo=TODO_15, check=CHECK_15, solution=SOLUTION_15 + as_printed(CHECK_15, "13.15"),
       record=r'''def transposed_accuracy_15(n_noise, seed):        # the noise drawn as (n_noise, rows), then transposed
    rng = np.random.default_rng(seed)
    train = np.hstack([Z_TRAIN_15, rng.normal(size=(n_noise, len(Z_TRAIN_15))).T])
    test = np.hstack([Z_TEST_15, rng.normal(size=(n_noise, len(Z_TEST_15))).T])
    return KNeighborsClassifier(n_neighbors=5).fit(train, Y_TRAIN_15).score(test, Y_TEST_15)


wb.record("13.15a", accuracies_15, decimals=4,
          mistakes={"le bruit n'est pas tiré dans la forme de l'énoncé : un appel rng.normal(size=(lignes, n_noise)) pour l'entraînement, puis un pour le test": [float(np.mean([transposed_accuracy_15(n, seed) for seed in SEEDS_15])) for n in NOISE_15]})
wb.record("13.15b", first_below_15, choices=NOISE_15,
          fractional="un nombre non entier : est-ce une accuracy ? on demande un nombre de colonnes de NOISE_15",
          mistakes={"c'est le dernier nombre de colonnes dont l'accuracy reste au-dessus de 0,9 : on demande le premier en dessous": NOISE_15[NOISE_15.index(first_below_15) - 1]})
wb.record("13.15c", ratios_15, decimals=4,
          mistakes={"des rapports nuls : chaque point est à distance 0 de lui-même ; exclus-le avant de chercher le plus proche": [0.0, 0.0, 0.0, 0.0]})''',
       note="Sans bruit, le kNN classe 98 % des manchots de test. Chaque colonne de bruit ajoute à la distance au carré "
            "un terme de même ordre que celui d'une mesure (environ 2 en moyenne entre deux points), mais sans aucun "
            "rapport avec l'espèce : à 20 colonnes, le bruit pèse cinq fois plus que les mesures, et l'accuracy tombe "
            "sous 0,9 ; à 200 colonnes, elle n'est plus qu'à environ 0,6. Dans le cube, le plus proche voisin est 50 "
            "fois plus près que le plus lointain en dimension 2, mais seulement 1,1 fois en dimension 1 000 : toutes "
            "les distances se ressemblent, et « le plus proche » ne veut presque plus rien dire. Sur de vraies "
            "données, on retire les features inutiles (sélection, ch. 12), on réduit la dimension (PCA, 🏆 13.36), "
            "ou l'on choisit un modèle qui pondère ses features (régression logistique, arbres)."),
])

# ---------------------------------------------------------------------------
# Part B: support vector machines (13.16 to 13.18)
# ---------------------------------------------------------------------------
PART_B_GIVEN = r'''def draw_band(ax, model, X, y, title):
    """The points, the boundary f(x) = 0 and the two edges f(x) = ±1 of a 2D SVC, its support vectors circled."""
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
    f = model.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, f > 0, alpha=0.15, cmap="coolwarm")
    ax.contour(xx, yy, f, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=1)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=14, edgecolors="white", linewidths=0.4)
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=60, facecolors="none", edgecolors="black", linewidths=0.8)
    ax.set_title(title)'''

GIVEN_16 = r'''X_16, Y_16 = wb.synth.make_blobs(n=200, centers=[[0.0, 0.0], [2.5, 2.0]], std=0.9, seed=1316)   # two classes, 0 and 1
CS_16 = [0.01, 1, 100]'''

TODO_16 = r'''n_support_16 = ...   # a) list: the total number of support vectors of SVC(kernel="linear", C=C) fitted on X_16, Y_16, for each C of CS_16
width_16 = ...       # b) for C = 1: the width of the band between the two edges f(x) = -1 and f(x) = +1
inside_16 = ...      # c) for C = 1: the number of points with y_i * f(x_i) < 0.999 (y_i = -1 for class 0, +1 for class 1)'''

SOLUTION_16 = r'''models_16 = {C: SVC(kernel="linear", C=C).fit(X_16, Y_16) for C in CS_16}
n_support_16 = [int(models_16[C].n_support_.sum()) for C in CS_16]
w_16, b_16 = models_16[1].coef_[0], models_16[1].intercept_[0]
width_16 = 2 / np.linalg.norm(w_16)
signs_16 = np.where(Y_16 == 1, 1, -1)
inside_16 = int(np.sum(signs_16 * models_16[1].decision_function(X_16) < 0.999))
# The support vectors alone give the same hyperplane:
sv_16 = models_16[1].support_
again_16 = SVC(kernel="linear", C=1).fit(X_16[sv_16], Y_16[sv_16])
print("w, b on the 200 points:", np.round(w_16, 4), round(b_16, 4), "· on the", len(sv_16), "support vectors only:",
      np.round(again_16.coef_[0], 4), round(again_16.intercept_[0], 4))
'''

CHECK_16 = r'''wb.check("13.16a", n_support_16)
wb.check("13.16b", width_16)
wb.check("13.16c", inside_16)
with wb.attempt("13.16"):
    if not filled(n_support_16, width_16, inside_16):
        raise NotImplementedError("n_support_16, width_16, inside_16")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, C in zip(axes, CS_16):
        model = SVC(kernel="linear", C=C).fit(X_16, Y_16)
        draw_band(ax, model, X_16, Y_16, f"C = {C}")
    plt.show()'''

GIVEN_17 = r'''X_17, Y_17 = wb.synth.make_circles(n=400, noise=0.1, factor=0.4, seed=1317)   # a ring (class 0) around a disc (class 1)
X_TRAIN_17, X_TEST_17, Y_TRAIN_17, Y_TEST_17 = train_test_split(X_17, Y_17, test_size=0.3, random_state=1317,
                                                                stratify=Y_17)'''

TODO_17 = r'''def lift_17(X):
    """(n, 2) -> (n, 3): the two coordinates, then x1**2 + x2**2."""
    raise NotImplementedError("lift_17")


def draw_17(ax, X3, y, model):
    """On the 3D axis `ax`: the lifted points X3 (n, 3) coloured by class y, and the plane w . z + b = 0 of `model`, a
    linear SVC fitted on lifted points (solve the plane for its third coordinate on a grid of (x1, x2))."""
    raise NotImplementedError("draw_17")


acc_2d_17 = ...   # a) test accuracy of SVC(kernel="linear", C=1) fitted on the 2 coordinates of X_TRAIN_17
acc_3d_17 = ...   # b) the same, fitted and scored on the lifted coordinates (lift_17)
radius_17 = ...   # c) the radius of the circle where the boundary of b) cuts the surface x3 = x1**2 + x2**2'''

SOLUTION_17 = r'''def lift_17(X):
    """(n, 2) -> (n, 3): the two coordinates, then x1**2 + x2**2."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([X, (X ** 2).sum(axis=1)])


def draw_17(ax, X3, y, model):
    """On the 3D axis `ax`: the lifted points X3 (n, 3) coloured by class y, and the plane w . z + b = 0 of `model`, a
    linear SVC fitted on lifted points (solve the plane for its third coordinate on a grid of (x1, x2))."""
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=y, cmap="coolwarm", s=10)
    w, b = model.coef_[0], model.intercept_[0]
    g1, g2 = np.meshgrid(np.linspace(-1.3, 1.3, 20), np.linspace(-1.3, 1.3, 20))
    ax.plot_surface(g1, g2, -(w[0] * g1 + w[1] * g2 + b) / w[2], alpha=0.3, color="gray")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_1^2 + x_2^2$")


acc_2d_17 = SVC(kernel="linear", C=1).fit(X_TRAIN_17, Y_TRAIN_17).score(X_TEST_17, Y_TEST_17)
model_3d_17 = SVC(kernel="linear", C=1).fit(lift_17(X_TRAIN_17), Y_TRAIN_17)
acc_3d_17 = model_3d_17.score(lift_17(X_TEST_17), Y_TEST_17)
w_17, b_17 = model_3d_17.coef_[0], model_3d_17.intercept_[0]
# w1 x1 + w2 x2 + w3 (x1² + x2²) + b = 0 is a circle of centre (-w1 / 2w3, -w2 / 2w3):
radius_17 = float(np.sqrt((w_17[0] ** 2 + w_17[1] ** 2) / (4 * w_17[2] ** 2) - b_17 / w_17[2]))
print("w =", np.round(w_17, 4), "· b =", round(b_17, 4), "· the simple formula sqrt(-b / w3):", round(float(np.sqrt(-b_17 / w_17[2])), 4))
'''

CHECK_17 = r'''wb.check("13.17a", acc_2d_17)
wb.check("13.17b", acc_3d_17)
wb.check("13.17c", radius_17)
with wb.attempt("13.17"):
    probe_17 = lift_17(np.array([[1.0, 2.0], [0.0, -3.0]]))
    if returned("13.17", "lift_17", probe_17):
        verdict("13.17", np.shape(probe_17) == (2, 3) and np.allclose(probe_17, [[1, 2, 5], [0, -3, 9]]),
                "lift_17 ajoute bien x1² + x2² en troisième colonne.",
                f"lift_17([[1, 2], [0, -3]]) rend {np.round(np.asarray(probe_17, dtype=float), 3).tolist()} au lieu de "
                "[[1, 2, 5], [0, -3, 9]] : les deux coordonnées, puis la somme de leurs carrés.")
        model_17 = SVC(kernel="linear", C=1).fit(lift_17(X_TRAIN_17), Y_TRAIN_17)
        fig = plt.figure(figsize=(13, 5))
        ax_17 = fig.add_subplot(1, 2, 1)
        wb.plot.plot_decision_boundary(lambda points: model_17.predict(lift_17(points)), X_TEST_17, Y_TEST_17,
                                       ax=ax_17, title="the plane, seen from above (test points)")
        draw_17(fig.add_subplot(1, 2, 2, projection="3d"), lift_17(X_TRAIN_17), Y_TRAIN_17, model_17)
        plt.show()'''

GIVEN_18 = r'''X_18, Y_18 = wb.synth.make_spirals(n=400, noise=0.8, seed=1318)   # two intertwined spirals
X_TRAIN_18, X_TEST_18, Y_TRAIN_18, Y_TEST_18 = train_test_split(X_18, Y_18, test_size=0.3, random_state=1318,
                                                                stratify=Y_18)
GRID_18 = {"C": [0.1, 1, 10, 100, 1000], "gamma": [0.1, 1, 10, 100, 1000]}
CV_18 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1318)'''

TODO_18 = r'''search_18 = ...    # GridSearchCV(SVC(kernel="rbf"), GRID_18, cv=CV_18), fitted on X_TRAIN_18, Y_TRAIN_18
best_18 = ...      # a) [C, gamma] of the best combination
best_cv_18 = ...   # b) its mean accuracy in cross-validation
test_18 = ...      # c) the test accuracy of search_18 (refitted on the whole training part)
overfit_18 = ...   # d) [training accuracy, test accuracy] of SVC(C=1000, gamma=1000) fitted on the training part'''

SOLUTION_18 = r'''search_18 = GridSearchCV(SVC(kernel="rbf"), GRID_18, cv=CV_18).fit(X_TRAIN_18, Y_TRAIN_18)
best_18 = [search_18.best_params_["C"], search_18.best_params_["gamma"]]
best_cv_18 = search_18.best_score_
test_18 = search_18.score(X_TEST_18, Y_TEST_18)
overfit_model_18 = SVC(C=1000, gamma=1000).fit(X_TRAIN_18, Y_TRAIN_18)
overfit_18 = [overfit_model_18.score(X_TRAIN_18, Y_TRAIN_18), overfit_model_18.score(X_TEST_18, Y_TEST_18)]
'''

CHECK_18 = r'''wb.check("13.18a", best_18)
wb.check("13.18b", best_cv_18)
wb.check("13.18c", test_18)
wb.check("13.18d", overfit_18)
with wb.attempt("13.18"):
    if not filled(search_18) or not hasattr(search_18, "cv_results_"):
        raise NotImplementedError("search_18")
    scores_18 = search_18.cv_results_["mean_test_score"].reshape(len(GRID_18["C"]), len(GRID_18["gamma"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    image = axes[0].imshow(scores_18, cmap="viridis", vmin=0.5, vmax=1)
    axes[0].set_xticks(range(5), GRID_18["gamma"])
    axes[0].set_yticks(range(5), GRID_18["C"])
    axes[0].set_xlabel("gamma")
    axes[0].set_ylabel("C")
    axes[0].set_title("mean accuracy in cross-validation")
    axes[0].grid(False)
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f"{scores_18[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axes[0])
    wb.plot.plot_decision_boundary(search_18.best_estimator_, X_TEST_18, Y_TEST_18, ax=axes[1],
                                   title=f"best: C = {search_18.best_params_['C']}, gamma = {search_18.best_params_['gamma']}")
    wb.plot.plot_decision_boundary(SVC(C=1000, gamma=1000).fit(X_TRAIN_18, Y_TRAIN_18), X_TEST_18, Y_TEST_18,
                                   ax=axes[2], title="C = 1000, gamma = 1000")
    plt.show()'''

PART_B = Part("B", "Les machines à vecteurs de support",
              "Les SVM de scikit-learn (`sklearn.svm.SVC`) : la marge et les vecteurs de support d'un SVM linéaire "
              "(13.16), une transformation que tu calcules toi-même pour séparer des cercles par un plan (13.17), puis "
              "le noyau RBF réglé par validation croisée (13.18). Le workbook ne te fait pas coder de SVM : "
              "l'optimisation de la marge demande des outils (la programmation quadratique) qui dépassent le chapitre. "
              "La cellule suivante définit `draw_band`, qui trace la bande d'un SVM en 2D.",
              given=PART_B_GIVEN, exercises=[
    Ex("13.16", "📦", 2, 25, "SVM linéaire : vecteurs de support et paramètre C",
       "lire la marge et les vecteurs de support d'un SVM linéaire, et voir ce que C change.",
       "✏️ 13.4 · fiche §13.4", thread="synthétique", tracks="R, M, C",
       body=r"""`X_16` : 200 points de deux classes (0 et 1), deux nuages gaussiens qui se chevauchent un peu. On entraîne `SVC(kernel="linear", C=C)` pour les trois valeurs de `CS_16` ; sa fonction de décision est $f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$, et la bande est l'ensemble des points où $-1 < f(\mathbf{x}) < 1$. Calcule :
a) `n_support_16` : la liste des nombres totaux de vecteurs de support, un par valeur de `CS_16` (dans cet ordre) ;
b) `width_16` : pour $C = 1$, la largeur de la bande, la distance entre les droites $f(\mathbf{x}) = -1$ et $f(\mathbf{x}) = 1$ ;
c) `inside_16` : pour $C = 1$, le nombre de points tels que $y_i f(\mathbf{x}_i) < 0{,}999$, avec $y_i = -1$ pour la classe 0 et $+1$ pour la classe 1 (les points dans la bande ou du mauvais côté ; le seuil 0,999 tolère l'arrondi des points posés sur un bord).

La vérification contrôle a) à c), puis trace les trois bandes, vecteurs de support entourés.

Dans tes notes : comment la largeur de la bande et le nombre de vecteurs de support varient-ils avec C, et pourquoi ? Réentraîne le SVM ($C = 1$) sur ses seuls vecteurs de support : que deviennent $\mathbf{w}$ et $b$, et qu'est-ce que cela dit de ce que le SVM « retient » des données ?""",
       given=GIVEN_16, todo=TODO_16, check=CHECK_16, solution=SOLUTION_16 + as_printed(CHECK_16, "13.16"),
       record=r'''wb.record("13.16a", n_support_16,
          mistakes={"les nombres sont dans l'ordre de CS_16 : C = 0,01, puis 1, puis 100": n_support_16[::-1]})
wb.record("13.16b", width_16, decimals=4,
          mistakes={"c'est la demi-largeur, la distance de la frontière à un bord : la bande va de f = -1 à f = +1": width_16 / 2,
                    "c'est la norme de w : la distance d'un point à la droite f = 0 vaut |f| / ||w||": float(np.linalg.norm(w_16))})
wb.record("13.16c", inside_16,
          mistakes={"ce sont tous les vecteurs de support : ceux qui sont posés sur un bord ont y f(x) = 1, ils ne sont pas dans la bande": int(len(sv_16)),
                    "ce sont les points mal classés (y f(x) < 0) : compte aussi les points bien classés qui sont dans la bande": int(np.sum(signs_16 * models_16[1].decision_function(X_16) < 0)),
                    "un point posé sur un bord a y f(x) = 0,9999997, à l'arrondi près : utilise le seuil 0,999 de l'énoncé": int(np.sum(signs_16 * models_16[1].decision_function(X_16) < 1))})''',
       note="Petit C (0,01) : les violations coûtent peu, la bande s'élargit (2,80) et contient 84 points, tous "
            "vecteurs de support (86). Grand C (100) : les violations coûtent cher, la bande se resserre (0,54) et "
            "il ne reste que 14 vecteurs de support. À $C = 1$, la bande mesure 1,148 ; 16 points sont dedans ou du "
            "mauvais côté, dont 6 mal classés, et 2 vecteurs de support sont posés exactement sur un bord : 18 en "
            "tout. Réentraîné sur ces 18 points seulement, le SVM retrouve le même $\\mathbf{w}$ et le même $b$ "
            "(à $10^{-7}$ près) : les 182 autres points, hors de la bande et du bon côté, ne comptent pas pour "
            "l'hyperplan."),

    Ex("13.17", "🎨", 2, 30, "Relever les cercles en 3D pour les séparer par un plan",
       "voir qu'une transformation des features rend linéaire ce qui ne l'était pas, et le dessiner.",
       "Ex 13.16 · ✏️ 13.5 · fiche §13.4 (astuce du noyau)", thread="synthétique", tracks="M, C",
       body=r"""`X_17` : un anneau (classe 0) autour d'un disque (classe 1), deux cercles bruités de rayons 1 et 0,4 ; 70 % pour l'entraînement, 30 % pour le test. Aucune droite ne les sépare ; on les **relève** en ajoutant une troisième coordonnée, $x_3 = x_1^2 + x_2^2$, le carré de la distance au centre. Écris :
- `lift_17(X)` : le tableau `(n, 3)` des deux coordonnées, puis de $x_1^2 + x_2^2$ ;
- `draw_17(ax, X3, y, model)` : sur un axe 3D, les points relevés colorés par classe, et le plan $\mathbf{w} \cdot \mathbf{z} + b = 0$ du SVM linéaire `model` (résous-le en $z_3$ sur une grille de $(z_1, z_2)$, et trace-le avec `ax.plot_surface`) ;
- a) `acc_2d_17` : l'accuracy de test de `SVC(kernel="linear", C=1)` entraîné sur les deux coordonnées ;
- b) `acc_3d_17` : la même chose sur les trois coordonnées relevées ;
- c) `radius_17` : la frontière de b), revenue dans le plan des données, est un cercle : son rayon (garde tous les chiffres).

La vérification contrôle a) à c), essaie `lift_17` sur deux points, puis dessine la frontière vue de dessus et ta figure 3D.

Dans tes notes : pourquoi le rayon trouvé tombe-t-il entre 0,4 et 1 ? Écris le produit scalaire $\varphi(\mathbf{x}) \cdot \varphi(\mathbf{z})$ de ta transformation en fonction de $\mathbf{x}$ et $\mathbf{z}$ seulement : quel noyau as-tu, sans le savoir, utilisé ? Qu'aurait fait un noyau RBF sur ces données ?""",
       given=GIVEN_17, todo=TODO_17, check=CHECK_17, solution=SOLUTION_17 + as_printed(CHECK_17, "13.17"),
       record=r'''wb.record("13.17a", acc_2d_17, decimals=4,
          mistakes={"c'est l'accuracy d'entraînement : mesure-la sur X_TEST_17": SVC(kernel="linear", C=1).fit(X_TRAIN_17, Y_TRAIN_17).score(X_TRAIN_17, Y_TRAIN_17)})
wb.record("13.17b", acc_3d_17, decimals=4,
          mistakes={"c'est l'accuracy d'entraînement : mesure-la sur lift_17(X_TEST_17)": model_3d_17.score(lift_17(X_TRAIN_17), Y_TRAIN_17)})
wb.record("13.17c", radius_17, decimals=2,
          mistakes={"c'est le carré du rayon : sur le cercle, x1² + x2² = r²": radius_17 ** 2,
                    "c'est le milieu des deux rayons, deviné : calcule-le à partir de w et b": 0.70})''',
       note="Sur les deux coordonnées, un SVM linéaire fait 0,61, à peine mieux que la classe majoritaire (0,5) ; relevés, "
            "les points se séparent presque tous par un plan (0,983 en test). Le plan a $w_1$ et $w_2$ presque nuls "
            "et $w_3 \\approx -4{,}13$ : il coupe en $x_3 \\approx 0{,}516$, soit un cercle de rayon 0,72, entre "
            "le disque (0,4) et l'anneau (1), à peu près au milieu, comme le veut une marge maximale. Le produit "
            "scalaire des points relevés vaut $\\mathbf{x} \\cdot \\mathbf{z} + \\lVert \\mathbf{x} \\rVert^2 "
            "\\lVert \\mathbf{z} \\rVert^2$ : un noyau calculable sans relever les points, dans l'esprit de ✏️ 13.5. "
            "Un noyau RBF aurait trouvé une frontière fermée semblable, sans qu'on choisisse la transformation."),

    Ex("13.18", "📦", 3, 35, "Noyau RBF : régler C et gamma par validation croisée",
       "régler ensemble C et $\\gamma$ sur une grille logarithmique, et reconnaître un $\\gamma$ trop grand.",
       "Ex 13.16 · ch. 8 (validation croisée) · fiche §13.4", thread="synthétique", tracks="C",
       body=r"""`X_18` : deux spirales entrelacées et bruitées (400 points, 70 % pour l'entraînement). `GridSearchCV(estimator, param_grid, cv=...)` de scikit-learn (le ch. 15 y revient en détail) essaie toutes les combinaisons du dictionnaire `param_grid` en validation croisée, puis réentraîne la meilleure sur toute la partie d'entraînement. Après `fit`, son attribut `best_params_` est le dictionnaire de la combinaison retenue, et sa méthode `score(X, y)` note le modèle réentraîné. Avec `GridSearchCV` et `SVC(kernel="rbf")`, essaie les 25 combinaisons de `GRID_18` en validation croisée à 5 folds (`CV_18`), sur la partie d'entraînement. Calcule :
a) `best_18` : la liste `[C, gamma]` de la meilleure combinaison ;
b) `best_cv_18` : son accuracy moyenne en validation croisée ;
c) `test_18` : l'accuracy de test du modèle réentraîné sur toute la partie d'entraînement avec cette combinaison ;
d) `overfit_18` : la liste `[accuracy d'entraînement, accuracy de test]` de `SVC(C=1000, gamma=1000)` entraîné sur la partie d'entraînement.

La vérification contrôle a) à d), puis trace la carte des accuracies de validation croisée et deux frontières.

Dans tes notes : décris la carte : où sont les bonnes combinaisons ? Que fait un noyau de $\gamma = 1\,000$ autour de chaque vecteur de support, et pourquoi C n'y change presque rien ? Pourquoi la grille est-elle logarithmique ?""",
       given=GIVEN_18, todo=TODO_18, check=CHECK_18, solution=SOLUTION_18 + as_printed(CHECK_18, "13.18"),
       record=r'''wb.record("13.18a", best_18, mistakes={"l'ordre demandé est [C, gamma]": best_18[::-1]})
wb.record("13.18b", best_cv_18, decimals=4,
          mistakes={"c'est l'accuracy de test : on demande la moyenne de validation croisée (best_score_)": test_18})
wb.record("13.18c", test_18, decimals=4,
          mistakes={"c'est le SVC par défaut (gamma=\"scale\", C=1) : utilise le modèle réglé par la recherche": SVC().fit(X_TRAIN_18, Y_TRAIN_18).score(X_TEST_18, Y_TEST_18),
                    "c'est la moyenne de validation croisée : on demande l'accuracy sur X_TEST_18": best_cv_18})
wb.record("13.18d", overfit_18, decimals=4,
          mistakes={"l'ordre demandé est [entraînement, test]": overfit_18[::-1]})''',
       note="La meilleure combinaison est $C = 1$, $\\gamma = 10$ (0,929 en validation croisée, 0,95 en test). La "
            "carte montre une bande de bonnes valeurs autour de $\\gamma = 10$ à 100, et une compensation entre les "
            "deux réglages : un petit $\\gamma$ (frontière trop lisse, 0,56 à 0,87) se rattrape en partie par un grand "
            "C. Avec $\\gamma = 1\\,000$, le noyau d'un vecteur de support vaut $e^{-1000 d^2}$ : il tombe à 0,37 dès "
            "$d \\approx 0{,}03$, si bien que chaque point d'entraînement crée sa petite bosse et que presque tous "
            "deviennent vecteurs de support. L'entraînement est parfait (1,0), le test chute à 0,83, et C n'y change "
            "presque rien (0,74 à 0,75 en validation croisée de $C = 1$ à $1\\,000$) : la frontière est déjà aussi souple que "
            "possible. La grille est logarithmique parce que C et $\\gamma$ agissent par ordres de grandeur."),
])

# ---------------------------------------------------------------------------
# Part C: decision trees (13.19 to 13.26)
# ---------------------------------------------------------------------------
PART_C_GIVEN = r'''from sklearn.tree import export_text, plot_tree


def tree_rules(node, feature_names, class_names=None, indent=""):
    """The rules of a mylearn tree (a Node and its children), one line per test, like export_text."""
    if node.left is None:
        value = np.ravel(np.asarray(node.value, dtype=float))
        what = f"class: {class_names[int(np.argmax(value))]}" if class_names is not None else f"value: {value[0]:.3f}"
        return [f"{indent}|--- {what}  ({node.n_samples} samples)"]
    name = feature_names[node.feature]
    return ([f"{indent}|--- {name} <= {node.threshold:.2f}"]
            + tree_rules(node.left, feature_names, class_names, indent + "|   ")
            + [f"{indent}|--- {name} >  {node.threshold:.2f}"]
            + tree_rules(node.right, feature_names, class_names, indent + "|   "))


def as_float32(X):
    """X rounded to float32, then back to float64: scikit-learn's trees work in float32, so this gives both libraries
    exactly the same values (and the same thresholds)."""
    return np.asarray(X, dtype=np.float32).astype(np.float64)'''

GIVEN_19 = r'''X_19, Y_19 = wb.synth.make_moons(n=600, noise=0.35, seed=1319)
DRAWS_19 = [np.random.default_rng(1319 + r).choice(600, 420, replace=False) for r in range(20)]   # 20 draws of 70 %
grid1_19, grid2_19 = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1.2, 1.7, 150))
GRID_19 = np.column_stack([grid1_19.ravel(), grid2_19.ravel()])   # 30,000 points of the plane
DEPTHS_19 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
CV_19 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1319)'''

TODO_19 = r'''def trees_19(max_depth):
    """The list of 20 trees DecisionTreeClassifier(max_depth=max_depth, random_state=0), one fitted on each draw of
    DRAWS_19 (the rows X_19[draw], Y_19[draw])."""
    raise NotImplementedError("trees_19")


def disagreement_19(trees):
    """Mean, over all the pairs of trees of the list, of the share of the points of GRID_19 where the two trees
    predict different classes."""
    raise NotImplementedError("disagreement_19")


leaves_19 = ...       # a) the mean number of leaves of the 20 trees of trees_19(None)
disagree_19 = ...     # b) [disagreement_19(trees_19(None)), disagreement_19(trees_19(3))]
root_range_19 = ...   # c) the largest minus the smallest root threshold (tree_.threshold[0]) of the trees of trees_19(None)
best_depth_19 = ...   # d) the max_depth of DEPTHS_19 with the best mean 5-fold accuracy (CV_19) on X_19, Y_19'''

SOLUTION_19 = r'''def trees_19(max_depth):
    """The list of 20 trees DecisionTreeClassifier(max_depth=max_depth, random_state=0), one fitted on each draw of
    DRAWS_19 (the rows X_19[draw], Y_19[draw])."""
    return [DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X_19[draw], Y_19[draw]) for draw in DRAWS_19]


def disagreement_19(trees):
    """Mean, over all the pairs of trees of the list, of the share of the points of GRID_19 where the two trees
    predict different classes."""
    predictions = np.array([tree.predict(GRID_19) for tree in trees])
    pairs = [(i, j) for i in range(len(trees)) for j in range(i + 1, len(trees))]
    return float(np.mean([np.mean(predictions[i] != predictions[j]) for i, j in pairs]))


full_trees_19 = trees_19(None)
leaves_19 = float(np.mean([tree.get_n_leaves() for tree in full_trees_19]))
disagree_19 = [disagreement_19(full_trees_19), disagreement_19(trees_19(3))]
roots_19 = [tree.tree_.threshold[0] for tree in full_trees_19]
root_range_19 = float(max(roots_19) - min(roots_19))
cv_means_19 = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), X_19, Y_19, cv=CV_19).mean()
               for d in DEPTHS_19]
best_depth_19 = DEPTHS_19[int(np.argmax(cv_means_19))]
print("root features:", sorted({int(tree.tree_.feature[0]) for tree in full_trees_19}),
      "· root thresholds:", np.round(sorted(roots_19), 3).tolist())
print("leaves: from", min(tree.get_n_leaves() for tree in full_trees_19), "to", max(tree.get_n_leaves() for tree in full_trees_19))
'''

CHECK_19 = r'''wb.check("13.19a", leaves_19)
wb.check("13.19b", disagree_19)
wb.check("13.19c", root_range_19)
wb.check("13.19d", best_depth_19)
with wb.attempt("13.19"):
    if not filled(leaves_19, best_depth_19):
        raise NotImplementedError("leaves_19, best_depth_19")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (depth, r) in zip(axes, [(None, 0), (None, 1), (3, 0), (3, 1)]):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_19[DRAWS_19[r]], Y_19[DRAWS_19[r]])
        wb.plot.plot_decision_boundary(tree, X_19[DRAWS_19[r]], Y_19[DRAWS_19[r]], ax=ax,
                                       title=f"draw {r}, max_depth={depth}: {tree.get_n_leaves()} leaves")
    plt.show()
    train_19 = [DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_19, Y_19).score(X_19, Y_19) for d in DEPTHS_19]
    cv_19 = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), X_19, Y_19, cv=CV_19).mean()
             for d in DEPTHS_19]
    labels_19 = [str(d) for d in DEPTHS_19]
    plt.plot(labels_19, train_19, marker="o", label="training")
    plt.plot(labels_19, cv_19, marker="o", label="5-fold cross-validation")
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()'''

CHECK_20 = r'''with checked("13.20"):
    cases_20 = [("gini_impurity([0, 0, 1, 1])", lambda: mylearn.tree.gini_impurity([0, 0, 1, 1]), 0.5),
                ('gini_impurity(["a", "a", "a", "b"])', lambda: mylearn.tree.gini_impurity(["a", "a", "a", "b"]), 0.375),
                ("gini_impurity([0, 1], sample_weight=[3.0, 1.0])",
                 lambda: mylearn.tree.gini_impurity([0, 1], sample_weight=[3.0, 1.0]), 0.375),
                ("entropy_impurity([0, 0, 0, 1])", lambda: mylearn.tree.entropy_impurity([0, 0, 0, 1]), 0.811278),
                ("variance_impurity([1.0, 2.0, 3.0, 4.0])", lambda: mylearn.tree.variance_impurity([1.0, 2.0, 3.0, 4.0]), 1.25),
                ('split_gain([0, 0, 1, 1], [True, True, False, False], criterion="entropy")',
                 lambda: mylearn.tree.split_gain([0, 0, 1, 1], [True, True, False, False], criterion="entropy"), 1.0)]
    for text_20, call_20, want_20 in cases_20:
        try:
            got_20 = call_20()
        except NotImplementedError:
            print(f"⏳ {text_20} : pas encore écrite")
            continue
        ok_20 = isinstance(got_20, (int, float, np.floating)) and np.isclose(got_20, want_20, atol=1e-6)
        print(("✅ " if ok_20 else "❌ ") + f"{text_20} = {got_20!r} (expected {want_20})")
with checked("13.20"):
    X_20, y_20 = P_TRAIN[MEASURES].to_numpy(), P_TRAIN["species"].to_numpy()
    for criterion, impurity_20 in [("gini", mylearn.tree.gini_impurity), ("entropy", mylearn.tree.entropy_impurity)]:
        t_20 = DecisionTreeClassifier(criterion=criterion, max_depth=1, random_state=0).fit(X_20, y_20).tree_
        left_20, right_20 = t_20.children_left[0], t_20.children_right[0]
        sk_gain_20 = t_20.impurity[0] - (t_20.n_node_samples[left_20] * t_20.impurity[left_20]
                                         + t_20.n_node_samples[right_20] * t_20.impurity[right_20]) / t_20.n_node_samples[0]
        mask_20 = as_float32(X_20)[:, t_20.feature[0]] <= t_20.threshold[0]
        root_20 = impurity_20(y_20)
        gain_20 = mylearn.tree.split_gain(y_20, mask_20, criterion=criterion)
        if not (returned("13.20", f"{criterion}_impurity", root_20) and returned("13.20", "split_gain", gain_20)):
            continue
        if not all(isinstance(value, (int, float, np.floating)) for value in (root_20, gain_20)):
            print(f"❌ Ex 13.20 : {criterion}_impurity et split_gain doivent rendre un nombre (un float), pas "
                  f"{type(root_20).__name__} et {type(gain_20).__name__}.")
        else:
            verdict("13.20", np.isclose(root_20, t_20.impurity[0]) and np.isclose(gain_20, sk_gain_20),
                    f"{criterion} : impureté de la racine {fr(root_20)} et gain du premier split de scikit-learn "
                    f"{fr(gain_20)}, comme scikit-learn.",
                    f"{criterion} : scikit-learn trouve une impureté de {fr(t_20.impurity[0])} à la racine et un gain "
                    f"de {fr(sk_gain_20)} pour son premier split ; les tests ci-dessous disent où est l'écart.")
if written(lambda: mylearn.tree.gini_impurity([0, 1])):
    run_mylearn_tests("tree", keyword="test_gini_impurity_ or test_entropy_impurity_ or test_variance_impurity_ or test_split_gain_")
else:
    print("⏳ Ex 13.20 : les tests attendent ton gini_impurity.")'''

GIVEN_21 = r'''# A colleague saves the 333 penguins in a CSV file (index included, the default of to_csv), then reads it back
CSV_21 = PENGUINS[MEASURES + ["species"]].to_csv()
DATA_21 = pd.read_csv(io.StringIO(CSV_21))
FEATURES_21 = [column for column in DATA_21.columns if column != "species"]   # "every column but the target"
TRAIN_21, TEST_21 = train_test_split(DATA_21, test_size=0.3, random_state=1321, stratify=DATA_21["species"])
TREE_21 = DecisionTreeClassifier(random_state=0).fit(TRAIN_21[FEATURES_21], TRAIN_21["species"])
print(f"the colleague's tree: test accuracy {TREE_21.score(TEST_21[FEATURES_21], TEST_21['species']):.3f}")
# Next season: the test penguins are measured again, appended to the file, which is saved and read back
NEXT_FILE_21 = pd.concat([PENGUINS[MEASURES + ["species"]], PENGUINS.loc[TEST_21.index, MEASURES + ["species"]]],
                         ignore_index=True).to_csv()
NEXT_21 = pd.read_csv(io.StringIO(NEXT_FILE_21)).iloc[len(PENGUINS):]
print(len(NEXT_21), "penguins next season")'''

TODO_21 = r'''top_feature_21 = ...      # a) the name of the feature of FEATURES_21 with the largest importance in TREE_21
next_accuracy_21 = ...    # b) the accuracy of TREE_21 on NEXT_21
clean_accuracy_21 = ...   # c) the test accuracy (TEST_21) of DecisionTreeClassifier(random_state=0) fitted on TRAIN_21
                          #    without the guilty column
clean_leaves_21 = ...     # d) the number of leaves of the tree of c)'''

SOLUTION_21 = r'''print(export_text(TREE_21, feature_names=FEATURES_21))
top_feature_21 = FEATURES_21[int(np.argmax(TREE_21.feature_importances_))]
next_accuracy_21 = TREE_21.score(NEXT_21[FEATURES_21], NEXT_21["species"])
print("next season, predicted species:", pd.Series(TREE_21.predict(NEXT_21[FEATURES_21])).value_counts().to_dict())
clean_21 = DecisionTreeClassifier(random_state=0).fit(TRAIN_21[MEASURES], TRAIN_21["species"])
clean_accuracy_21 = clean_21.score(TEST_21[MEASURES], TEST_21["species"])
clean_leaves_21 = clean_21.get_n_leaves()
'''

CHECK_21 = r'''wb.check("13.21a", top_feature_21)
wb.check("13.21b", next_accuracy_21)
wb.check("13.21c", clean_accuracy_21)
wb.check("13.21d", clean_leaves_21)'''

CHECK_22 = r'''with checked("13.22"):
    doc_22 = mylearn.tree.best_split(np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 8.0], [4.0, 1.0]]), np.array([0, 0, 1, 1]))
    if returned("13.22", "best_split", doc_22):
        print("docstring example:", doc_22, "(expected (0, 2.5, 0.5))")
with checked("13.22"):
    X_22, y_22 = as_float32(P_TRAIN[MEASURES]), P_TRAIN["species"].to_numpy()
    for criterion in ["gini", "entropy"]:
        t_22 = DecisionTreeClassifier(criterion=criterion, max_depth=1, random_state=0).fit(X_22, y_22).tree_
        left_22, right_22 = t_22.children_left[0], t_22.children_right[0]
        sk_gain_22 = t_22.impurity[0] - (t_22.n_node_samples[left_22] * t_22.impurity[left_22]
                                         + t_22.n_node_samples[right_22] * t_22.impurity[right_22]) / t_22.n_node_samples[0]
        mine_22 = mylearn.tree.best_split(X_22, y_22, criterion=criterion)
        ok_22 = (isinstance(mine_22, tuple) and len(mine_22) == 3 and mine_22[0] == t_22.feature[0]
                 and np.isclose(mine_22[1], t_22.threshold[0]) and np.isclose(mine_22[2], sk_gain_22))
        verdict("13.22", ok_22,
                f"{criterion} : {MEASURES[t_22.feature[0]]} <= {fr(t_22.threshold[0], 2)}, gain {fr(sk_gain_22)}, comme "
                "la racine de scikit-learn.",
                f"{criterion} : best_split rend {mine_22} ; scikit-learn coupe la feature {t_22.feature[0]} "
                f"({MEASURES[t_22.feature[0]]}) en {t_22.threshold[0]:.4f}, avec un gain de {sk_gain_22:.6f}.")
if written(lambda: mylearn.tree.best_split(np.array([[0.0], [1.0]]), np.array([0, 1]))):
    run_mylearn_tests("tree", keyword="test_best_split_")
else:
    print("⏳ Ex 13.22 : les tests attendent ton best_split.")'''

GIVEN_23 = r'''TRAIN_23, TEST_23 = train_test_split(PENGUINS, test_size=0.3, random_state=1323, stratify=PENGUINS["species"])
X_TRAIN_23, Y_TRAIN_23 = TRAIN_23[MEASURES].to_numpy(), TRAIN_23["species"].to_numpy()
X_TEST_23, Y_TEST_23 = TEST_23[MEASURES].to_numpy(), TEST_23["species"].to_numpy()
CV_23 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1323)
print(len(TRAIN_23), "training penguins,", len(TEST_23), "test penguins")'''

TODO_23 = r'''full_23 = ...         # DecisionTreeClassifier(random_state=0) fitted on X_TRAIN_23, Y_TRAIN_23
size_23 = ...         # a) [depth, number of leaves] of full_23
alphas_23 = ...       # the ccp_alphas of full_23.cost_complexity_pruning_path(X_TRAIN_23, Y_TRAIN_23), without the last
best_alpha_23 = ...   # b) the alpha of alphas_23 with the best mean 5-fold accuracy (CV_23) of
                      #    DecisionTreeClassifier(ccp_alpha=alpha, random_state=0); the LARGEST one if tied (np.isclose)
pruned_23 = ...       # DecisionTreeClassifier(ccp_alpha=best_alpha_23, random_state=0) fitted on the training part
pruned_size_23 = ...  # c) [depth, number of leaves] of pruned_23
test_23 = ...         # d) [test accuracy of full_23, test accuracy of pruned_23]'''

SOLUTION_23 = r'''full_23 = DecisionTreeClassifier(random_state=0).fit(X_TRAIN_23, Y_TRAIN_23)
size_23 = [full_23.get_depth(), full_23.get_n_leaves()]
alphas_23 = full_23.cost_complexity_pruning_path(X_TRAIN_23, Y_TRAIN_23).ccp_alphas[:-1]
cv_23 = np.array([cross_val_score(DecisionTreeClassifier(ccp_alpha=alpha, random_state=0), X_TRAIN_23, Y_TRAIN_23,
                                  cv=CV_23).mean() for alpha in alphas_23])
best_alpha_23 = float(alphas_23[np.isclose(cv_23, cv_23.max())].max())
pruned_23 = DecisionTreeClassifier(ccp_alpha=best_alpha_23, random_state=0).fit(X_TRAIN_23, Y_TRAIN_23)
pruned_size_23 = [pruned_23.get_depth(), pruned_23.get_n_leaves()]
test_23 = [full_23.score(X_TEST_23, Y_TEST_23), pruned_23.score(X_TEST_23, Y_TEST_23)]
for alpha, score in zip(alphas_23, cv_23):
    print(f"alpha = {alpha:.5f}: mean accuracy {score:.4f}")
'''

CHECK_23 = r'''wb.check("13.23a", size_23)
wb.check("13.23b", best_alpha_23)
wb.check("13.23c", pruned_size_23)
wb.check("13.23d", test_23)
with wb.attempt("13.23"):
    if not filled(pruned_23) or not hasattr(pruned_23, "tree_"):
        raise NotImplementedError("pruned_23")
    plt.figure(figsize=(12, 6))
    plot_tree(pruned_23, feature_names=MEASURES, class_names=list(pruned_23.classes_), filled=True, fontsize=9)
    plt.show()
    print(export_text(pruned_23, feature_names=MEASURES))'''

CHECK_24 = r'''with checked("13.24"):
    doc_24 = fitted(mylearn.tree.DecisionTreeClassifier(), np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))
    root_24 = getattr(doc_24, "root_", None)
    print("docstring example: root", None if root_24 is None else (root_24.feature, root_24.threshold),
          "(expected (0, 2.5)) · depth", doc_24.get_depth(), "and leaves", doc_24.get_n_leaves(), "(expected 1 and 2)")
with checked("13.24"):
    X_24, Y_24 = wb.synth.make_moons(n=300, noise=0.3, seed=1324)
    X_24 = as_float32(X_24)                                       # continuous data, no ties
    grid1_24, grid2_24 = np.meshgrid(np.linspace(-1.5, 2.5, 80), np.linspace(-1.2, 1.7, 60))
    grid_24 = as_float32(np.column_stack([grid1_24.ravel(), grid2_24.ravel()]))
    for params in [{"max_depth": 3}, {"min_samples_split": 10}, {"criterion": "entropy", "min_samples_leaf": 5}]:
        mine_24 = fitted(mylearn.tree.DecisionTreeClassifier(**params), X_24, Y_24)
        oracle_24 = DecisionTreeClassifier(random_state=0, **params).fit(X_24, Y_24)
        shape_24 = (mine_24.get_depth(), mine_24.get_n_leaves())
        if shape_24 != (oracle_24.get_depth(), oracle_24.get_n_leaves()):
            print(f"❌ Ex 13.24 : {params} : scikit-learn a une profondeur de {oracle_24.get_depth()} et "
                  f"{oracle_24.get_n_leaves()} feuilles, ton arbre {shape_24[0]} et {shape_24[1]} ; les tests "
                  "ci-dessous disent où.")
            continue
        pred_24 = mine_24.predict(grid_24)
        if not returned("13.24", "predict", pred_24):
            continue
        pred_24 = np.asarray(pred_24)
        if pred_24.shape != (len(grid_24),):
            print(f"❌ Ex 13.24 : {params} : predict doit rendre un tableau ({len(grid_24)},), un label par ligne ; "
                  f"reçu la forme {pred_24.shape}.")
        elif not np.array_equal(pred_24, oracle_24.predict(grid_24)):
            print(f"❌ Ex 13.24 : {params} : même profondeur et mêmes feuilles que scikit-learn, mais d'autres "
                  "prédictions sur la grille : un seuil ou une feature diffère ; les tests ci-dessous disent où.")
        else:
            print(f"✅ Ex 13.24 : {params} : même profondeur ({shape_24[0]}), même nombre de feuilles ({shape_24[1]}) "
                  "et mêmes prédictions que scikit-learn.")
with checked("13.24"):
    penguin_tree_24 = fitted(mylearn.tree.DecisionTreeClassifier(max_depth=2), P_TRAIN[MEASURES].to_numpy(),
                             P_TRAIN["species"].to_numpy())
    print("\n".join(tree_rules(penguin_tree_24.root_, MEASURES, penguin_tree_24.classes_)))
    print("test accuracy of this depth-2 tree:", fr(penguin_tree_24.score(P_TEST[MEASURES].to_numpy(), P_TEST["species"].to_numpy())))
if written(lambda: mylearn.tree.DecisionTreeClassifier().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))):
    run_mylearn_tests("tree", keyword="test_decision_tree_classifier_ and not ensemble")
else:
    print("⏳ Ex 13.24 : les tests attendent le fit de ton DecisionTreeClassifier.")'''

CHECK_25 = r'''with checked("13.25"):
    X_25, Y_25 = wb.synth.make_moons(n=200, noise=0.3, seed=1325)
    rng_25 = np.random.default_rng(1325)
    X_25 = as_float32(np.hstack([X_25, rng_25.normal(size=(200, 3))]))   # 2 useful features, then 3 of noise
    W_25 = rng_25.integers(1, 4, size=200).astype(float)                   # weights 1, 2 or 3
    weighted_25 = fitted(mylearn.tree.DecisionTreeClassifier(max_depth=4), X_25, Y_25, sample_weight=W_25)
    repeated_25 = fitted(mylearn.tree.DecisionTreeClassifier(max_depth=4), np.repeat(X_25, W_25.astype(int), axis=0),
                         np.repeat(Y_25, W_25.astype(int)))
    verdict("13.25", np.allclose(weighted_25.predict_proba(X_25), repeated_25.predict_proba(X_25)),
            "des poids 1, 2 ou 3 donnent le même arbre que des exemples répétés 1, 2 ou 3 fois.",
            "les poids ne font pas le même arbre que des exemples répétés : relis comment ils entrent dans les "
            "proportions, les impuretés et les gains.")
with checked("13.25"):
    mine_25 = fitted(mylearn.tree.DecisionTreeClassifier(max_depth=5), X_25, Y_25)
    oracle_25 = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_25, Y_25)
    print("feature_importances_ (yours):        ", np.round(np.asarray(mine_25.feature_importances_, dtype=float), 4).tolist())
    print("feature_importances_ (scikit-learn): ", np.round(oracle_25.feature_importances_, 4).tolist())
    verdict("13.25", np.allclose(mine_25.feature_importances_, oracle_25.feature_importances_),
            "mêmes importances que scikit-learn.", "tes importances diffèrent de celles de scikit-learn.")
with checked("13.25"):
    grid_25 = as_float32(np.random.default_rng(25).normal(size=(500, 5)))
    first_25 = mylearn.tree.DecisionTreeClassifier(max_features="sqrt", random_state=0)
    first_predictions_25 = fitted(first_25, X_25, Y_25).predict(grid_25)
    refit_predictions_25 = fitted(first_25, X_25, Y_25).predict(grid_25)            # the same object, fitted again
    other_25 = fitted(mylearn.tree.DecisionTreeClassifier(max_features="sqrt", random_state=1), X_25, Y_25)
    verdict("13.25", np.array_equal(first_predictions_25, refit_predictions_25)
            and not np.array_equal(first_predictions_25, other_25.predict(grid_25)),
            "max_features=\"sqrt\" : un second fit du même arbre redonne le même arbre, un autre random_state en "
            "donne un autre.",
            "max_features=\"sqrt\" : un second fit avec le même random_state doit redonner le même arbre (le "
            "générateur est créé dans fit, pas dans __init__), et un autre random_state un autre arbre.")
if written(lambda: mylearn.tree.DecisionTreeClassifier(max_features=1, random_state=0).fit(np.array([[0.0], [1.0]]), np.array([0, 1]))):
    run_mylearn_tests("tree", keyword="test_decision_tree_classifier_ensemble_")
else:
    print("⏳ Ex 13.25 : les tests attendent le fit de ton DecisionTreeClassifier.")'''

GIVEN_26 = r'''CAL_26 = wb.datasets.load_california()
CAL_FEATURES_26 = [column for column in CAL_26.columns if column != "MedHouseVal"]
rows_26 = np.random.default_rng(1326).choice(len(CAL_26), 3000, replace=False)       # 3,000 districts
X_CAL_26 = as_float32(CAL_26[CAL_FEATURES_26].to_numpy()[rows_26])
Y_CAL_26 = CAL_26["MedHouseVal"].to_numpy()[rows_26]
X_TRAIN_26, X_TEST_26, Y_TRAIN_26, Y_TEST_26 = train_test_split(X_CAL_26, Y_CAL_26, test_size=1000, random_state=1326)'''

CHECK_26 = r'''with checked("13.26"):
    doc_26 = fitted(mylearn.tree.DecisionTreeRegressor(max_depth=1), np.array([[1.0], [2.0], [3.0], [4.0]]),
                    np.array([1.0, 1.2, 3.0, 3.4]))
    print("docstring example:", np.round(np.asarray(doc_26.predict(np.array([[0.0], [5.0]])), dtype=float), 4).tolist(),
          "(expected [1.1, 3.2])")
with checked("13.26"):
    for params in [{"max_depth": 5}, {"min_samples_leaf": 50}]:
        start_26 = time.perf_counter()
        mine_26 = fitted(mylearn.tree.DecisionTreeRegressor(**params), X_TRAIN_26, Y_TRAIN_26)
        seconds_26 = time.perf_counter() - start_26
        oracle_26 = DecisionTreeRegressor(random_state=0, **params).fit(X_TRAIN_26, Y_TRAIN_26)
        same_26 = (mine_26.get_depth() == oracle_26.get_depth() and mine_26.get_n_leaves() == oracle_26.get_n_leaves()
                   and np.allclose(np.ravel(mine_26.predict(X_TEST_26)), oracle_26.predict(X_TEST_26)))
        verdict("13.26", same_26,
                f"{params} : même arbre que scikit-learn ({oracle_26.get_n_leaves()} feuilles, R² de test "
                f"{fr(oracle_26.score(X_TEST_26, Y_TEST_26))}), construit en {fr(seconds_26, 2)} s.",
                f"{params} : scikit-learn a une profondeur de {oracle_26.get_depth()} et {oracle_26.get_n_leaves()} "
                f"feuilles, ton arbre {mine_26.get_depth()} et {mine_26.get_n_leaves()} (ou d'autres prédictions) ; "
                "les tests ci-dessous disent où.")
with checked("13.26"):
    small_26 = fitted(mylearn.tree.DecisionTreeRegressor(max_depth=2), X_TRAIN_26, Y_TRAIN_26)
    print("\n".join(tree_rules(small_26.root_, CAL_FEATURES_26)))
if written(lambda: mylearn.tree.DecisionTreeRegressor().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))):
    run_mylearn_tests("tree", keyword="test_decision_tree_regressor_ and not ensemble")
else:
    print("⏳ Ex 13.26 : les tests attendent le fit de ton DecisionTreeRegressor.")'''

PART_C = Part("C", "Les arbres de décision",
              "D'abord les arbres de scikit-learn : leur instabilité (13.19) et une colonne qui triche (13.21), puis "
              "l'élagage (13.23). En parallèle, tu construis `mylearn/tree.py` pièce par pièce : les impuretés et le "
              "gain d'un split (13.20), la recherche du meilleur split (13.22), l'arbre de classification (13.24), ses "
              "options pour les ensembles du ch. 14 (13.25), puis l'arbre de régression (13.26). Ces cinq exercices 🔨 "
              "forment un tout : ton arbre servira **tel quel** au ch. 14. La cellule suivante définit `tree_rules`, "
              "qui affiche les règles d'un arbre de ta librairie, et `as_float32`, qui arrondit des données comme le "
              "fait scikit-learn (ses arbres travaillent en `float32`), pour comparer les deux librairies à l'identique.",
              given=PART_C_GIVEN, exercises=[
    Ex("13.19", "🔬", 2, 30, "Instabilité et profondeur d'un arbre",
       "mesurer combien un arbre change d'un échantillon à l'autre, et ce que la profondeur y fait.",
       "Ex 13.12 · ch. 9 (overfitting, biais et variance) · fiche §13.5.1 et §13.5.3", thread="synthétique",
       tracks="R, C",
       body=r"""`X_19` : 600 points de deux demi-lunes **très** bruitées. `DRAWS_19` contient 20 tirages de 70 % des points (420 indices chacun). Écris :
- `trees_19(max_depth)` : la liste des 20 arbres `DecisionTreeClassifier(max_depth=max_depth, random_state=0)`, chacun entraîné sur un tirage ;
- `disagreement_19(trees)` : la moyenne, sur toutes les paires d'arbres de la liste, de la part des points de `GRID_19` (30 000 points du plan) où les deux arbres prédisent des classes différentes ;
- a) `leaves_19` : le nombre moyen de feuilles des 20 arbres sans limite ;
- b) `disagree_19` : la liste `[désaccord des arbres sans limite, désaccord des arbres de profondeur 3]` ;
- c) `root_range_19` : l'écart entre le plus grand et le plus petit seuil de la racine des 20 arbres sans limite (`tree_.threshold[0]`) ;
- d) `best_depth_19` : la profondeur de `DEPTHS_19` (de 1 à 12) qui a la meilleure accuracy moyenne en validation croisée à 5 folds (`CV_19`) sur les 600 points.

La vérification contrôle a) à d), puis trace quatre arbres et les courbes d'accuracy (entraînement et validation croisée) selon la profondeur.

Dans tes notes : les racines coupent-elles toujours la même feature ? Pourquoi le seuil varie-t-il autant, et pourquoi toute la suite de l'arbre en dépend-elle ? Pourquoi les arbres de profondeur 3 sont-ils plus stables ? Que disent les deux courbes sur le biais et la variance ?""",
       given=GIVEN_19, todo=TODO_19, check=CHECK_19, solution=SOLUTION_19 + as_printed(CHECK_19, "13.19"),
       record=r'''wb.record("13.19a", leaves_19, decimals=2)
wb.record("13.19b", disagree_19, decimals=4,
          mistakes={"l'ordre demandé est [sans limite, profondeur 3]": disagree_19[::-1]})
wb.record("13.19c", root_range_19, decimals=4)
train_best_19 = DEPTHS_19[int(np.argmax([DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_19, Y_19).score(X_19, Y_19)
                                         for d in DEPTHS_19]))]
others_19d = {f"max_depth = {d} n'a pas la meilleure accuracy moyenne en validation croisée : rends la profondeur du maximum": d
              for d in DEPTHS_19 if d not in (best_depth_19, train_best_19)}
wb.record("13.19d", best_depth_19, choices=DEPTHS_19,
          fractional="un nombre non entier : est-ce une accuracy ? on demande la profondeur, un élément de DEPTHS_19",
          mistakes={"sur l'entraînement, l'arbre le plus profond gagne toujours : choisis la profondeur sur la validation croisée": train_best_19,
                    **others_19d})''',
       note="Les arbres sans limite ont 57 feuilles en moyenne (de 49 à 64) pour 420 points : ils isolent les points "
            "bruités. Deux d'entre eux, entraînés sur deux tirages de 70 % du même jeu, prédisent des classes "
            "différentes sur 10,7 % du plan ; limités à la profondeur 3, sur 6,5 %. Les 20 racines coupent toutes "
            "$x_2$, mais à des seuils qui s'étalent sur plus d'une demi-unité : les gains des seuils voisins sont "
            "presque égaux, et quelques points en plus ou en moins suffisent à changer le gagnant ; tout ce qui suit "
            "dépend de cette première question. L'accuracy d'entraînement monte jusqu'à 1 avec la profondeur ; celle "
            "de validation croisée plafonne vers 0,84-0,85, et sa meilleure valeur (profondeur 5) ne bat les "
            "profondeurs 2 à 12 que de moins de 0,02 : au-delà, la baisse du biais ne compense plus la hausse de la variance."),

    Ex("13.20", "🔨", 2, 25, "Coder les impuretés et le gain d'un split",
       "écrire les trois impuretés et le gain pondéré d'un split, avec des poids d'exemples.",
       "✏️ 13.2 et 13.3 · ch. 6 (entropie en bits) · fiche §13.5.2", thread="—", tracks="M, C",
       mylearn="tree.py",
       body=mylearn_howto("tree.py") + r"""

Écris `gini_impurity`, `entropy_impurity`, `variance_impurity` et `split_gain`. **Contrat** (les docstrings, plus ces précisions que vérifient les tests) :
- les trois impuretés reçoivent `y` (une liste ou un tableau 1-D, des labels quelconques pour les deux premières, des nombres pour la variance) et `sample_weight` (`None` : tous les poids valent 1) et rendent un `float` Python ; les proportions et la moyenne sont **pondérées** : un poids 2 compte comme un exemple répété deux fois, une classe de poids nul ne compte pas (et ne donne pas de `nan` dans l'entropie) ;
- l'entropie est en **bits** ($\log_2$, avec $0 \log 0 = 0$) ; la variance est la moyenne pondérée des carrés des écarts à la moyenne pondérée (ddof = 0) ;
- `split_gain(y, left_mask, criterion="gini", sample_weight=None)` rend $I(\text{parent}) - \frac{w_G}{w} I(G) - \frac{w_D}{w} I(D)$, avec les poids totaux $w_G$, $w_D$ et $w$ (les nombres d'exemples quand les poids valent 1), pour `criterion` dans `"gini"`, `"entropy"`, `"squared_error"` ; si un enfant est vide, le gain vaut `0.0` ;
- erreurs (`ValueError`) : `y` vide ; des poids négatifs, de mauvaise forme ou de somme nulle ; un critère inconnu ; des formes différentes pour `y`, `left_mask` et `sample_weight`.

La vérification essaie les exemples des docstrings, compare l'impureté de la racine et le gain du premier split d'un arbre scikit-learn sur les manchots, puis lance les tests.

Dans tes notes : pourquoi pondérer chaque enfant par sa part des exemples ? Que vaudrait le « gain » d'un split qui envoie un seul exemple à gauche, si l'on oubliait la pondération ?""",
       check=RELOAD + CHECK_20, solution=ref_check(CHECK_20, "13.20"),
       note="Sans pondération, un split qui isole un seul exemple pur dans un enfant fait chuter la moyenne simple des "
            "impuretés (l'une d'elles vaut 0) alors qu'il ne trie presque rien : la pondération mesure la baisse "
            "**moyenne** d'impureté par exemple, et l'enfant d'un seul exemple ne pèse que $1/n$."),

    Ex("13.21", "🐛", 2, 20, "L'arbre parfait qui triche : une colonne identifiant",
       "repérer qu'un modèle trop beau s'appuie sur une colonne qui n'apprend rien du phénomène.",
       "Ex 13.19 · ch. 12 (lecture des fichiers, fuites) · fiche §13.5.3 (pièges : les `feature_importances_`)",
       thread="Penguins", tracks="C",
       body=r"""Un collègue enregistre les 333 manchots dans un fichier CSV, le relit, et entraîne un arbre sur « toutes les colonnes sauf l'espèce ». Son arbre est **parfait** sur le test : 100 %. La saison suivante, les manchots du test sont remesurés et ajoutés au fichier : `NEXT_21`. Mène l'enquête, sans modifier les données du collègue :
a) `top_feature_21` : le nom de la feature la plus importante de `TREE_21` (`feature_importances_`) ;
b) `next_accuracy_21` : l'accuracy de `TREE_21` sur `NEXT_21` ;
c) `clean_accuracy_21` : l'accuracy de test (`TEST_21`) d'un `DecisionTreeClassifier(random_state=0)` entraîné sur `TRAIN_21` **sans** la colonne fautive ;
d) `clean_leaves_21` : le nombre de feuilles de l'arbre de c).

Dans tes notes : d'où vient la colonne fautive, et pourquoi suffit-elle à classer parfaitement ? Combien de feuilles lui faut-il, et pourquoi exactement ce nombre ? Qu'arrive-t-il aux manchots de la saison suivante ? Comment le collègue aurait-il dû écrire et relire son fichier, et quel contrôle aurait révélé le problème avant le test ?""",
       given=GIVEN_21, todo=TODO_21, check=CHECK_21, solution=SOLUTION_21 + as_printed(CHECK_21, "13.21"),
       record=r'''wb.record("13.21a", top_feature_21,
          mistakes={"c'est la feature la plus importante d'un arbre sans la colonne fautive : lis feature_importances_ de TREE_21": MEASURES[int(np.argmax(clean_21.feature_importances_))]})
wb.record("13.21b", next_accuracy_21, decimals=4,
          mistakes={"c'est l'accuracy sur TEST_21 : mesure-la sur NEXT_21": TREE_21.score(TEST_21[FEATURES_21], TEST_21["species"])})
wb.record("13.21c", clean_accuracy_21, decimals=4,
          mistakes={"un arbre parfait : la colonne fautive est-elle bien retirée des features ?": 1.0})
wb.record("13.21d", clean_leaves_21,
          mistakes={"c'est le nombre de feuilles de TREE_21 : compte celles de l'arbre sans la colonne fautive": TREE_21.get_n_leaves()})''',
       note="`to_csv()` écrit l'index par défaut, et `read_csv` le relit comme une colonne sans nom, `Unnamed: 0` : le "
            "numéro de ligne. Le fichier est trié par espèce (146 Adélie, puis 119 Gentoo, puis "
            "68 Chinstrap, après le retrait des lignes incomplètes) : deux seuils sur ce numéro suffisent, d'où un "
            "arbre de **3 feuilles** et une importance de 1 pour cette seule colonne. La saison suivante, les lignes "
            "ajoutées portent des numéros au-delà de 332 : toutes tombent dans la dernière feuille, Chinstrap, et "
            "l'accuracy chute à 0,2, la part des Chinstrap. Sans la colonne, l'arbre fait 0,96 avec 9 feuilles. Il "
            "fallait écrire `to_csv(index=False)` (ou relire avec `index_col=0`), choisir les features par leur nom "
            "plutôt que « tout sauf la cible », et regarder les `feature_importances_` (ou `export_text`) d'un modèle "
            "trop beau pour être vrai."),

    Ex("13.22", "🔨", 3, 45, "Coder la recherche du meilleur split",
       "écrire la recherche gloutonne d'un split, avec les conventions qui la rendent identique à scikit-learn.",
       "Ex 13.20 · ✏️ 13.3 · fiche §13.5.1 et §13.5.2", thread="Penguins", tracks="M, C", mylearn="tree.py",
       body=r"""> **mylearn** : dans `mon_travail/mylearn/tree.py`, mêmes règles qu'en 13.20 (NumPy seulement, scikit-learn pour oracle). Enregistre, puis relance la cellule de vérification.

Écris `best_split`. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `best_split(X, y, criterion="gini", sample_weight=None, feature_indices=None, min_samples_leaf=1, splitter="best", rng=None)` rend `(feature, threshold, gain)` (un `int` et deux `float` Python) ou `None` ; les exemples tels que `X[:, feature] <= threshold` vont à gauche ;
- `splitter="best"` : les seuils candidats d'une feature sont les **milieux** de deux valeurs distinctes consécutives (triées) ; `splitter="random"` (pour les ExtraTrees du ch. 14) : un seul seuil par feature, tiré par `rng.uniform(min, max)` des valeurs du nœud, dans l'ordre de `feature_indices` ; une feature constante est sautée (aucun tirage) ; `rng=None` : `np.random.default_rng()` ;
- `feature_indices` : les features à examiner (`None` : toutes) ; la feature rendue est son indice **dans `X`** ;
- un candidat n'est valide que si chaque enfant reçoit au moins `min_samples_leaf` exemples (comptés sans les poids) ; son gain est celui de `split_gain`, avec les poids ;
- le meilleur gain l'emporte ; les égalités se jugent **à l'arrondi près** (un candidat ne remplace le meilleur que si son gain le dépasse de plus de $10^{-12}$ environ) et vont à la plus petite feature, puis au plus petit seuil, quel que soit l'ordre de `feature_indices` ;
- `None` si aucun candidat valide n'a un gain strictement positif (au-delà de l'arrondi) : le nœud deviendra une feuille ; la fonction ne modifie pas ses arguments ; `ValueError` pour un critère ou un `splitter` inconnu, ou des formes incohérentes.

La vérification essaie l'exemple de la docstring, compare ton premier split des manchots à celui de scikit-learn (Gini et entropie), puis lance les tests.

Dans tes notes : combien de seuils candidats examines-tu pour la longueur des nageoires des 233 manchots ? Pourquoi juger les égalités à l'arrondi près, et pourquoi la plus petite feature ? Pourquoi un gain nul doit-il arrêter ta recherche (le « ou exclusif » de la fiche) ?""",
       check=RELOAD + CHECK_22, solution=ref_check(CHECK_22, "13.22"),
       note="Une feature de $m$ valeurs distinctes donne $m - 1$ seuils candidats : 51 pour les longueurs de nageoires "
            "des 233 manchots d'entraînement (52 valeurs entières distinctes, de 172 à 231 mm). Deux gains égaux en théorie peuvent "
            "différer d'une unité du dernier chiffre selon l'ordre des opérations : sans tolérance, un arrondi "
            "choisirait la feature, et ton arbre différerait de la référence ; la règle « plus petite "
            "feature, puis plus petit seuil » rend le choix reproductible, quel que soit l'ordre de tirage d'une "
            "forêt. Un gain nul ne trie rien : ta librairie s'arrête, comme le dit la fiche (le « ou exclusif » : "
            "aucune question seule ne fait baisser l'impureté)."),

    Ex("13.23", "📦", 3, 35, "Arbres scikit-learn : plot_tree, export_text et élagage ccp_alpha",
       "élaguer un arbre par validation croisée, puis le lire comme une suite de règles.",
       "Ex 13.19 · fiche §13.5.3 (élagage par coût-complexité) et §13.7", thread="Penguins", tracks="R, C",
       body=r"""Les 333 manchots, découpés en 70 % pour l'entraînement et 30 % pour le test, avec les 4 mesures. Calcule :
- `full_23` : un `DecisionTreeClassifier(random_state=0)` sans limite, entraîné sur la partie d'entraînement ;
- a) `size_23` : la liste `[profondeur, nombre de feuilles]` de `full_23` ;
- `alphas_23` : les valeurs de `ccp_alpha` que rend `full_23.cost_complexity_pruning_path(...)`, **sans la dernière** (celle qui réduit l'arbre à sa racine) ;
- b) `best_alpha_23` : la valeur de `alphas_23` qui donne la meilleure accuracy moyenne en validation croisée à 5 folds (`CV_23`) ; à égalité (`np.isclose`), la **plus grande** ;
- `pruned_23` : l'arbre élagué avec cette valeur, entraîné sur toute la partie d'entraînement ;
- c) `pruned_size_23` : la liste `[profondeur, nombre de feuilles]` de `pruned_23` ;
- d) `test_23` : la liste `[accuracy de test de full_23, accuracy de test de pruned_23]`.

La vérification contrôle a) à d), puis dessine l'arbre élagué (`plot_tree`) et affiche ses règles (`export_text`).

Dans tes notes : pourquoi retirer la dernière valeur, et pourquoi préférer la plus grande en cas d'égalité ? Écris en une phrase française la règle qui reconnaît un Gentoo. L'arbre élagué a-t-il « perdu » sur le test ? Que vaut une différence d'un manchot sur 100 ?""",
       given=GIVEN_23, todo=TODO_23, check=CHECK_23, solution=SOLUTION_23 + as_printed(CHECK_23, "13.23"),
       record=r'''wb.record("13.23a", size_23, mistakes={"l'ordre demandé est [profondeur, nombre de feuilles]": size_23[::-1]})
tied_23 = alphas_23[np.isclose(cv_23, cv_23.max())]
wb.record("13.23b", best_alpha_23, decimals=5,
          mistakes={"à égalité, garde la PLUS GRANDE valeur : l'arbre le plus simple": float(tied_23.min())})
wb.record("13.23c", pruned_size_23, mistakes={"l'ordre demandé est [profondeur, nombre de feuilles]": pruned_size_23[::-1]})
wb.record("13.23d", test_23, decimals=4, mistakes={"l'ordre demandé est [arbre complet, arbre élagué]": test_23[::-1]})''',
       note="L'arbre complet a une profondeur de 5 et 12 feuilles. Le chemin d'élagage donne 7 valeurs utiles de "
            "`ccp_alpha` ; la dernière (environ 0,32) réduirait l'arbre à sa seule racine, un modèle qui prédit la "
            "classe majoritaire. Deux valeurs voisines (environ 0,0081 et 0,0082) ont la même accuracy moyenne "
            "(0,957) : on garde la plus grande, qui donne l'arbre le plus simple, selon le principe de parcimonie. "
            "L'arbre élagué n'a plus que 5 feuilles et une profondeur de 3 : « si les nageoires dépassent 206,5 mm et "
            "le bec a moins de 17,65 mm d'épaisseur, c'est un Gentoo ». Il fait 0,96 en test contre 0,97 : un seul "
            "manchot d'écart sur 100, bien moins que l'incertitude d'une accuracy mesurée sur 100 exemples (un "
            "écart-type d'environ 2 points). À accuracy presque égale, on garde l'arbre qu'on peut lire."),

    Ex("13.24", "🔨", 4, 150, "Coder un arbre de décision récursif",
       "construire un arbre CART complet, identique nœud par nœud à celui de scikit-learn.",
       "Ex 13.22 · 0A (fonctions récursives, `@dataclass`) · fiche §13.5 et §13.5.1 (pseudo-code)",
       thread="Penguins", tracks="M, C", mylearn="tree.py",
       body=r"""> **mylearn** : dans `mon_travail/mylearn/tree.py`, mêmes règles qu'en 13.20. Enregistre, puis relance la cellule de vérification. La classe `Node` est fournie complète.

Écris la classe `DecisionTreeClassifier` (sauf les options du 13.25 : ici, `sample_weight=None`, `max_features=None` et `splitter="best"` suffisent). **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__` range ses sept arguments tels quels ; `fit(X, y, sample_weight=None)` vérifie, apprend `classes_` (les labels distincts triés), `n_classes_`, `n_features_in_` et `root_`, puis renvoie `self` ; un second `fit` oublie le premier ; `fit` et `predict` ne modifient pas leurs arguments, et acceptent des listes ;
- un nœud devient une **feuille** s'il est pur, si la profondeur `max_depth` est atteinte, s'il a **moins** de `min_samples_split` exemples, ou si `best_split` ne trouve rien (ta fonction du 13.22, avec `min_samples_leaf`) ; sinon il est coupé, et ses deux enfants se construisent de la même façon ;
- chaque nœud est un `Node` : `feature` et `threshold` (`None` pour une feuille), `left` et `right`, `value` (les proportions des classes dans le nœud, alignées sur `classes_`), `n_samples` (le nombre d'exemples qui l'atteignent) et `impurity` (son impureté pour le critère de l'arbre) ;
- `predict_proba(X)` rend la `value` de la feuille atteinte par chaque exemple ; `predict(X)` la classe de plus grande proportion (à égalité, la première de `classes_`) ; `get_depth()` (0 si la racine est une feuille) ; `get_n_leaves()` ; `score(X, y)` l'accuracy, un `float` Python ;
- erreurs : `ValueError` pour un critère autre que `"gini"` ou `"entropy"`, `max_depth < 1` (`None` est permis), `min_samples_split < 2`, `min_samples_leaf < 1`, un `X` qui n'est pas 2-D ou contient `NaN`, `len(X) != len(y)`, ou un mauvais nombre de colonnes à la prédiction ; `RuntimeError` si `predict`, `predict_proba`, `get_depth` ou `get_n_leaves` est appelée avant `fit`.

La vérification essaie l'exemple de la docstring, compare ton arbre à celui de scikit-learn sur des demi-lunes (trois réglages : même profondeur, même nombre de feuilles, mêmes prédictions sur une grille ; ces réglages évitent les nœuds minuscules, où deux features ont souvent le même gain : scikit-learn départage ces égalités au hasard, ta librairie par le plus petit indice), affiche les règles de ton arbre de profondeur 2 sur les manchots, puis lance les tests (ceux du 13.25 restent pour plus tard).

Dans tes notes : ta construction est-elle récursive ou utilise-t-elle une pile ? Combien de fois appelles-tu `best_split` pour un arbre de $L$ feuilles ? Quelle est la profondeur maximale d'un arbre sur $n$ exemples, et quel risque cela fait-il courir à une fonction récursive en Python ?""",
       check=RELOAD + CHECK_24, solution=ref_check(CHECK_24, "13.24"),
       note="Un arbre de $L$ feuilles a $L - 1$ nœuds internes : `best_split` réussit $L - 1$ fois, et échoue (ou n'est "
            "pas appelée) une fois par feuille. Un arbre sur $n$ exemples peut atteindre une profondeur de $n - 1$ "
            "(un exemple détaché à chaque niveau) ; Python limite la récursion à environ 1 000 appels "
            "(`sys.getrecursionlimit()`) : une construction récursive peut échouer sur un jeu de plusieurs milliers "
            "d'exemples mal séparés, d'où la pile explicite de la référence (même ordre : enfant gauche d'abord)."),

    Ex("13.25", "🔨", 3, 50, "Préparer l'arbre pour les ensembles : poids, features tirées, importances",
       "ajouter à ton arbre les options dont les ensembles du ch. 14 auront besoin.",
       "Ex 13.24 · fiche §13.5.2 et §13.5.3", thread="synthétique", tracks="C", mylearn="tree.py",
       body=r"""> **mylearn** : dans `mon_travail/mylearn/tree.py`, mêmes règles qu'en 13.20. Enregistre, puis relance la cellule de vérification.

Complète `DecisionTreeClassifier` avec ce dont le ch. 14 aura besoin. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `sample_weight` (pour AdaBoost) : des poids positifs ou nuls, de forme `(n,)`, de somme non nulle (`ValueError` sinon) ; ils entrent dans les proportions (`value`), les impuretés et les gains, mais pas dans `n_samples` ni dans les comptes de `min_samples_split` et `min_samples_leaf` ; un poids 2 équivaut à un exemple répété deux fois, multiplier tous les poids par une constante ne change rien, et `fit` ne modifie pas `sample_weight` ;
- `max_features` (pour les forêts) : `None` (toutes les $p$ features), un entier $k$ de 1 à $p$, un réel $f$ dans $]0, 1]$ (`max(1, int(f * p))`), `"sqrt"` (`max(1, int(sqrt(p)))`) ou `"log2"` (`max(1, int(log2(p)))`), `ValueError` sinon ; à **chaque** nœud qu'on tente de couper, on tire $k$ features sans remise avec `rng.choice(p, size=k, replace=False)`, où `rng = np.random.default_rng(random_state)` est créé **une fois** dans `fit` ; on construit l'arbre en profondeur, enfant gauche d'abord ; avec `max_features=None` et `splitter="best"`, rien n'est tiré ;
- `splitter="random"` (pour les ExtraTrees) : transmis à `best_split` avec le même `rng` ; `ValueError` pour une autre valeur que `"best"` ou `"random"` ;
- `feature_importances_` : pour chaque feature, la somme, sur les nœuds coupés selon elle, de $\frac{w_{\text{nœud}}}{w_{\text{racine}}} \times \text{gain}$, puis normalisée pour sommer à 1 (des zéros si la racine est une feuille).

La vérification compare des poids entiers à des exemples répétés, tes importances à celles de scikit-learn, vérifie que `random_state` rend `max_features` reproductible, puis lance les tests.

Dans tes notes : pourquoi créer le générateur dans `fit` plutôt que dans `__init__` ? Sur ces données (deux features utiles, trois de bruit), que valent les importances des features de bruit, et pourquoi ne sont-elles pas nulles ?""",
       check=RELOAD + CHECK_25, solution=ref_check(CHECK_25, "13.25"),
       note="Créé dans `fit`, le générateur repart de la même graine à chaque entraînement : deux `fit` avec le même "
            "`random_state` donnent le même arbre (un générateur créé dans `__init__` continuerait sa suite d'un `fit` "
            "à l'autre). Un arbre de profondeur 5 coupe aussi sur les features de bruit, dans des nœuds profonds "
            "et petits où un seuil de hasard fait baisser l'impureté de quelques exemples : leur importance n'est pas "
            "nulle (quelques pour cent chacune). C'est la limite de cette importance mesurée à l'entraînement : "
            "l'importance par permutation, mesurée sur une validation, la corrige."),

    Ex("13.26", "🔨", 2, 30, "Arbre de régression : même squelette, autre critère",
       "réutiliser la construction de l'arbre pour une cible réelle, avec la variance pour impureté.",
       "Ex 13.24 · ch. 9 (MSE, R²) · fiche §13.5 (arbres de régression)", thread="California", tracks="M, C",
       mylearn="tree.py",
       body=r"""> **mylearn** : dans `mon_travail/mylearn/tree.py`, mêmes règles qu'en 13.20. Enregistre, puis relance la cellule de vérification.

Écris la classe `DecisionTreeRegressor`. **Contrat** (la docstring, plus ces précisions que vérifient les tests) : la même construction que `DecisionTreeClassifier` (mêmes règles d'arrêt, mêmes hyperparamètres, mêmes erreurs), avec ces différences :
- `criterion="squared_error"` seulement (`ValueError` pour toute autre valeur, `"gini"` compris) : l'impureté d'un nœud est sa variance (`variance_impurity`), et une feuille est « pure » quand ses cibles sont toutes égales ;
- la `value` d'un nœud est un tableau de forme `(1,)` : la moyenne (pondérée) des cibles qui l'atteignent ;
- `predict(X)` rend un tableau `float64` de forme `(n,)` ; il n'y a ni `classes_` ni `predict_proba` ; `score(X, y)` rend le $R^2$, un `float` Python.

La vérification essaie l'exemple de la docstring, compare ton arbre à celui de scikit-learn sur 2 000 districts de California (deux réglages), affiche les règles d'un arbre de profondeur 2, puis lance les tests.

Dans tes notes : que prédit ton arbre pour un district dont le revenu médian dépasse tous ceux de l'entraînement ? Pourquoi un arbre de régression sans limite a-t-il un $R^2$ d'entraînement de 1 (ou presque) ?""",
       given=GIVEN_26, check=RELOAD + CHECK_26, solution=ref_check(CHECK_26, "13.26"),
       note="Un arbre de régression ne prédit que des moyennes de feuilles : au-delà du plus grand revenu "
            "d'entraînement, le district tombe dans la feuille la plus à droite et reçoit sa moyenne ; l'arbre "
            "n'extrapole pas (fiche §13.5.1). Sans limite, chaque feuille finit par ne contenir que des cibles égales "
            "(souvent un seul district) : l'arbre recopie l'entraînement, $R^2 = 1$, sauf si deux districts de mêmes "
            "features ont des cibles différentes."),
])

# ---------------------------------------------------------------------------
# Part D: naive Bayes (13.27 to 13.30)
# ---------------------------------------------------------------------------
GIVEN_27 = r'''X_BLOBS_27, Y_BLOBS_27 = wb.synth.make_blobs(n=600, centers=[[0.0, 0.0], [1.5, 0.0]], std=[0.5, 1.5], seed=1327)'''

EXPERIMENT_27 = r'''fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
blobs_27 = GaussianNB().fit(X_BLOBS_27, Y_BLOBS_27)
wb.plot.plot_decision_boundary(blobs_27, X_BLOBS_27, Y_BLOBS_27, ax=axes[0], title="GaussianNB, two blobs")
for n in [50, 200, 5000]:
    accuracies = []
    for r in range(10):                                   # 10 draws: n training points, 2,000 test points
        X, y = wb.synth.make_moons(n=n + 2000, noise=0.2, seed=1327 + r)
        accuracies.append(GaussianNB().fit(X[:n], y[:n]).score(X[n:], y[n:]))
    print(f"moons, {n:4d} training points: mean test accuracy {np.mean(accuracies):.3f} "
          f"(from {min(accuracies):.3f} to {max(accuracies):.3f})")
X_moons_27, Y_moons_27 = wb.synth.make_moons(n=2200, noise=0.2, seed=1327)
wb.plot.plot_decision_boundary(GaussianNB().fit(X_moons_27[:200], Y_moons_27[:200]), X_moons_27[200:700],
                               Y_moons_27[200:700], ax=axes[1], title="GaussianNB, moons (200 training points)")
plt.show()
wb.check("13.27a", surround_27)
wb.check("13.27b", moons_27)
wb.check("13.27c", more_data_27)'''

CHECK_28 = r'''with checked("13.28"):
    doc_28 = mylearn.naive_bayes.log_gaussian_pdf(np.array([0.0, 1.0]), 0.0, 1.0)
    if returned("13.28", "log_gaussian_pdf", doc_28):
        print("docstring example: log_gaussian_pdf([0, 1], 0, 1) =", np.round(np.asarray(doc_28, dtype=float), 4).tolist(),
              "(expected [-0.9189, -1.4189])")
with checked("13.28"):
    lse_28 = mylearn.naive_bayes.logsumexp(np.array([-1000.0, -1000.0]))
    if returned("13.28", "logsumexp", lse_28):
        print("docstring example: logsumexp([-1000, -1000]) =", round(float(lse_28), 4), "(expected -999.3069)")
with checked("13.28"):
    X_train_28, y_train_28 = P_TRAIN[MEASURES].to_numpy(), P_TRAIN["species"].to_numpy()
    X_test_28 = P_TEST[MEASURES].to_numpy()
    mine_28 = fitted(mylearn.naive_bayes.GaussianNB(), X_train_28, y_train_28)
    oracle_28 = GaussianNB().fit(X_train_28, y_train_28)
    proba_28 = mine_28.predict_proba(X_test_28)
    if returned("13.28", "predict_proba", proba_28):
        verdict("13.28", np.allclose(mine_28.theta_, oracle_28.theta_) and np.allclose(mine_28.var_, oracle_28.var_)
                and np.allclose(proba_28, oracle_28.predict_proba(X_test_28)),
                f"manchots : mêmes moyennes, mêmes variances et mêmes posteriors que scikit-learn (accuracy de test "
                f"{fr(oracle_28.score(X_test_28, P_TEST['species']))}).",
                "manchots : tes moyennes, tes variances ou tes posteriors diffèrent de ceux de scikit-learn ; les tests "
                "ci-dessous disent où.")
with checked("13.28"):
    X_mnist_28, y_mnist_28 = wb.datasets.load_mnist("train", n=3000, flatten=True, seed=1328)
    X_mnist_28 = X_mnist_28 / 255.0
    digits_28 = fitted(mylearn.naive_bayes.GaussianNB(), X_mnist_28[:2500], y_mnist_28[:2500])
    joint_28 = np.asarray(digits_28.predict_joint_log_proba(X_mnist_28[2500:]), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        products_28 = np.exp(joint_28)                                    # what multiplying probabilities gives
        naive_28 = products_28 / products_28.sum(axis=1, keepdims=True)
    print(f"MNIST, 784 pixels: joint log-probabilities from {joint_28.min():.3g} to {joint_28.max():.3g}")
    print(f"multiply, then normalise: {int(np.isnan(naive_28).any(axis=1).sum())} of the 500 test images get nan")
    digits_proba_28 = digits_28.predict_proba(X_mnist_28[2500:])
    if returned("13.28", "predict_proba", digits_proba_28):
        digits_proba_28 = np.asarray(digits_proba_28, dtype=float)
        verdict("13.28", np.isfinite(digits_proba_28).all() and np.allclose(digits_proba_28.sum(axis=1), 1),
                f"en log-probabilités avec logsumexp : aucune image sans posterior (accuracy "
                f"{fr(digits_28.score(X_mnist_28[2500:], y_mnist_28[2500:]))}).",
                "tes posteriors contiennent des nan ou ne somment pas à 1 : normalise les log-probabilités jointes "
                "avec logsumexp avant de prendre l'exponentielle.")
if written(lambda: mylearn.naive_bayes.log_gaussian_pdf(0.0, 0.0, 1.0)):
    run_mylearn_tests("naive_bayes", keyword="test_log_gaussian_pdf_ or test_logsumexp_ or test_gaussian_nb_")
else:
    print("⏳ Ex 13.28 : les tests attendent ton log_gaussian_pdf.")'''

GIVEN_29 = r'''def letter_counts(text, size):
    """(n_chunks, 26): the counts of the letters a to z in the consecutive chunks of `size` characters of the text
    (lower-cased; the other characters, accented letters included, are ignored)."""
    text = text.lower()
    n_chunks = len(text) // size
    codes = np.frombuffer(text[:n_chunks * size].encode("utf-32-le"), dtype=np.uint32).reshape(n_chunks, size)
    return np.stack([(codes == ord("a") + j).sum(axis=1) for j in range(26)], axis=1)


HOLMES_29, VERNE_29 = wb.datasets.load_holmes(), wb.datasets.load_verne()


def chunks_29(size):
    """Training and test parts (70/30) of the chunks of `size` characters of Holmes ("en") and Verne ("fr")."""
    english, french = letter_counts(HOLMES_29, size), letter_counts(VERNE_29, size)
    X = np.vstack([english, french])
    y = np.array(["en"] * len(english) + ["fr"] * len(french))
    return train_test_split(X, y, test_size=0.3, random_state=1329, stratify=y)'''

CHECK_29 = r'''with checked("13.29"):
    X_doc_29 = np.array([[2, 1, 0], [3, 0, 1], [0, 2, 3], [1, 0, 4]])
    doc_29 = fitted(mylearn.naive_bayes.MultinomialNB(alpha=1.0), X_doc_29, np.array(["en", "en", "fr", "fr"]))
    print("docstring example: P(feature | class) =", np.round(np.exp(doc_29.feature_log_prob_), 4).tolist(),
          "(expected [[0.6, 0.2, 0.2], [0.1538, 0.2308, 0.6154]])")
with checked("13.29"):
    X_train_29, X_test_29, y_train_29, y_test_29 = chunks_29(20)
    mine_29 = fitted(mylearn.naive_bayes.MultinomialNB(), X_train_29, y_train_29)
    oracle_29 = MultinomialNB(alpha=1.0, force_alpha=True).fit(X_train_29, y_train_29)
    verdict("13.29", np.allclose(mine_29.feature_log_prob_, oracle_29.feature_log_prob_)
            and np.allclose(mine_29.predict_proba(X_test_29), oracle_29.predict_proba(X_test_29)),
            f"morceaux de 20 caractères : mêmes probabilités que scikit-learn ({len(X_train_29)} morceaux "
            "d'entraînement).",
            "morceaux de 20 caractères : tes probabilités diffèrent de celles de scikit-learn ; les tests ci-dessous "
            "disent où.")
    for size in [10, 20, 50, 200]:
        X_tr, X_te, y_tr, y_te = chunks_29(size)
        model = fitted(mylearn.naive_bayes.MultinomialNB(), X_tr, y_tr)
        print(f"chunks of {size:3d} characters: test accuracy {model.score(X_te, y_te):.4f} ({len(X_te)} test chunks)")
if written(lambda: mylearn.naive_bayes.MultinomialNB().fit(np.array([[1, 0], [0, 1]]), np.array([0, 1]))):
    run_mylearn_tests("naive_bayes", keyword="test_multinomial_nb_")
else:
    print("⏳ Ex 13.29 : les tests attendent le fit de ton MultinomialNB.")'''

GIVEN_30 = r'''from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline


def holmes_sentences():
    """The sentences of Holmes, each labelled 1 (dialogue: all its letters between “ and ”) or 0 (narration: none of
    them); the sentences that mix both, or have fewer than 4 words, are left out, and the quotes are removed."""
    text = wb.datasets.load_holmes()
    text = text[text.index("To Sherlock Holmes she is always"):]           # skip the title and the contents
    boundary = re.compile(r"(?:(?<=[.!?])|(?<=[.!?]”))\s+")                # a space after . ! or ?, possibly after ”
    sentences, labels = [], []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.replace("\n", " ").strip()
        if len(paragraph.split()) < 4 or paragraph.isupper():              # headings
            continue
        inside, state = np.zeros(len(paragraph), dtype=bool), False        # a paragraph starts outside the quotes
        for i, char in enumerate(paragraph):
            if char == "“":
                state = True
            inside[i] = state
            if char == "”":
                state = False
        start = 0
        for piece in boundary.split(paragraph):
            start = paragraph.index(piece, start)
            letters = [inside[i] for i in range(start, start + len(piece)) if paragraph[i].isalpha()]
            start += len(piece)
            clean = piece.replace("“", "").replace("”", "").strip()
            if len(clean.split()) >= 4 and letters and (all(letters) or not any(letters)):
                sentences.append(clean)
                labels.append(int(all(letters)))
    return sentences, np.array(labels)


SENTENCES_30, LABELS_30 = holmes_sentences()
S_TRAIN_30, S_TEST_30, Y_TRAIN_30, Y_TEST_30 = train_test_split(SENTENCES_30, LABELS_30, test_size=0.3,
                                                                random_state=1330, stratify=LABELS_30)
print(len(SENTENCES_30), "sentences, for example:")
for label in (1, 0):
    for k in np.flatnonzero(LABELS_30 == label)[:2]:
        print(f"  {label}  {SENTENCES_30[k][:100]}")'''

TODO_30 = r'''dialogue_share_30 = ...  # a) the share of dialogue sentences (label 1) among all of SENTENCES_30
baseline_30 = ...        # b) the test accuracy of a model that always predicts the majority class of the training part
count_nb_30 = ...        # c) the test accuracy of make_pipeline(CountVectorizer(), MultinomialNB()) fitted on S_TRAIN_30
tfidf_nb_30 = ...        # d) the same with TfidfVectorizer() instead of CountVectorizer()
tfidf_lr_30 = ...        # e) the same with TfidfVectorizer() and LogisticRegression(max_iter=1000)
top_dialogue_30 = ...    # f) the 5 words with the largest feature_log_prob_[1] - feature_log_prob_[0] in the pipeline
                         #    of c) (a list of strings, in any order)'''

SOLUTION_30 = r'''dialogue_share_30 = float(LABELS_30.mean())
majority_30 = int(Y_TRAIN_30.mean() > 0.5)
baseline_30 = float(np.mean(Y_TEST_30 == majority_30))
count_pipe_30 = make_pipeline(CountVectorizer(), MultinomialNB()).fit(S_TRAIN_30, Y_TRAIN_30)
count_nb_30 = count_pipe_30.score(S_TEST_30, Y_TEST_30)
tfidf_nb_30 = make_pipeline(TfidfVectorizer(), MultinomialNB()).fit(S_TRAIN_30, Y_TRAIN_30).score(S_TEST_30, Y_TEST_30)
tfidf_lr_30 = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)).fit(S_TRAIN_30, Y_TRAIN_30).score(S_TEST_30, Y_TEST_30)
vocabulary_30 = count_pipe_30[0].get_feature_names_out()
ratio_30 = count_pipe_30[1].feature_log_prob_[1] - count_pipe_30[1].feature_log_prob_[0]
top_dialogue_30 = [str(word) for word in vocabulary_30[np.argsort(ratio_30)[::-1][:5]]]
print("narration words:", [str(word) for word in vocabulary_30[np.argsort(ratio_30)[:10]]])
print("vocabulary:", len(vocabulary_30), "words")
'''

CHECK_30 = r'''wb.check("13.30a", dialogue_share_30)
wb.check("13.30b", baseline_30)
wb.check("13.30c", count_nb_30)
wb.check("13.30d", tfidf_nb_30)
wb.check("13.30e", tfidf_lr_30)
wb.check("13.30f", top_dialogue_30)'''

PART_D = Part("D", "Naïve Bayes",
              "Naïve Bayes gaussien sur deux jeux de points (13.27), puis ta librairie `mylearn/naive_bayes.py` : "
              "`GaussianNB` en log-probabilités (13.28) et `MultinomialNB` avec lissage, sur des lettres de Holmes et "
              "de Verne (13.29). Enfin, un vrai problème de texte : reconnaître les phrases de dialogue de Holmes avec "
              "un sac de mots (13.30).",
              exercises=[
    Ex("13.27", "🔮", 1, 10, "Naïve Bayes sur des nuages gaussiens puis sur des lunes",
       "prévoir la frontière d'un Naïve Bayes gaussien et ce que des données en plus lui apportent.",
       "Ex 13.12 · fiche §13.6 (Naïve Bayes gaussien)", thread="synthétique", tracks="C", hypothesis=True,
       body=r"""Deux expériences avec `GaussianNB()` de scikit-learn. Les **nuages gaussiens** : 300 points de la classe 0 autour de (0, 0), d'écart-type 0,5 ; 300 points de la classe 1 autour de (1,5 ; 0), d'écart-type 1,5 (trois fois plus étalés). Les **lunes** : deux demi-lunes peu bruitées (bruit 0,2), dont on garde $n$ points pour l'entraînement et 2 000 pour le test, 10 tirages par valeur de $n$.

Prédis, **avant** d'exécuter quoi que ce soit :
a) `surround_27` : la frontière sur les nuages gaussiens : `"A"` une droite entre les deux nuages ; `"B"` une courbe fermée autour de la classe 0 ; `"C"` une courbe fermée autour de la classe 1 ; `"D"` deux droites parallèles, la classe 0 entre elles ;
b) `moons_27` : l'accuracy de test moyenne sur les lunes avec 200 points d'entraînement : `"A"` environ 0,5 ; `"B"` environ 0,7 ; `"C"` environ 0,86 ; `"D"` environ 0,97 ;
c) `more_data_27` : passer de 200 à 5 000 points d'entraînement fait-il gagner plus de 3 points d'accuracy sur les lunes ? (`True` ou `False`)

Puis exécute l'expérience.""",
       given=GIVEN_27,
       todo=r'''surround_27 = ...    # a) "A", "B", "C" or "D"
moons_27 = ...       # b) "A", "B", "C" or "D"
more_data_27 = ...   # c) True or False''',
       solution=r'''surround_27, moons_27, more_data_27 = "B", "C", False   # the answers, for the record''',
       after=[("code", guarded(EXPERIMENT_27, ["surround_27", "moons_27", "more_data_27"],
                               "⏳ Ex 13.27 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : compare avec tes prédictions. Sur les nuages gaussiens, pourquoi la frontière a-t-elle "
                     "cette forme ? Sur les lunes, quelles erreurs restent, et que font les données en plus ? Les "
                     "erreurs viennent-elles du biais ou de la variance ?")],
       record=r'''wb.record("13.27a", "B", mistakes={"une droite demanderait deux classes de même variance : ici, l'une est trois fois plus étalée que l'autre": "A",
                                       "la frontière est là où les deux densités sont égales : loin des deux centres, laquelle des deux gaussiennes décroît le plus lentement ?": "C",
                                       "les deux classes diffèrent par leur variance sur les DEUX coordonnées : la frontière peut-elle rester ouverte dans la direction x2 ?": "D"})
wb.record("13.27b", "C", mistakes={"deux demi-lunes peu bruitées : un modèle qui sépare grossièrement les deux nuages fait-il aussi mal que le hasard ?": "A",
                                    "chaque lune ressemble grossièrement à une gaussienne allongée : la frontière quadratique en sépare-t-elle seulement 70 % ?": "B",
                                    "une frontière quadratique peut-elle suivre les deux pointes recourbées des lunes ?": "D"})
wb.record("13.27c", False, mistakes={"Naïve Bayes gaussien n'a que 2 moyennes et 2 variances par classe : ce que la forme gaussienne ne sait pas représenter, des données en plus le lui apprennent-elles ?": True})''',
       note="Sur les nuages gaussiens, Naïve Bayes trace une courbe **fermée autour de la classe 0**, la plus resserrée : loin des "
            "deux centres, la gaussienne large (classe 1) décroît bien moins vite que l'étroite, et l'emporte dans "
            "toutes les directions ; c'est la version 2D des deux seuils de la figure de la fiche. Sur les lunes, "
            "l'accuracy moyenne vaut 0,863 avec 50 points, 0,864 avec 200 et 0,871 avec 5 000 : chaque classe est "
            "résumée par une gaussienne aux axes parallèles aux coordonnées, et la frontière quadratique coupe les "
            "deux pointes recourbées des lunes. Des données en plus estiment mieux les 8 paramètres, mais le modèle "
            "reste le même : c'est une erreur de **biais**, pas de variance."),

    Ex("13.28", "🔨", 3, 40, "Coder GaussianNB avec log-densités et logsumexp",
       "écrire Naïve Bayes gaussien entièrement en log-probabilités, sans jamais multiplier des probabilités.",
       "✏️ 13.7 · ch. 4 (règle de Bayes, logsumexp en 4.18) · ch. 2 (loi normale) · fiche §13.6",
       thread="Penguins", tracks="M, C", mylearn="naive_bayes.py",
       body=mylearn_howto("naive_bayes.py") + r"""

Écris `log_gaussian_pdf`, `logsumexp` et la classe `GaussianNB`. **Contrat** (les docstrings, plus ces précisions que vérifient les tests) :
- `log_gaussian_pdf(x, mean, var)` rend, élément par élément et avec le broadcasting, $-\frac12 \log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}$ ; le résultat reste fini même très loin de la moyenne (les tests l'éprouvent) ; `ValueError` si une variance est négative ou nulle ; la fonction ne modifie pas ses arguments ;
- `logsumexp(a, axis=-1)` rend $\log \sum e^{a}$ le long de l'axe (qui disparaît), sans dépassement ni sous-dépassement, même pour des valeurs autour de ±1 000 ; un `-np.inf` parmi des valeurs finies compte pour 0 dans la somme ; ne modifie pas `a` ;
- `GaussianNB(var_smoothing=1e-9, priors=None)` : `__init__` range ses deux arguments ; `fit(X, y)` apprend `classes_` (triés), `class_count_`, `class_prior_` (les fréquences des classes, ou `priors` s'il est donné), `theta_` (les moyennes par classe) et `var_` (les variances par classe, ddof = 0, **plus** `epsilon_`), avec `epsilon_` = `var_smoothing` × la plus grande variance des features calculée sur tout `X`, et renvoie `self` ; un second `fit` oublie le premier ; `fit` ne modifie pas ses arguments ;
- `predict_joint_log_proba(X)` rend la matrice `(n, K)` des $\log P(c) + \sum_j \log \mathcal{N}(x_j ; \theta_{cj}, \sigma^2_{cj})$ ; `predict_log_proba` la normalise avec **ton** `logsumexp` ; `predict_proba` en prend l'exponentielle ; `predict` rend la classe de plus grand posterior ; `score` l'accuracy, un `float` Python ;
- erreurs : `ValueError` si `X` n'est pas 2-D ou contient `NaN` (comme scikit-learn), si `len(X) != len(y)`, ou si `priors` n'a pas une valeur positive ou nulle par classe, de somme 1 à $10^{-6}$ près ; `RuntimeError` si `predict_joint_log_proba` est appelée avant `fit` (les autres méthodes passent par elle).

La vérification essaie les exemples des docstrings, compare tes moyennes, tes variances et tes posteriors à ceux de scikit-learn sur les manchots, montre ce que donnerait le produit des probabilités sur les 784 pixels de MNIST, puis lance les tests.

Dans tes notes : sur MNIST, pourquoi le produit des probabilités donne-t-il `nan` ? À quoi sert `epsilon_` (pense à un pixel toujours noir dans une classe) ? Pourquoi Naïve Bayes gaussien classe-t-il si mal les chiffres, comparé à un kNN ?""",
       check=RELOAD + CHECK_28, solution=ref_check(CHECK_28, "13.28"),
       note="Sur MNIST, les log-probabilités jointes vont d'environ $-10^{11}$ à $+4\\,560$ : un pixel presque "
            "constant dans une classe a une variance minuscule (de l'ordre de `epsilon_`), si bien que sa densité est "
            "énorme (log-densité très positive) quand le pixel a sa valeur habituelle, et infime dès qu'il s'en "
            "écarte. L'exponentielle déborde vers `inf` pour la meilleure classe et vaut 0 pour les autres : la "
            "normalisation calcule inf/inf = `nan` (0/0 quand tout sous-déborde). logsumexp retire d'abord le "
            "maximum, si bien que la meilleure classe garde $e^0 = 1$. `epsilon_` évite une variance nulle (une division par zéro) pour un pixel toujours noir. Naïve "
            "Bayes classe mal les chiffres (autour de 0,55) : les pixels voisins sont très dépendants, et une "
            "gaussienne par pixel décrit mal des valeurs presque toujours 0 ou 1."),

    Ex("13.29", "🔨", 2, 30, "Coder MultinomialNB avec lissage",
       "écrire Naïve Bayes multinomial pour des comptes, avec le lissage de Laplace.",
       "✏️ 13.6 · Ex 13.28 · fiche §13.6 (lissage)", thread="Holmes/Verne", tracks="C",
       mylearn="naive_bayes.py",
       body=r"""> **mylearn** : dans `mon_travail/mylearn/naive_bayes.py`, mêmes règles qu'en 13.28. Enregistre, puis relance la cellule de vérification.

Écris la classe `MultinomialNB`. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `MultinomialNB(alpha=1.0, fit_prior=True)` : `__init__` range ses deux arguments ; `fit(X, y)` reçoit des **comptes** (positifs ou nuls) et apprend `classes_` (triés), `class_count_`, `feature_count_` (la matrice `(K, d)` des comptes totaux $N_{cj}$ de chaque feature dans chaque classe), `class_log_prior_` (le log des fréquences des classes, ou $-\log K$ partout si `fit_prior=False`) et `feature_log_prob_` $= \log \frac{N_{cj} + \alpha}{N_c + \alpha d}$, puis renvoie `self` ; un second `fit` oublie le premier ; `fit` ne modifie pas ses arguments ;
- `alpha=0` est permis : les fréquences brutes (un compte nul donne $\log 0 = -\infty$) ;
- `predict_joint_log_proba(X)` $=$ `X @ feature_log_prob_.T + class_log_prior_` ; `predict_log_proba`, `predict_proba`, `predict` et `score` comme pour `GaussianNB` (avec ton `logsumexp`) : des documents très longs ne donnent ni `nan` ni 0 partout ;
- erreurs : `ValueError` si `X` a une valeur négative ou `NaN`, n'est pas 2-D, si `len(X) != len(y)`, ou si `alpha < 0` ; `RuntimeError` si `predict_joint_log_proba` est appelée avant `fit`.

Les données : Holmes (anglais, `"en"`) et Verne (français, `"fr"`) découpés en morceaux de quelques caractères ; chaque morceau devient les 26 comptes de ses lettres a à z (`letter_counts`). La vérification essaie l'exemple de la docstring, compare tes probabilités à celles de scikit-learn sur des morceaux de 20 caractères, mesure ton accuracy pour quatre longueurs de morceaux, puis lance les tests.

Dans tes notes : comment l'accuracy varie-t-elle avec la longueur des morceaux, et pourquoi ? Sans lissage, qu'arriverait-il à un morceau anglais qui contient une lettre absente de tous les morceaux français d'entraînement (si le « w », rare en français, n'y apparaissait jamais, par exemple) ?""",
       given=GIVEN_29, check=RELOAD + CHECK_29, solution=ref_check(CHECK_29, "13.29"),
       note="Avec 10 caractères (environ 8 lettres), un morceau sur quatre est mal classé (0,77) ; avec 20, 0,86 ; "
            "avec 50, 0,95 ; avec 200, presque tous (0,999). Chaque lettre apporte un petit indice, et les "
            "log-probabilités s'additionnent : plus il y a de lettres, plus le total sépare les deux langues. Sans "
            "lissage, une lettre absente d'une classe à l'entraînement y reçoit une probabilité nulle : si le « w » "
            "n'apparaissait dans aucun morceau français, un seul « w » rendrait la classe française impossible "
            "($\\log 0 = -\\infty$), quel que soit le reste du morceau. Chez Verne, il apparaît tout de même 248 fois, si "
            "bien qu'ici aucun compte n'est nul."),

    Ex("13.30", "📦", 3, 40, "Dialogue ou narration ? Classer des phrases de Holmes : sac de mots, TF-IDF et Naïve Bayes",
       "classer des textes avec un sac de mots, et lire ce que le modèle a appris.",
       "Ex 13.12 · ch. 6 (fréquences) · ch. 3 (accuracy, classe majoritaire) · fiche §13.6 (Naïve Bayes sur des textes)",
       thread="Holmes/Verne", tracks="R, C",
       body=r"""La cellule suivante découpe les nouvelles de Holmes en phrases et donne à chacune un label : 1 (**dialogue**) si toutes ses lettres sont entre guillemets « “ » et « ” », 0 (**narration**) si aucune ne l'est ; les phrases qui mêlent les deux sont écartées, et les guillemets retirés du texte : le modèle doit reconnaître le dialogue à ses **mots**. 70 % des phrases servent à l'entraînement. Avec `make_pipeline`, calcule :
a) `dialogue_share_30` : la part des phrases de dialogue parmi toutes les phrases ;
b) `baseline_30` : l'accuracy de test d'un modèle qui prédit toujours la classe majoritaire de l'entraînement ;
c) `count_nb_30` : l'accuracy de test de `make_pipeline(CountVectorizer(), MultinomialNB())` entraîné sur `S_TRAIN_30` ;
d) `tfidf_nb_30` : la même chose avec `TfidfVectorizer()` à la place de `CountVectorizer()` ;
e) `tfidf_lr_30` : la même chose avec `TfidfVectorizer()` et `LogisticRegression(max_iter=1000)` ;
f) `top_dialogue_30` : les 5 mots qui ont le plus grand écart `feature_log_prob_[1] - feature_log_prob_[0]` dans le pipeline de c) (une liste de chaînes, dans n'importe quel ordre ; `get_feature_names_out()` donne les mots).

Dans tes notes : pourquoi le dialogue est-il si majoritaire dans Holmes (pense à qui raconte les affaires) ? Que vaut c) comparé à b) ? Pourquoi TF-IDF fait-il **moins bien** avec Naïve Bayes qu'avec la régression logistique (regarde ce que deviennent les « comptes » d'une phrase, et le poids du prior) ? Les mots de f) te surprennent-ils ?""",
       given=GIVEN_30, todo=TODO_30, check=CHECK_30, solution=SOLUTION_30 + as_printed(CHECK_30, "13.30"),
       record=r'''wb.record("13.30a", dialogue_share_30, decimals=4,
          mistakes={"c'est la part de la narration : le label 1 est le dialogue": 1 - dialogue_share_30,
                    "c'est la part dans l'entraînement seulement : on la demande parmi TOUTES les phrases (LABELS_30)": float(Y_TRAIN_30.mean()),
                    "c'est la part dans le test seulement : on la demande parmi TOUTES les phrases (LABELS_30)": float(Y_TEST_30.mean())})
wb.record("13.30b", baseline_30, decimals=4,
          mistakes={"c'est l'accuracy de la classe minoritaire : prédis toujours la classe la plus fréquente": 1 - baseline_30,
                    "c'est l'accuracy de la baseline sur l'entraînement : mesure-la sur Y_TEST_30": float(np.mean(Y_TRAIN_30 == majority_30))})
wb.record("13.30c", count_nb_30, decimals=4,
          mistakes={"c'est l'accuracy d'entraînement : mesure-la sur S_TEST_30": count_pipe_30.score(S_TRAIN_30, Y_TRAIN_30)})
wb.record("13.30d", tfidf_nb_30, decimals=4)
wb.record("13.30e", tfidf_lr_30, decimals=4)
frequent_30 = [str(word) for word in vocabulary_30[np.argsort(count_pipe_30[1].feature_log_prob_[1])[::-1][:5]]]
wb.record("13.30f", top_dialogue_30, ordered=False,
          mistakes={"ce sont les mots les plus caractéristiques de la narration : garde les plus grands écarts, pas les plus petits": [str(word) for word in vocabulary_30[np.argsort(ratio_30)[:5]]],
                    "ce sont les mots les plus fréquents du dialogue, pas les plus caractéristiques : compare les deux classes (l'écart des log-probabilités)": frequent_30})''',
       note="Le dialogue fait 82 % des phrases : chez Conan Doyle, les clients racontent leur affaire entre "
            "guillemets, souvent pendant des pages, et Holmes expose ses déductions de même. Une baseline à la classe "
            "majoritaire fait donc déjà 0,820 ; le sac de mots avec Naïve Bayes monte à 0,897, une erreur presque "
            "divisée par deux. Avec TF-IDF, Naïve Bayes retombe à 0,834 : chaque phrase devient un vecteur normalisé "
            "dont les « comptes » sont des fractions, si bien que la somme $\\sum_j x_j \\log P(j \\mid c)$ ne pèse "
            "presque plus face au log du prior, et le modèle prédit presque toujours la classe majoritaire. La "
            "régression logistique, qui apprend un poids par mot sans hypothèse de comptes, fait 0,888 avec les "
            "mêmes vecteurs. Les mots du dialogue sont ceux de la deuxième personne et du présent (« you », "
            "« your », « will », « are », « can ») ; ceux de la narration décrivent la scène, vue par "
            "Watson."),
])

# ---------------------------------------------------------------------------
# Part E: logistic regression (13.31 and 13.32)
# ---------------------------------------------------------------------------
PART_E_GIVEN = r'''# Adélie or Chinstrap? The 214 penguins of these two species (70 % for training, 30 % for testing)
AC_PENGUINS = PENGUINS[PENGUINS["species"].isin(["Adelie", "Chinstrap"])]
AC_TRAIN, AC_TEST = train_test_split(AC_PENGUINS, test_size=0.3, random_state=1331, stratify=AC_PENGUINS["species"])
print(len(AC_TRAIN), "training penguins,", len(AC_TEST), "test penguins")'''

GIVEN_31 = r'''SCALER_31 = StandardScaler().fit(AC_TRAIN[MEASURES])
Z_TRAIN_31, Z_TEST_31 = SCALER_31.transform(AC_TRAIN[MEASURES]), SCALER_31.transform(AC_TEST[MEASURES])
Y_TRAIN_31, Y_TEST_31 = AC_TRAIN["species"].to_numpy(), AC_TEST["species"].to_numpy()
NEW_31 = pd.DataFrame([[45.0, 17.5, 195.0, 3700.0]], columns=MEASURES)   # a new penguin, raw measures'''

TODO_31 = r'''model_31 = ...   # LogisticRegression() (default settings) fitted on Z_TRAIN_31, Y_TRAIN_31
coef_31 = ...    # a) its 4 coefficients, in the order of MEASURES (a list or a 1-D array)
odds_31 = ...    # b) the factor that multiplies the odds P(Chinstrap) / P(Adelie) when the bill length grows by one
                 #    standard deviation (the other measures unchanged)
proba_31 = ...   # c) the probability that NEW_31 is a Chinstrap
norms_31 = ...   # d) [||coef|| with C = 1, ||coef|| with C = 100], for LogisticRegression(C=C, max_iter=10_000)'''

SOLUTION_31 = r'''model_31 = LogisticRegression().fit(Z_TRAIN_31, Y_TRAIN_31)
coef_31 = model_31.coef_[0]
odds_31 = float(np.exp(coef_31[0]))
proba_31 = float(model_31.predict_proba(SCALER_31.transform(NEW_31))[0, 1])     # classes_ = [Adelie, Chinstrap]
norms_31 = [float(np.linalg.norm(LogisticRegression(C=C, max_iter=10_000).fit(Z_TRAIN_31, Y_TRAIN_31).coef_))
            for C in (1, 100)]
print("classes_:", model_31.classes_.tolist(), "· test accuracy:", model_31.score(Z_TEST_31, Y_TEST_31))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                       # no penalty: the solver may warn
    unpenalized_31 = LogisticRegression(C=np.inf, max_iter=10_000).fit(Z_TRAIN_31, Y_TRAIN_31)
print("C = inf: ||coef|| =", round(float(np.linalg.norm(unpenalized_31.coef_)), 1), "· training accuracy:",
      unpenalized_31.score(Z_TRAIN_31, Y_TRAIN_31))
'''

CHECK_31 = r'''wb.check("13.31a", coef_31)
wb.check("13.31b", odds_31)
wb.check("13.31c", proba_31)
wb.check("13.31d", norms_31)'''

CHECK_32 = r'''if written(lambda: mylearn.logistic.LogisticRegression) and not written(
        lambda: mylearn.info.log_loss(np.array([0, 1]), np.array([0.2, 0.7]))):
    print("⏳ Ex 13.32 : ta fonction log_loss du ch. 6 (mylearn/info.py) n'est pas écrite ; ta LogisticRegression "
          "s'en sert pour son objectif : complète-la d'abord (exercices du ch. 6).")
with checked("13.32"):
    doc_32 = fitted(mylearn.logistic.LogisticRegression(), np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
                    np.array([0, 1, 0, 1, 1]))
    print("docstring example: loss_history_[0] =", round(float(doc_32.loss_history_[0]), 4), "(expected 0.6931) · "
          "predict([[-3], [3]]) =", np.asarray(doc_32.predict(np.array([[-3.0], [3.0]]))).tolist(), "(expected [0, 1])")
with checked("13.32"):
    pair_32 = ["bill_depth_mm", "flipper_length_mm"]       # two measures where the two species overlap
    scaler_32 = StandardScaler().fit(AC_TRAIN[pair_32])
    Z_32, y_32 = scaler_32.transform(AC_TRAIN[pair_32]), AC_TRAIN["species"].to_numpy()
    for alpha, C, label in [(0.0, np.inf, "C=np.inf"), (0.01, 1 / (0.01 * len(Z_32)), "C=1/(alpha*n)")]:
        mine_32 = fitted(mylearn.logistic.LogisticRegression(alpha=alpha, learning_rate=0.5, max_iter=5000, tol=1e-9),
                         Z_32, y_32)
        oracle_32 = LogisticRegression(C=C, tol=1e-10, max_iter=10_000).fit(Z_32, y_32)
        coef_32 = np.asarray(mine_32.coef_, dtype=float)
        print(f"alpha = {alpha}: {mine_32.n_iter_} steps · coef_ {np.round(coef_32, 4).tolist()} (shape {coef_32.shape}), "
              f"intercept_ {float(mine_32.intercept_):.4f} · scikit-learn ({label}): "
              f"{np.round(oracle_32.coef_[0], 4).tolist()}, {oracle_32.intercept_[0]:.4f}")
        verdict("13.32", coef_32.shape == (2,) and np.allclose(coef_32, oracle_32.coef_[0], atol=1e-3)
                and np.isclose(float(mine_32.intercept_), oracle_32.intercept_[0], atol=1e-3),
                f"alpha = {alpha} : mêmes poids et même ordonnée à l'origine que scikit-learn avec {label}.",
                f"alpha = {alpha} : tes poids ou ton ordonnée à l'origine diffèrent de ceux de scikit-learn avec {label} ; "
                "les tests ci-dessous disent où.")
    default_32 = LogisticRegression().fit(Z_32, y_32)
    print("scikit-learn with its default settings (C=1.0):", np.round(default_32.coef_[0], 4).tolist())
with checked("13.32"):
    histories_32 = {rate: list(fitted(mylearn.logistic.LogisticRegression(learning_rate=rate, max_iter=300, tol=0.0),
                                      Z_32, y_32).loss_history_) for rate in [0.05, 0.5, 5.0]}
    for rate, history_32 in histories_32.items():
        plt.plot(history_32, label=f"learning_rate = {rate}")
    plt.xlabel("gradient step")
    plt.ylabel("objective J(w, b)")
    plt.legend()
    plt.show()
if written(lambda: mylearn.logistic.LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))):
    run_mylearn_tests("logistic", keyword="test_logistic_regression_")
else:
    print("⏳ Ex 13.32 : les tests attendent le fit de ta LogisticRegression.")'''

PART_E = Part("E", "La régression logistique",
              "Le classifieur paramétrique de référence, absent du livre à ce chapitre : celui de scikit-learn, ses "
              "coefficients et ses probabilités (13.31), puis le tien, dans `mylearn/logistic.py` (13.32). Les deux "
              "exercices séparent les Adélie des Chinstrap ; la cellule suivante garde ces 214 manchots et les "
              "découpe (70/30).",
              given=PART_E_GIVEN, exercises=[
    Ex("13.31", "📦", 2, 25, "Régression logistique scikit-learn : coefficients et probabilités",
       "lire les coefficients d'une régression logistique en odds, et voir l'effet de sa pénalité par défaut.",
       "Ex 13.12 · ch. 12 (standardisation) · fiche « Au-delà du livre : la régression logistique »",
       thread="Penguins", tracks="R, C",
       body=r"""Les 4 mesures des manchots Adélie et Chinstrap, standardisées sur l'entraînement (`Z_TRAIN_31`). Avec `LogisticRegression` de scikit-learn, calcule :
- `model_31` : `LogisticRegression()`, avec ses réglages par défaut, entraîné sur `Z_TRAIN_31`, `Y_TRAIN_31` ;
- a) `coef_31` : ses 4 coefficients, dans l'ordre de `MEASURES` ;
- b) `odds_31` : le facteur qui multiplie les **odds** $P(\text{Chinstrap}) / P(\text{Adélie})$ quand la longueur du bec augmente d'un écart-type, les autres mesures restant fixes ;
- c) `proba_31` : la probabilité que le nouveau manchot `NEW_31` (mesures brutes : bec de 45 mm de long et 17,5 mm d'épaisseur, nageoires de 195 mm, 3 700 g) soit un Chinstrap ;
- d) `norms_31` : la liste `[‖coef‖ avec C = 1, ‖coef‖ avec C = 100]` de `LogisticRegression(C=C, max_iter=10_000)`.

Dans tes notes : quelle mesure compte le plus, et le signe de chaque coefficient te paraît-il logique (compare les moyennes des deux espèces) ? Que fait C, et que se passe-t-il avec `C=np.inf` sur ces données (essaie : regarde la norme et l'accuracy d'entraînement) ? Pourquoi standardiser avant de comparer des coefficients ?""",
       given=GIVEN_31, todo=TODO_31, check=CHECK_31, solution=SOLUTION_31 + as_printed(CHECK_31, "13.31"),
       record=r'''wb.record("13.31a", coef_31, decimals=3)
wb.record("13.31b", odds_31, decimals=4,
          mistakes={"c'est le coefficient lui-même : il s'ajoute au log des odds, et multiplie donc les odds par son exponentielle": float(coef_31[0]),
                    "c'est le facteur des odds de l'Adélie : classes_ est trié, et la classe 1 est le Chinstrap": float(np.exp(-coef_31[0]))})
raw_31 = float(model_31.predict_proba(NEW_31.to_numpy())[0, 1])
wb.record("13.31c", proba_31, decimals=4,
          mistakes={"c'est la probabilité de l'Adélie : la colonne 1 de predict_proba est classes_[1], le Chinstrap": 1 - proba_31,
                    "NEW_31 contient des mesures brutes : le modèle attend des mesures standardisées par SCALER_31": raw_31})
wb.record("13.31d", norms_31, decimals=4, mistakes={"l'ordre demandé est [C = 1, C = 100]": norms_31[::-1]})''',
       note="La longueur du bec domine (coefficient d'environ 3,9) : les Chinstrap ont un bec bien plus long que les "
            "Adélie ; un écart-type de bec en plus multiplie les odds du Chinstrap par environ 49. Les autres "
            "coefficients sont négatifs et plus petits : à longueur de bec égale, un manchot plus lourd ou au bec plus "
            "épais est plutôt un Adélie, et les nageoires ne comptent presque pas. Ce sont des effets **à bec égal** : "
            "en moyenne, les Chinstrap sont même un peu plus lourds (3 733 g contre 3 706 g) ; ces mesures sont "
            "corrélées à la longueur du bec, et leur coefficient ne dit que ce qu'elles ajoutent à celle-ci. Le nouveau manchot, au bec long "
            "mais assez lourd, est un Chinstrap à 72 %. C est l'inverse de la force de la pénalité : avec $C = 100$, "
            "la norme des coefficients est multipliée par 3,6 environ. Sur ces données, les deux "
            "espèces sont **séparables** par un hyperplan des 4 mesures : sans pénalité (`C=np.inf`), la log-loss "
            "baisse toujours quand $\\lVert\\mathbf{w}\\rVert$ grandit, et le solveur s'arrête seulement à sa tolérance, "
            "avec une norme de l'ordre de 100 et une accuracy d'entraînement de 1. Standardiser met les coefficients "
            "dans la même unité (un écart-type) : sans cela, celui de la masse (en grammes) serait minuscule sans être "
            "moins important."),

    Ex("13.32", "🔨", 3, 70, "Coder la régression logistique par descente de gradient",
       "écrire la régression logistique et sa descente de gradient, et retrouver exactement scikit-learn.",
       "∂ 13.8 · ch. 5 (descente de gradient) · ch. 6 (log-loss, `info.log_loss`) · ch. 12 (standardisation) · "
       "fiche « Au-delà du livre : la régression logistique »",
       thread="Penguins", tracks="M, C", mylearn="logistic.py",
       body=mylearn_howto("logistic.py") + r"""

Écris la classe `LogisticRegression` (`sigmoid` est fournie). **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__(alpha=0.0, learning_rate=0.1, max_iter=1000, tol=1e-6, fit_intercept=True)` range ses arguments tels quels ;
- `fit(X, y)` code les labels par `classes_` (les deux labels distincts triés : `classes_[0]` → 0, `classes_[1]` → 1), part de $\mathbf{w} = \mathbf{0}$ et $b = 0$, et minimise $J(\mathbf{w}, b) =$ `log_loss(y, p)` $+ \frac{\alpha}{2}\lVert\mathbf{w}\rVert^2$, où `log_loss` est **ta** fonction du ch. 6, déjà importée par le fichier (`from .info import log_loss`) et $b$ n'est pas pénalisé ;
- `loss_history_` commence par $J$ au départ ($\ln 2$) ; chaque itération calcule les deux gradients au point courant ($\frac1n \mathbf{X}^\top(\mathbf{p} - \mathbf{y}) + \alpha\mathbf{w}$ et la moyenne de $\mathbf{p} - \mathbf{y}$), fait un pas sur $\mathbf{w}$ et $b$ ensemble, puis ajoute le nouveau $J$ ; l'entraînement s'arrête **après** le pas où $J$ a bougé de moins de `tol` (en valeur absolue), ou après `max_iter` pas ; `n_iter_` est le nombre de pas faits, et `len(loss_history_) == n_iter_ + 1` ; avec `fit_intercept=False`, $b$ reste à 0 ;
- après `fit` : `coef_` (forme `(d,)`, pas `(1, d)` comme scikit-learn), `intercept_` (un `float`), `n_iter_` (un `int`), `loss_history_` (une liste de `float`) ; `fit` renvoie `self`, oublie un `fit` précédent et ne modifie pas ses arguments ;
- `decision_function(X)` rend $\mathbf{X}\mathbf{w} + b$ ; `predict_proba(X)` la matrice `(n, 2)` $[1 - p, p]$ ; `predict(X)` rend `classes_[1]` si le score est **strictement** positif, `classes_[0]` sinon ; `score` l'accuracy, un `float` Python ;
- erreurs : `ValueError` si `y` n'a pas exactement deux classes, si `learning_rate <= 0`, `alpha < 0` ou `max_iter < 1`, si `X` n'est pas 2-D, contient `NaN` (comme scikit-learn) ou si `len(X) != len(y)` ; `RuntimeError` avant `fit`.

La vérification essaie l'exemple de la docstring, compare tes poids à ceux de scikit-learn sur deux mesures des manchots, sans pénalité (`C=np.inf`) puis avec $\alpha = 0{,}01$ ($C = 1/(\alpha n)$), trace l'objectif au fil des pas pour trois learning rates, puis lance les tests.

Dans tes notes : pourquoi la pénalité de scikit-learn s'écrit-elle $C = 1/(\alpha n)$ ? Pourquoi les poids de `LogisticRegression()` par défaut sont-ils plus petits que les tiens sans pénalité ? Que montrent les trois courbes ?""",
       check=RELOAD + CHECK_32, solution=ref_check(CHECK_32, "13.32"),
       note="scikit-learn minimise $\\frac12\\lVert\\mathbf{w}\\rVert^2 + C \\sum_i \\ell_i$, une **somme** de "
            "log-loss ; ta librairie minimise la **moyenne** plus $\\frac{\\alpha}{2}\\lVert\\mathbf{w}\\rVert^2$. En "
            "divisant la première par $Cn$, on retrouve la seconde avec $\\alpha = 1/(Cn)$, d'où $C = 1/(\\alpha n)$. "
            "`LogisticRegression()` a `C=1.0` par défaut, une pénalité de force $\\alpha = 1/n$ : ses poids sont plus "
            "petits que ceux d'une descente sans pénalité (l'annexe des erreurs fréquentes le signale). Avec un "
            "learning rate de 0,05, l'objectif descend lentement et n'a pas fini après 300 pas ; avec 0,5, il atteint "
            "son minimum en 70 pas environ ; avec 5, en quelques pas : sur des données standardisées, la "
            "courbure de la log-loss reste faible, et un grand pas reste stable."),
])

# ---------------------------------------------------------------------------
# Part F: synthesis (13.33 to 13.36)
# ---------------------------------------------------------------------------
PART_F_GIVEN = r'''from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline'''

GIVEN_33 = r'''X_33, Y_33 = wb.synth.make_moons(n=300, noise=0.3, seed=1333)
CLASSIFIERS_33 = {"A": KNeighborsClassifier(n_neighbors=1), "B": DecisionTreeClassifier(max_depth=6, random_state=0),
                  "C": GaussianNB(), "D": LogisticRegression()}
ORDER_33 = "".join(np.random.default_rng(1302).permutation(list(CLASSIFIERS_33)))   # the order of the panels
fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
for panel, (ax, letter) in enumerate(zip(axes, ORDER_33), start=1):
    model = CLASSIFIERS_33[letter].fit(X_33, Y_33)
    wb.plot.plot_decision_boundary(model, X_33, Y_33, ax=ax, title=f"panel {panel}")
plt.show()
del ORDER_33                         # the answer is not kept in the notebook'''

TODO_33 = r'''panels_33 = ...    # a) the letters of the classifiers of panels 1, 2, 3 and 4, in this order (e.g. "ABCD")
perfect_33 = ...   # b) the number (1 to 4) of the panel whose classifier classifies ALL its training points right'''

CHECK_33 = r'''wb.check("13.33a", panels_33)
wb.check("13.33b", perfect_33)'''

GIVEN_34 = r'''TRAIN_34, Y_TRAIN_34 = wb.datasets.load_mnist("train", n=2000, flatten=True, seed=1334)
TEST_34, Y_TEST_34 = wb.datasets.load_mnist("test", n=500, flatten=True, seed=1334)
TRAIN_34, TEST_34 = TRAIN_34 / 255.0, TEST_34 / 255.0     # pixels in [0, 1], float64
print(TRAIN_34.shape, TEST_34.shape)'''

TODO_34 = r'''def distances_loop_34(A, B):
    """Euclidean distance matrix (len(A), len(B)), with two Python loops over the rows (NumPy for one pair)."""
    raise NotImplementedError("distances_loop_34")


def distances_broadcast_34(A, B):
    """The same, with one broadcast difference A[:, None, :] - B[None, :, :] and no Python loop."""
    raise NotImplementedError("distances_broadcast_34")


def distances_dot_34(A, B):
    """The same, with ||a - b||² = ||a||² - 2 a.b + ||b||² (one matrix product for all the a.b), never nan."""
    raise NotImplementedError("distances_dot_34")


def timed_34(function, *args, repeat=3):
    """The smallest of `repeat` durations of function(*args), in seconds (time.perf_counter)."""
    raise NotImplementedError("timed_34")'''

SOLUTION_34 = r'''def distances_loop_34(A, B):
    """Euclidean distance matrix (len(A), len(B)), with two Python loops over the rows (NumPy for one pair)."""
    D = np.empty((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            D[i, j] = np.sqrt(np.sum((A[i] - B[j]) ** 2))
    return D


def distances_broadcast_34(A, B):
    """The same, with one broadcast difference A[:, None, :] - B[None, :, :] and no Python loop."""
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def distances_dot_34(A, B):
    """The same, with ||a - b||² = ||a||² - 2 a.b + ||b||² (one matrix product for all the a.b), never nan."""
    squared = (A ** 2).sum(axis=1)[:, None] - 2 * A @ B.T + (B ** 2).sum(axis=1)[None, :]
    return np.sqrt(np.maximum(squared, 0.0))           # rounding can give -1e-12: no sqrt of a negative number


def timed_34(function, *args, repeat=3):
    """The smallest of `repeat` durations of function(*args), in seconds (time.perf_counter)."""
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        function(*args)
        durations.append(time.perf_counter() - start)
    return min(durations)
'''

CHECK_34 = r'''def as_matrix_34(name, D, shape):
    """The distance matrix returned by `name`, or None with a ❌ line when it is None or has the wrong shape."""
    if not returned("13.34", name, D):
        return None
    try:
        D = np.asarray(D, dtype=float)
    except (TypeError, ValueError):
        print(f"❌ Ex 13.34 : {name} doit rendre un tableau de nombres, pas un {type(D).__name__}.")
        return None
    if D.shape != shape:
        print(f"❌ Ex 13.34 : {name} doit rendre une matrice de forme {shape} (une ligne par ligne de A, une colonne "
              f"par ligne de B), pas {D.shape}.")
        return None
    return D


with checked("13.34"):
    A_34, B_34 = TEST_34[:10], TRAIN_34[:200]
    exact_34 = cdist(A_34, B_34)
    for name_34, function_34 in [("distances_loop_34", distances_loop_34), ("distances_broadcast_34", distances_broadcast_34),
                                 ("distances_dot_34", distances_dot_34)]:
        try:
            D_34 = as_matrix_34(name_34, function_34(A_34, B_34), (10, 200))
        except NotImplementedError:
            print(f"⏳ Ex 13.34 : {name_34} n'est pas encore écrite.")
            continue
        if D_34 is not None:
            verdict("13.34", np.allclose(D_34, exact_34), f"{name_34} rend les distances de scipy (cdist).",
                    f"{name_34}(TEST_34[:10], TRAIN_34[:200]) a la bonne forme, mais pas les distances de cdist "
                    f"(par exemple {D_34[0, 0]:.4f} au lieu de {exact_34[0, 0]:.4f} pour la première case) : la racine "
                    "de la somme des carrés des différences ?")
with checked("13.34"):
    calls_34 = []
    def slow_first_34():                                  # 0.1 s at the first call, then 0.01 s
        time.sleep(0.1 if not calls_34 else 0.01)
        calls_34.append(1)
    probe_34 = timed_34(slow_first_34, repeat=3)
    if returned("13.34", "timed_34", probe_34):
        if not isinstance(probe_34, (int, float, np.floating)):
            print(f"❌ Ex 13.34 : timed_34 doit rendre UNE durée (un nombre de secondes), pas un {type(probe_34).__name__}.")
        else:
            verdict("13.34", 0.009 < probe_34 < 0.02,
                    "timed_34 rend bien la plus petite des durées, en secondes.",
                    f"timed_34 rend {probe_34!r} pour trois appels de 0,1 s, 0,01 s et 0,01 s : la plus petite durée, "
                    "en secondes (environ 0,01), était attendue (pas la moyenne, ni la somme, ni des millisecondes).")
with checked("13.34"):
    pairs_full_34 = len(TEST_34) * len(TRAIN_34)               # 500 x 2,000 distances
    rows_34 = []
    for name_34, function_34, a_34, b_34 in [("loop", distances_loop_34, TEST_34[:10], TRAIN_34[:200]),
                                              ("broadcast", distances_broadcast_34, TEST_34[:50], TRAIN_34[:500]),
                                              ("dot product", distances_dot_34, TEST_34, TRAIN_34)]:
        seconds_34 = float(timed_34(function_34, a_34, b_34))
        rows_34.append((name_34, f"{len(a_34)} x {len(b_34)}", seconds_34, seconds_34 * pairs_full_34 / (len(a_34) * len(b_34))))
    fastest_34 = min(row[3] for row in rows_34)
    print(f"{'version':12s} {'measured on':12s} {'time (s)':>9s} {'500 x 2000 (s)':>15s} {'x fastest':>10s}")
    for name_34, size_34, seconds_34, full_34 in rows_34:
        print(f"{name_34:12s} {size_34:12s} {seconds_34:9.4f} {full_34:15.3f} {full_34 / fastest_34:10.0f}")
    print(f"memory of the broadcast difference for 500 x 2000 images: {500 * 2000 * 784 * 8 / 1e9:.1f} GB")
with checked("13.34"):
    with np.errstate(invalid="ignore"):
        squared_34 = ((TRAIN_34 ** 2).sum(axis=1)[:, None] - 2 * TRAIN_34 @ TRAIN_34.T
                      + (TRAIN_34 ** 2).sum(axis=1)[None, :])
        naive_nan_34 = int(np.isnan(np.sqrt(squared_34)).sum())
        mine_34 = as_matrix_34("distances_dot_34", distances_dot_34(TRAIN_34, TRAIN_34), (2000, 2000))
    if mine_34 is not None:
        mine_nan_34 = int(np.isnan(mine_34).sum())
        print(f"distances between the 2,000 training images: {naive_nan_34} nan with a plain sqrt of the identity, "
              f"{mine_nan_34} with distances_dot_34")
        verdict("13.34", mine_nan_34 == 0, "distances_dot_34 ne rend jamais nan.",
                "distances_dot_34 rend des nan : l'arrondi donne des carrés légèrement négatifs, dont la racine "
                "n'existe pas.")
        nearest_34 = np.argmin(as_matrix_34("distances_dot_34", distances_dot_34(TEST_34, TRAIN_34), (500, 2000)), axis=1)
        accuracy_34 = float(np.mean(Y_TRAIN_34[nearest_34] == Y_TEST_34))
        oracle_34 = KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(TRAIN_34, Y_TRAIN_34).score(TEST_34, Y_TEST_34)
        verdict("13.34", np.isclose(accuracy_34, oracle_34),
                f"le plus proche voisin calculé avec tes distances fait {fr(accuracy_34)}, comme scikit-learn.",
                f"le plus proche voisin calculé avec tes distances fait {fr(accuracy_34)}, scikit-learn {fr(oracle_34)}.")'''

GIVEN_35 = r'''X_RAW_TRAIN_35, Y_TRAIN_35 = wb.datasets.load_mnist("train", n=500, flatten=True, seed=1335)
X_RAW_TEST_35, Y_TEST_35 = wb.datasets.load_mnist("test", n=500, flatten=True, seed=1335)
PCA_35 = PCA(n_components=50, random_state=0).fit(X_RAW_TRAIN_35 / 255.0)
Z_TRAIN_35, Z_TEST_35 = PCA_35.transform(X_RAW_TRAIN_35 / 255.0), PCA_35.transform(X_RAW_TEST_35 / 255.0)
print(Z_TRAIN_35.shape, Z_TEST_35.shape, "(500 images, 50 PCA coordinates)")'''

TODO_35 = r'''def timed_fit_score_35(model, X_train, y_train, X_test, y_test):
    """Fit `model` on the training part, then score it on the test part. Return the tuple (accuracy, fit time in
    seconds, scoring time in seconds), both times measured with time.perf_counter."""
    raise NotImplementedError("timed_fit_score_35")'''

SOLUTION_35 = r'''def timed_fit_score_35(model, X_train, y_train, X_test, y_test):
    """Fit `model` on the training part, then score it on the test part. Return the tuple (accuracy, fit time in
    seconds, scoring time in seconds), both times measured with time.perf_counter."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    middle = time.perf_counter()
    accuracy = model.score(X_test, y_test)
    return float(accuracy), middle - start, time.perf_counter() - middle
'''

CHECK_35 = r'''def measured_35(result):
    """The tuple (accuracy, fit seconds, score seconds) returned by timed_fit_score_35, or None with a ❌ line."""
    if not returned("13.35", "timed_fit_score_35", result):
        return None
    if not isinstance(result, tuple) or len(result) != 3:
        print("❌ Ex 13.35 : timed_fit_score_35 doit rendre un tuple de trois nombres (accuracy, durée du fit, durée "
              f"du score), pas {type(result).__name__} {result!r}.")
        return None
    accuracy, fit_seconds, score_seconds = (float(value) for value in result)
    if not 0 <= accuracy <= 1:
        print(f"❌ Ex 13.35 : une accuracy de {accuracy} : elle est entre 0 et 1, en première position du tuple.")
        return None
    return accuracy, fit_seconds, score_seconds


pairs_35 = [("kNN (k = 3)", "13.14", lambda: mylearn.neighbors.KNeighborsClassifier(n_neighbors=3),
             lambda: KNeighborsClassifier(n_neighbors=3, algorithm="brute")),
            ("GaussianNB", "13.28", lambda: mylearn.naive_bayes.GaussianNB(), lambda: GaussianNB()),
            ("tree", "13.24", lambda: mylearn.tree.DecisionTreeClassifier(), lambda: DecisionTreeClassifier(random_state=0))]
results_35 = {}
with checked("13.35"):
    print(f"{'classifier':14s} {'library':13s} {'accuracy':>9s} {'fit (s)':>9s} {'score (s)':>10s}")
    for name_35, exercise_35, mine_35, oracle_35 in pairs_35:
        if not written(lambda: mine_35().fit(Z_TRAIN_35[:5], Y_TRAIN_35[:5])):
            print(f"⏳ {name_35} : ta classe ({exercise_35}) n'est pas encore écrite.")
            continue
        for library_35, model_35 in [("mylearn", mine_35()), ("scikit-learn", oracle_35())]:
            try:
                result_35 = measured_35(timed_fit_score_35(model_35, Z_TRAIN_35, Y_TRAIN_35, Z_TEST_35, Y_TEST_35))
            except AttributeError as exc:
                if "NoneType" not in str(exc):
                    raise
                print(f"❌ Ex 13.35 : {name_35} ({library_35}) : {exc} : n'enchaîne pas fit(...).score(...) (un fit "
                      "sans return self rend None), appelle fit, puis score.")
                result_35 = None
            if result_35 is None:
                break
            results_35[name_35, library_35] = result_35
            print(f"{name_35:14s} {library_35:13s} {result_35[0]:9.3f} {result_35[1]:9.3f} {result_35[2]:10.3f}")
if ("kNN (k = 3)", "mylearn") in results_35:
    wb.check("13.35a", results_35["kNN (k = 3)", "mylearn"][0], computed=True)
if ("GaussianNB", "mylearn") in results_35:
    wb.check("13.35b", results_35["GaussianNB", "mylearn"][0], computed=True)
if ("tree", "mylearn") in results_35 and ("tree", "scikit-learn") in results_35:
    tree_gap_35 = abs(results_35["tree", "mylearn"][0] - results_35["tree", "scikit-learn"][0])
    verdict("13.35", tree_gap_35 <= 0.03,
            "ton arbre a l'accuracy de celui de scikit-learn, à 3 points près (les égalités de gain se départagent "
            "autrement).",
            f"ton arbre s'écarte de {fr(tree_gap_35)} de celui de scikit-learn : plus de 3 points, ce n'est pas qu'une "
            "affaire d'égalités ; relance les tests de 13.24.")'''

GIVEN_36 = r'''X_TEST_36, Y_TEST_36 = wb.datasets.load_mnist("test", flatten=True)
X_TEST_36, Y_TEST_36 = X_TEST_36[:2000] / 255.0, Y_TEST_36[:2000]       # the first 2,000 test images, never for tuning
print(X_TEST_36.shape)'''

TODO_36 = r'''N_TRAIN_36 = ...   # the number of training images you use (at most 20,000)


def build_model_36():
    """Return a NEW, unfitted scikit-learn classifier (often a pipeline) that takes pixels in [0, 1], shape (n, 784).
    No neural network: a classifier of this chapter, with any preprocessing of chapter 12."""
    raise NotImplementedError("build_model_36")'''

SOLUTION_36 = r'''N_TRAIN_36 = 20_000


def build_model_36():
    """Return a NEW, unfitted scikit-learn classifier (often a pipeline) that takes pixels in [0, 1], shape (n, 784).
    No neural network: a classifier of this chapter, with any preprocessing of chapter 12."""
    return make_pipeline(PCA(n_components=50, random_state=0), SVC(C=10))   # chosen by cross-validation (05_solutions.md)
'''

CHECK_36 = r'''with wb.attempt("13.36"):
    if not filled(N_TRAIN_36):
        raise NotImplementedError("N_TRAIN_36")
    if not 1 <= int(N_TRAIN_36) <= 20_000:
        print(f"❌ Ex 13.36 : {N_TRAIN_36} images d'entraînement : le défi en demande de 1 à 20 000.")
    else:
        X_train_36, y_train_36 = wb.datasets.load_mnist("train", n=int(N_TRAIN_36), flatten=True, seed=1336)
        model_36 = build_model_36()
        if returned("13.36", "build_model_36", model_36):
            start_36 = time.perf_counter()
            model_36.fit(X_train_36 / 255.0, y_train_36)
            fit_36 = time.perf_counter() - start_36
            start_36 = time.perf_counter()
            accuracy_36 = model_36.score(X_TEST_36, Y_TEST_36)
            print(f"{int(N_TRAIN_36)} training images · fit {fit_36:.1f} s · prediction of 2,000 images "
                  f"{time.perf_counter() - start_36:.1f} s · test accuracy {accuracy_36:.4f}")
            verdict("13.36", accuracy_36 >= 0.96, f"{fr(100 * accuracy_36, 2)} % : défi réussi.",
                    f"{fr(100 * accuracy_36, 2)} % : il manque {fr(100 * (0.96 - accuracy_36), 2)} point(s) ; règle ton "
                    "modèle par validation croisée sur l'entraînement, jamais sur ces 2 000 images.")'''

PART_F = Part("F", "Synthèse : lire, mesurer, comparer",
              "Reconnaître un classifieur à sa frontière (13.33), mesurer et accélérer le calcul des distances d'un "
              "kNN (13.34), comparer ta librairie à scikit-learn en accuracy et en temps (13.35), puis le défi : 96 % "
              "sur MNIST avec un classifieur classique (13.36).",
              given=PART_F_GIVEN, exercises=[
    Ex("13.33", "📈", 1, 15, "Associer quatre frontières à leurs classifieurs",
       "reconnaître un classifieur à la forme de sa frontière.",
       "Ex 13.12 · fiche §13.7 (tableau « frontière »)", thread="synthétique", tracks="R",
       body=r"""La cellule suivante entraîne quatre classifieurs sur les mêmes 300 points de deux demi-lunes bruitées, et trace leurs frontières dans un ordre mélangé (panneaux 1 à 4). Les classifieurs : A `KNeighborsClassifier(n_neighbors=1)` ; B `DecisionTreeClassifier(max_depth=6)` ; C `GaussianNB()` ; D `LogisticRegression()`. Réponds sans réentraîner les modèles :
a) `panels_33` : les lettres des classifieurs des panneaux 1, 2, 3 et 4, dans cet ordre (par exemple `"ABCD"`) ;
b) `perfect_33` : le numéro du panneau dont le classifieur classe correctement **tous** ses points d'entraînement.

Dans tes notes : pour chaque panneau, le détail qui t'a décidé. Lequel de ces quatre modèles choisirais-tu pour ces données, et comment le vérifierais-tu ?""",
       given=GIVEN_33, todo=TODO_33, check=CHECK_33,
       solution=r'''panels_33, perfect_33 = "CADB", 2   # the answers, for the record
for letter, model in CLASSIFIERS_33.items():
    print(letter, type(model).__name__, "training accuracy:", round(model.score(X_33, Y_33), 3))
''' + as_printed(CHECK_33, "13.33"),
       record=r'''wb.record("13.33a", "CADB",
          mistakes={"relis les panneaux 1 et 3 : lequel trace une droite, lequel une courbe ? (§13.7, frontière de Naïve Bayes et de la régression logistique)": "DACB",
                    "relis les panneaux 2 et 4 : lequel est fait de segments horizontaux et verticaux, lequel épouse chaque point ? (§13.3 et §13.5.1)": "CBDA",
                    "relis les quatre panneaux : quel modèle trace des escaliers parallèles aux axes, lequel une droite ? (§13.7)": "ABCD"})
wb.record("13.33b", 2, choices=[1, 2, 3, 4],
          mistakes={"un modèle qui classe parfaitement l'entraînement fait de chaque point une région à lui : cherche les îlots autour des points isolés": 4,
                    "ce panneau laisse des points d'entraînement du mauvais côté de sa frontière : cherche celui où chaque point a sa région": 1,
                    "ce panneau-là laisse aussi des points d'entraînement du mauvais côté : cherche celui où chaque point a sa région": 3})''',
       note="Panneau 1 : une courbe lisse, ni droite ni en escalier, la frontière quadratique de Naïve Bayes "
            "gaussien (C). Panneau 2 : une frontière qui contourne chaque point, des îlots autour des points isolés : "
            "le plus proche voisin (A), seul à classer parfaitement l'entraînement (chaque point est son propre "
            "voisin). Panneau 3 : une droite, la régression logistique (D). Panneau 4 : des segments horizontaux et "
            "verticaux, l'arbre (B), qui fait 0,94 sur l'entraînement. Pour ces lunes, on comparerait un kNN avec "
            "un $k$ plus grand, un arbre réglé et un SVM RBF en validation croisée, sur les mêmes folds."),

    Ex("13.34", "🛠️", 2, 25, "Mesurer et vectoriser : accélérer son kNN",
       "mesurer un temps d'exécution proprement, et remplacer une boucle par du calcul vectorisé.",
       "Ex 13.14 · 0B (produit matriciel) · fiche §13.3 (le coût d'un kNN)", thread="MNIST", tracks="C",
       body=r"""`TRAIN_34` : 2 000 images de MNIST, `TEST_34` : 500, pixels dans $[0, 1]$. Un kNN passe son temps à calculer des distances : trois façons de calculer la matrice `(len(A), len(B))` des distances euclidiennes, et un outil de mesure. Écris :
- `distances_loop_34(A, B)` : deux boucles Python sur les lignes (NumPy pour la distance d'une paire) ;
- `distances_broadcast_34(A, B)` : une seule différence par broadcasting, `A[:, None, :] - B[None, :, :]`, sans boucle ;
- `distances_dot_34(A, B)` : l'identité $\lVert\mathbf{a} - \mathbf{b}\rVert^2 = \lVert\mathbf{a}\rVert^2 - 2\,\mathbf{a}\cdot\mathbf{b} + \lVert\mathbf{b}\rVert^2$, avec un seul produit matriciel pour tous les $\mathbf{a}\cdot\mathbf{b}$ ; elle ne doit **jamais** rendre `nan` ;
- `timed_34(function, *args, repeat=3)` : la plus petite de `repeat` durées de `function(*args)`, mesurées avec `time.perf_counter()` (le minimum écarte les mesures ralenties par autre chose ; `timeit` fait la même chose).

La vérification compare les trois versions à `scipy.spatial.distance.cdist`, mesure chacune sur un morceau du problème et estime son temps pour les 500 × 2 000 distances, compte les `nan` de l'identité, puis classe les images de test par leur plus proche voisin avec tes distances. Ne lance pas la version par broadcasting sur tout le problème : la vérification affiche la mémoire qu'elle demanderait. Cet exercice se corrige avec `05_solutions.md`.

Dans tes notes : classe les trois versions par vitesse ; pourquoi le broadcasting n'est-il pas le plus rapide ? D'où viennent les `nan` de l'identité, et pourquoi n'apparaissent-ils qu'entre une image et elle-même (ou une image presque identique) ?""",
       given=GIVEN_34, todo=TODO_34, check=CHECK_34, solution=SOLUTION_34 + CHECK_34.replace('with wb.attempt("13.34"):\n', "if True:\n"),
       note="La boucle calcule environ 4 µs par paire, soit quelques secondes pour les 500 × 2 000 distances. Le "
            "broadcasting est du même ordre que la boucle (l'une ou l'autre gagne selon la machine) : il fabrique un "
            "tableau intermédiaire de 500 × 2 000 × 784 nombres "
            "(6,3 Go, d'où sa mesure sur un morceau) et passe son temps à écrire et relire la mémoire. Le produit "
            "matriciel calcule tout en quelques centièmes de seconde : la bibliothèque d'algèbre linéaire découpe le "
            "calcul en blocs qui tiennent dans le cache du processeur, sur plusieurs cœurs. Entre une image et "
            "elle-même, $\\lVert\\mathbf{a}\\rVert^2 - 2\\,\\mathbf{a}\\cdot\\mathbf{a} + \\lVert\\mathbf{a}\\rVert^2$ "
            "vaut 0 en théorie, mais l'arrondi des deux calculs (une somme de carrés, un produit matriciel) peut "
            "laisser $-10^{-12}$ : sa racine vaut `nan`. `np.maximum(..., 0)` avant la racine règle le problème. "
            "Les pixels entiers de 0 à 255 ne le montrent pas : leurs sommes sont exactes en `float64`."),

    Ex("13.35", "🔬", 3, 45, "Comparatif mylearn contre scikit-learn : accuracy et temps",
       "comparer des implémentations sur les mêmes données, en accuracy et en temps.",
       "Ex 13.14, 13.24 et 13.28 · ch. 12 (PCA) · fiche §13.7", thread="MNIST", tracks="C",
       body=r"""500 images de MNIST pour l'entraînement et 500 pour le test, réduites à 50 coordonnées par une PCA (ch. 12). Écris `timed_fit_score_35(model, X_train, y_train, X_test, y_test)`, qui entraîne le modèle, mesure son accuracy sur le test, et rend le tuple `(accuracy, durée du fit en s, durée du score en s)`, mesurées avec `time.perf_counter()`.

La vérification l'applique à trois classifieurs de **ta** librairie et à leurs équivalents de scikit-learn : le kNN ($k = 3$), Naïve Bayes gaussien et l'arbre sans limite, affiche un tableau, contrôle les accuracies de ton kNN et de ton Naïve Bayes (a et b) et vérifie que ton arbre est à 3 points de celui de scikit-learn (les égalités de gain, fréquentes ici, se départagent autrement).

Dans tes notes : qui passe son temps à l'entraînement, qui à la prédiction ? Combien de fois ta librairie est-elle plus lente que scikit-learn pour chacun, et pourquoi (pense à ce que scikit-learn écrit en C, et à la boucle sur les seuils de ton `best_split`) ? Lequel des trois choisirais-tu sur ces données ?""",
       given=GIVEN_35, todo=TODO_35, check=CHECK_35, solution=SOLUTION_35 + CHECK_35.replace('with checked("13.35"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''wb.record("13.35a", results_35["kNN (k = 3)", "scikit-learn"][0], decimals=4)
wb.record("13.35b", results_35["GaussianNB", "scikit-learn"][0], decimals=4)''',
       note="Le kNN ne fait rien à l'entraînement et passe tout son temps à la prédiction (250 000 distances de 50 "
            "coordonnées) ; Naïve Bayes est rapide partout (un passage pour les moyennes, un calcul par classe à la "
            "prédiction) ; l'arbre passe son temps à l'entraînement (la recherche des seuils) et prédit presque "
            "instantanément. Naïve Bayes est du même ordre dans les deux librairies. Ton kNN, écrit avec la "
            "différence par broadcasting (13.14), est plusieurs fois plus lent que celui de scikit-learn (de 3 à "
            "plusieurs dizaines de fois selon la machine), qui "
            "passe par le produit matriciel de 🛠️ 13.34 et du code compilé. Ton arbre l'est de quelques fois (la "
            "référence, qui trie chaque feature une fois) à plusieurs centaines de fois (un `best_split` qui "
            "recalcule les impuretés à chaque seuil, en Python) : scikit-learn parcourt les seuils en C en mettant "
            "les comptes à jour. Sur ces données, le kNN (0,84) devance Naïve Bayes "
            "(0,82) et de loin l'arbre (0,58) : un arbre seul sur 50 coordonnées de PCA et 500 images surapprend ; "
            "les forêts du ch. 14 corrigeront cela."),

    Ex("13.36", "🏆", 4, 100, "Défi : 96 % sur MNIST avec un classifieur classique",
       "construire, régler et valider un classifieur classique qui atteint 96 % sur MNIST.",
       "Ex 13.18, 13.14 · ch. 12 (PCA, mise à l'échelle) · ch. 8 (validation croisée)", thread="MNIST", tracks="C",
       body=r"""**Objectif** : au moins **96 %** d'accuracy sur les 2 000 premières images de test de MNIST, avec au plus **20 000** images d'entraînement, sans réseau de neurones : un classifieur de ce chapitre (kNN, SVM, arbre, Naïve Bayes, régression logistique), avec la préparation du ch. 12 si tu veux (PCA, mise à l'échelle). Écris `N_TRAIN_36`, le nombre d'images d'entraînement, et `build_model_36()`, qui rend un modèle scikit-learn **neuf** (souvent un pipeline) qui reçoit des pixels dans $[0, 1]$.

Règles : les 2 000 images de test ne servent qu'à la mesure finale ; règle tes hyperparamètres par validation croisée sur l'entraînement (dans une cellule à toi, sur un sous-ensemble si c'est long). La vérification charge `N_TRAIN_36` images d'entraînement, entraîne ton modèle, mesure les temps et l'accuracy. Ce défi se corrige avec `05_solutions.md`.

Dans tes notes : les modèles essayés, leurs réglages et leurs scores de validation croisée ; le temps d'entraînement et de prédiction du modèle retenu ; ce que tu essaierais pour aller plus loin.""",
       given=GIVEN_36, todo=TODO_36, check=CHECK_36, solution=SOLUTION_36 + CHECK_36.replace('with wb.attempt("13.36"):\n', "if True:\n"),
       note="Une PCA à 50 composantes suivie d'un SVM RBF ($C = 10$, $\\gamma$ par défaut) atteint 96,7 % avec "
            "20 000 images, en quelques secondes. Un kNN sur les mêmes 50 composantes plafonne vers 95 % : le défi "
            "est réel. Les arbres seuls et Naïve Bayes en sont loin. Les réglages viennent de la validation croisée "
            "sur l'entraînement (05_solutions.md donne la grille)."),
])
PARTS = [PART_A, PART_B, PART_C, PART_D, PART_E, PART_F]
NEXT_SESSION = [
    ("A", "13.12 à 13.15", "prédire les frontières de cinq classifieurs, choisir k, KNeighborsClassifier dans "
                           "mylearn.neighbors, kNN et malédiction de la dimension"),
    ("B", "13.16 à 13.18", "SVM linéaire et paramètre C, relever des cercles en 3D, noyau RBF réglé par "
                           "validation croisée"),
    ("C", "13.19 à 13.26", "instabilité d'un arbre, impuretés, meilleur split et arbres dans mylearn.tree, "
                           "une colonne qui triche, plot_tree et élagage, arbre de régression"),
    ("D", "13.27 à 13.30", "Naïve Bayes sur des nuages gaussiens puis des lunes, GaussianNB et MultinomialNB dans "
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
               "`mylearn`). « Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est "
               "pas fait affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la "
               "section « Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch13_classifieurs/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 13`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
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
