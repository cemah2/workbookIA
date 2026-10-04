#!/usr/bin/env python
"""Build the two notebooks of chapter 12 from a single source (used by Claude).

    python tools/chapters/build_ch12.py
    python tools/run_all_notebooks.py chapitres/ch12_preparation/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch12_preparation/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q11, of the recalls R1 to R3 and of the ✏️ paper
exercises 12.1, 12.2, 12.3, 12.4, 12.5 and 12.7 (02_exercices.md). The proofs ∂ 12.6 and 12.8, the
reflection (🗣️ 12.9, ⚖️ 12.10) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 12.11 to 12.33) come with the second generation session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Paper, badge, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "12"
FOLDER = "chapitres/ch12_preparation"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np


def minmax(x, lo, hi, a=0.0, b=1.0):
    """Min-max scaling of x with the training minimum lo and maximum hi, towards [a, b]."""
    return a + (b - a) * (np.asarray(x, dtype=float) - lo) / (hi - lo)


# Q6: training temperatures from -5 to 25 degrees
Q6_30 = float(minmax(30, -5, 25))

# R3: f(x, y) = x^2 + 100 y^2
GRAD_R3 = [2 * 1, 200 * 1]

# 12.2: five training temperatures
T_122 = np.array([14.0, 17.0, 21.0, 23.0, 25.0])
MU_122, SD0_122, SD1_122 = T_122.mean(), T_122.std(ddof=0), T_122.std(ddof=1)

# 12.3: three feature ranges and one example
LO_123, HI_123 = np.array([10.0, 0.0, -20.0]), np.array([30.0, 5.0, 20.0])
X_123 = np.array([20.0, 4.0, 0.0])
GLO_123, GHI_123 = LO_123.min(), HI_123.max()

# 12.4: temperatures from -18 to 12, cars from 120 to 870
T_LO, T_HI, C_LO, C_HI = -18.0, 12.0, 120.0, 870.0

# 12.5: a 3 x 3 training table
X_125 = np.array([[1.0, 4.0, 10.0], [3.0, 8.0, 20.0], [5.0, 6.0, 30.0]])
NEW_125 = np.array([2.0, 2.0, 40.0])
COL_LO, COL_HI = X_125.min(axis=0), X_125.max(axis=0)


def row_minmax(row):
    return (row - row.min()) / (row.max() - row.min())


# 12.7: five points, their covariance and the principal direction (2, 1) / sqrt(5)
P_127 = np.array([[1.0, 1.0], [3.0, 3.0], [4.0, 3.0], [5.0, 5.0], [7.0, 3.0]])
MU_127 = P_127.mean(axis=0)
XC_127 = P_127 - MU_127
COV_127 = XC_127.T @ XC_127 / (len(P_127) - 1)
U_127 = np.array([2.0, 1.0]) / math.sqrt(5)
V_127 = np.array([-1.0, 2.0]) / math.sqrt(5)
LAMBDA_127 = float((COV_127 @ np.array([2.0, 1.0]))[0] / 2.0)
TOTAL_127 = float(np.trace(COV_127))
T_127 = XC_127 @ U_127
D_127 = P_127[3]
T_D = float((D_127 - MU_127) @ U_127)
REC_D = MU_127 + T_D * U_127'''

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("12.Q1", "La règle d'or de la préparation", [
        ("a", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"les données du test entrent alors dans le calcul de la moyenne : c'est une fuite (fiche §12.2 et §12.10)": "A",
          "le modèle a appris sur des données transformées avec certains paramètres : lesquels faut-il réutiliser ?": "C",
          "avec deux jeux de paramètres, un même nombre n'a plus le même sens à l'entraînement et au test": "D"}'''),
        ("b", 'the letter of your choice', '"B"',
         r'''mistakes={"réapprendre les paramètres sur une seule mesure change le sens des nombres que reçoit le modèle": "A",
          "le modèle a appris sur des données transformées : il ne connaît pas les unités brutes": "C",
          "réapprendre les paramètres sur d'autres données, même nombreuses, change le sens des nombres que reçoit le modèle": "D"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"couper les silences fait partie de la préparation : est-ce encore fait en service ?": True}'''),
        ("d", "True or False", "True",
         r'''mistakes={"toute étape de préparation, une colonne retirée comprise, fait partie du modèle (fiche §12.2)": False}'''),
    ]),
    Paper("12.Q2", "Numérique, ordinale ou nominale ?", [
        ("a", 'six letters Q, O or N, in the order of the items, e.g. "XYZXYZ"', '"QONNOQ"',
         r'''mistakes={"élément 4 (le code postal) : des chiffres ne font pas un nombre ; une moyenne de codes postaux aurait-elle un sens ?": "QONQOQ",
          "élément 4 (le code postal) : un ordre entre deux codes postaux aurait-il un sens pour le problème ?": "QONOOQ",
          "élément 2 (les avis) : ces trois réponses ont-elles un ordre qui a un sens ?": "QNNNOQ",
          "élément 5 (les tailles) : ces catégories ont-elles un ordre naturel ?": "QONNNQ",
          "élément 6 (le nombre d'enfants) : c'est un nombre, qui se compare et se calcule": "QONNOO",
          "élément 3 (l'île) : quel ordre aurait un sens entre des îles ?": "QOONOQ"}'''),
        ("b", 'the letter of your choice', '"B"',
         r'''mistakes={"c'est l'ordre naturel, mais l'encodeur ne le connaît pas : selon quel critère trie-t-il sans indication ?": "A",
          "l'encodeur range bien les catégories, mais pas par taille : selon quel critère, sans indication ?": "C",
          "l'encodeur trie les catégories qu'il a vues : selon quel critère ?": "D"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"un ordre inventé permet de trier, pas de donner un sens à « entre » (encadré ⚠️ du §12.3)": True}'''),
    ]),
    Paper("12.Q3", "Pourquoi un one-hot plutôt qu'un entier ?", [
        ("a", 'the letter of your choice', '"A"',
         r'''mistakes={"de train (1) à tram (2), de combien le code augmente-t-il ? Et le terme w × code ?": "B",
          "ce n'est le cas que si w vaut 0": "C",
          "avec une seule colonne et un seul poids, chaque catégorie peut-elle avoir son effet propre ?": "D"}'''),
        ("b", "a whole number", "3",
         r'''mistakes={"sans drop, chaque catégorie a sa propre colonne": 2,
          "un one-hot ne garde pas une seule colonne : relis sa définition": 1}'''),
        ("c", "a whole number", "2",
         r'''mistakes={"drop=\"first\" retire une colonne : compte celles qui restent": 3,
          "drop=\"first\" ne retire qu'une seule colonne": 1}'''),
        ("d", "a list of 0s and 1s, e.g. [1, 1, 1]", "[0, 0, 1]",
         r'''mistakes={"c'est le vecteur de la première catégorie dans l'ordre alphabétique ; où se range tram ?": [1, 0, 0],
          "train et tram commencent par les mêmes lettres : compare-les lettre à lettre": [0, 1, 0]}'''),
        ("e", "True or False", "True",
         r'''mistakes={"compare les écarts entre les codes de bus et de tram, puis de bus et de train": False}'''),
    ]),
    Paper("12.Q4", "Doublons, NaN et points aberrants", [
        ("a", 'the letter of your choice', '"B"',
         r'''mistakes={"38 kg pour une espèce qui pèse de 3 à 5 kg : est-ce plausible ?": "A",
          "une valeur manquante est une case vide ou un NaN ; ici, une valeur est présente": "C",
          "un doublon est une ligne répétée ; ici, les quatre masses sont différentes": "D"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"une ligne répétée entre deux fois dans chaque somme : moyenne, loss, vote": False}'''),
        ("c", "True or False", "False",
         r'''mistakes={"pour pandas, « ? » est un texte comme un autre : quelle option de read_csv lui dit d'y voir une valeur manquante ?": True}'''),
        ("d", 'the letter of your choice', '"C"',
         r'''mistakes={"Python lit le point comme séparateur décimal ; et la virgule ?": "A",
          "float() ne saute pas les caractères qu'il ne comprend pas : relis les formats dans la fiche §12.4": "B",
          "float() ne renvoie jamais 0 pour un texte illisible : relis les formats dans la fiche §12.4": "D"}'''),
        ("e", "True or False", "False",
         r'''mistakes={"NaN n'est égal à rien, pas même à lui-même : que vaut np.nan == np.nan ?": True}'''),
    ]),
    Paper("12.Q5", "Normaliser ou standardiser ?", [
        ("a", 'the letter of your choice', '"B"',
         r'''mistakes={"le min-max divise par l'étendue, ici fixée par la valeur 1 000 : où tombent alors les petites valeurs ?": "A",
          "le min-max n'envoie aucune valeur au milieu par principe : relis sa formule": "C",
          "le min-max vers [0, 1] ne donne aucune valeur négative sur l'entraînement": "D"}'''),
        ("b", "True or False", "False",
         r'''mistakes={"un z-score n'est pas borné par 3 : pense à 1 000 valeurs dont une très éloignée des autres": True}'''),
        ("c", "True or False", "False",
         r'''mistakes={"la standardisation déplace et dilate l'histogramme : change-t-elle sa forme ?": True}'''),
        ("d", "True or False", "True",
         r'''mistakes={"une transformation affine déplace et dilate l'histogramme sans changer sa forme": False}'''),
    ]),
    Paper("12.Q6", "Données de test hors de [0, 1] : bug ou normal ?", [
        ("a", "a number with 2 decimals", "Q6_30",
         r'''decimals=2, mistakes={"sans option, transform ne borne rien ; et le numérateur retire-t-il bien le minimum ?": 1.0,
          "le numérateur retire le minimum, pas le maximum": 0.17,
          "l'étendue vaut max − min, avec un minimum négatif : attention au signe": 1.75,
          "le numérateur retire le minimum, qui est négatif : attention au signe": 0.83,
          "si 1,2 est un arrondi, garde 2 décimales ; sinon, le minimum est-il retiré au numérateur et au dénominateur ?": 1.2,
          "le minimum vaut −5 : le retirer, c'est ajouter 5, au numérateur comme au dénominateur": 1.25,
          "le dénominateur est l'étendue max − min, pas le maximum": 1.4}'''),
        ("b", "True or False", "False",
         r'''mistakes={"une valeur plus chaude que toutes celles de l'entraînement dépasse 1 : relis la fiche §12.5.3": True}'''),
        ("c", "a number", "1.0",
         r'''decimals=2, mistakes={"clip=True ramène les valeurs aux bornes de feature_range": round(Q6_30, 2),
          "clip ramène une valeur trop grande à la borne la plus proche : laquelle ?": 0}'''),
        ("d", 'the letter of your choice', '"B"',
         r'''mistakes={"le test n'a pas servi à calculer la moyenne : pourquoi aurait-il une moyenne exactement nulle ?": "A",
          "le scaler fixe l'écart-type de l'entraînement à 1, pas la moyenne du test": "C",
          "rien n'oblige le test à être plus bas que l'entraînement": "D"}'''),
    ]),
    Paper("12.Q7", "Univarié ou multivarié ?", [
        ("a", 'six letters U or M, in the order of the items', '"UMMUMU"',
         r'''mistakes={"élément 1 (StandardScaler) : la moyenne et l'écart-type d'une colonne utilisent-ils les autres colonnes ?": "MMMUMU",
          "élément 3 (le min-max global) : le minimum et le maximum viennent de toutes les colonnes ensemble": "UMUUMU",
          "élément 5 (KNNImputer) : pour trouver les plus proches voisins, on compare les autres features de l'exemple": "UMMUUU",
          "élément 6 (le logarithme) : le logarithme d'une valeur dépend-il des autres colonnes ?": "UMMUMM",
          "élément 4 (la médiane) : la médiane d'une colonne se calcule-t-elle avec les autres colonnes ?": "UMMMMU",
          "élément 2 (la PCA) : une direction principale combine plusieurs features": "UUMUMU"}'''),
        ("b", "True or False", "False",
         r'''mistakes={"univariée : chaque colonne est transformée avec ses seules valeurs": True}'''),
    ]),
    Paper("12.Q8", "Sélectionner ou réduire la dimension ?", [
        ("a", 'five letters S or R, in the order of the items', '"SRSRS"',
         r'''mistakes={"élément 1 (la colonne constante) : les colonnes qui restent sont-elles modifiées ?": "RRSRS",
          "élément 4 (l'IMC) : garde-t-on une colonne d'origine, ou en fabrique-t-on une nouvelle ?": "SRSSS",
          "élément 3 (les 5 features les plus corrélées) : ces colonnes sont-elles modifiées ?": "SRRRS",
          "élément 5 (VarianceThreshold) : que fait-il des colonnes qu'il garde ?": "SRSRR",
          "élément 2 (la PCA) : ses composantes sont-elles des colonnes d'origine ?": "SSSRS"}'''),
        ("b", "True or False", "False",
         r'''mistakes={"la cible du test a servi à choisir les features : relis 8.25 et la fiche §12.6": True}'''),
        ("c", "True or False", "False",
         r'''mistakes={"une composante est une somme pondérée de toutes les features": True}'''),
    ]),
    Paper("12.Q9", "Ce que fait (et ne fait pas) une PCA", [
        ("a", "True or False", "True",
         r'''mistakes={"relis la définition de la première composante (fiche §12.7.1)": False}'''),
        ("b", "True or False", "True",
         r'''mistakes={"chaque composante est cherchée parmi les directions orthogonales aux précédentes": False}'''),
        ("c", "True or False", "False",
         r'''mistakes={"la PCA ne reçoit que X, jamais y : elle est non supervisée": True}'''),
        ("d", "True or False", "True",
         r'''mistakes={"une composante est une somme pondérée des features centrées, rien d'autre": False}'''),
        ("e", "True or False", "True",
         r'''mistakes={"en grammes, les nombres sont 1 000 fois plus grands : que devient la variance de cette feature ?": False}'''),
        ("f", "True or False", "False",
         r'''mistakes={"la PCA ne regarde pas les labels : une direction de petite variance peut séparer les classes": True}'''),
    ]),
    Paper("12.Q10", "Quelle découpe pour ces données ?", [
        ("a", 'six letters L, C or E, in the order of the items', '"ELCELC"',
         r'''mistakes={"élément 2 (l'électrocardiogramme) : son propre maximum, c'est le maximum de quelle tranche du tableau ?": "ECCELC",
          "élément 5 (Normalizer) : la norme de quoi, d'une ligne ou d'une colonne ?": "ELCECC",
          "élément 5 (Normalizer) : la norme utilise-t-elle d'autres cases que celle qu'on transforme ?": "ELCEEC",
          "élément 1 (diviser par 255) : 255 dépend-il des autres pixels de l'image ?": "LLCELC",
          "élément 1 (diviser par 255) : 255 est-il calculé sur les autres images ?": "CLCELC",
          "élément 4 (un taux fixé) : ce taux dépend-il des autres prix de l'exemple ?": "ELCLLC",
          "élément 4 (un taux fixé) : ce taux est-il calculé sur les autres exemples ?": "ELCCLC",
          "élément 6 (la médiane de chaque colonne) : sur quelles valeurs est-elle calculée ?": "ELCELE"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"une telle transformation n'utilise aucune autre ligne : par où l'information circulerait-elle ?": False}'''),
    ]),
    Paper("12.Q11", "Où se cache la fuite ?", [
        ("a", 'the letters, in alphabetical order, e.g. "BF"', '"ACE"',
         r'''mistakes={"F : diviser par une constante fixée d'avance n'apprend rien des données": "ACEF",
          "une procédure qui regarde la cible sur tout le dataset manque": "AC",
          "D : le scaler n'a vu que l'entraînement": "ACDE",
          "une imputation calculée avant le découpage apprend aussi quelque chose du test": "AE",
          "B : le pipeline est réajusté en entier dans chaque fold": "ABCE",
          "une standardisation faite avant la validation croisée a vu tous les folds": "CE"}'''),
        ("b", "True or False", "False",
         r'''mistakes={"cross_val_score reçoit des données déjà transformées : peut-il défaire ce fit ?": True}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("12.R1", "Représentation, évaluation, optimisation — où placer la préparation ?", [
        ("a", 'the letter of your choice', '"A"',
         r'''mistakes={"ajouter x₁² change l'ensemble des fonctions que le modèle peut exprimer": "C",
          "la mesure de la qualité d'une solution ne change pas": "B"}'''),
        ("b", 'the letter of your choice', '"C"',
         r'''mistakes={"sans pénalité, une régression sur des features standardisées exprime exactement les mêmes fonctions": "A",
          "la loss mesure toujours l'écart aux cibles de la même façon": "B"}'''),
        ("c", 'the letter of your choice', '"B"',
         r'''mistakes={"le F1-score juge les solutions, il n'en cherche pas": "C",
          "choisir la mesure ne change pas les modèles possibles": "A"}'''),
        ("d", "True or False", "True",
         r'''mistakes={"une information absente des features ne peut pas être exprimée, quel que soit l'optimiseur": False}'''),
    ]),
    Paper("12.R2", "Pourquoi la pénalité ridge dépend de l'échelle des features", [
        ("a", 'the letter of your choice', '"C"',
         r'''mistakes={"x' = 100 x : pour garder w' x' = w x, le poids doit compenser le facteur 100": "B",
          "sans changer le poids, la contribution de la feature serait multipliée par 100": "A",
          "w' x' doit valoir w x : de quel facteur faut-il diviser w ?": "D"}'''),
        ("b", "a whole number", "100 ** 2",
         r'''mistakes={"relis la pénalité : α w'², c'est α fois quoi ?": 100},
          fractional="on demande par combien la pénalité est divisée : un nombre plus grand que 1"'''),
        ("c", 'the letter of your choice', '"B"',
         r'''mistakes={"en centimètres, le poids devient plus petit : et sa pénalité ?": "A",
          "la pénalité dépend de la taille du poids, qui dépend de l'unité": "C"}'''),
        ("d", "True or False", "True",
         r'''mistakes={"x' = 100 x a les mêmes z-scores que x : la standardisation efface l'unité": False}'''),
    ]),
    Paper("12.R3", "Descente de gradient dans une vallée très allongée", [
        ("a", "a list of two numbers", "GRAD_R3",
         r'''mistakes={"la dérivée de 100 y² par rapport à y n'est pas 100": [2, 100],
          "la dérivée de x² par rapport à x n'est pas 1": [1, 200]}'''),
        ("b", "a number", "0.01",
         r'''decimals=4, mistakes={"ce pas annule le facteur : c'est le plus rapide sur y, pas la limite de divergence": 0.005,
          "la direction x n'est pas la plus contraignante : regarde le facteur de y": 1.0,
          "calcule le facteur 1 − 200η pour ce pas : il sort de [−1, 1]": 0.02}'''),
        ("c", "a number", "1 - 2 * 0.009",
         r'''decimals=3, mistakes={"la dérivée de x² n'est pas x : refais le pas sur x": 0.991,
          "−0,8 est le facteur de y ; la question porte sur x": -0.8}'''),
        ("d", "a number", "1.0",
         r'''decimals=2, mistakes={"après le changement d'échelle, recalcule la dérivée de y'²": 0.01,
          "c'est le pas qui annule le facteur, pas la limite de divergence": 0.5,
          "écris 100 y² en fonction de y' = 10 y, puis refais le calcul de b) avec ce nouveau terme": 0.1}'''),
        ("e", "True or False", "True",
         r'''mistakes={"quand la vallée redevient ronde, le même learning rate convient aux deux directions": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("12.1", "One-hot à la main sur Penguins", [
        ("a", "a whole number", "3 + 2 + 1",
         r'''mistakes={"n'oublie pas flipper_length_mm, gardée telle quelle": 5,
          "compte une colonne par catégorie dans chaque bloc, plus la feature numérique": 3}'''),
        ("b", "a list of numbers", "[0, 0, 1, 1, 0, 185]",
         r'''mistakes={"bloc sex : dans l'ordre alphabétique, female vient avant male": [0, 0, 1, 0, 1, 185],
          "bloc island : Torgersen est-elle la première île dans l'ordre alphabétique ?": [1, 0, 0, 1, 0, 185]}'''),
        ("c", "a whole number", "2 + 1 + 1",
         r'''mistakes={"drop=\"first\" retire une colonne par feature catégorielle, rien à la feature numérique": 3,
          "deux features catégorielles : une colonne retirée pour chacune": 5}'''),
        ("d", "a list of numbers", "[0, 0, 1, 220]",
         r'''mistakes={"drop=\"first\" retire la colonne de la PREMIÈRE catégorie de chaque bloc, dans l'ordre alphabétique": [1, 0, 0, 220],
          "drop=\"first\" retire une colonne dans chaque bloc catégoriel : compte les colonnes qui restent": [1, 0, 0, 0, 1, 220]}'''),
        ("e", "a whole number", "3 + 1 + 1",
         r'''mistakes={"if_binary ne retire une colonne qu'aux features à deux catégories": 4,
          "une feature à deux catégories perd une colonne avec if_binary": 6}'''),
        ("f", "a list of numbers", "[0, 0, 0, 1, 0, 195]",
         r'''mistakes={"une île inconnue n'est rangée dans aucune des catégories apprises": [0, 0, 1, 1, 0, 195],
          "une île inconnue ne prend pas la place de la première catégorie apprise": [1, 0, 0, 1, 0, 195]}'''),
        ("g", "a whole number", "2",
         r'''mistakes={"les codes commencent à 0": 3,
          "l'ordre est alphabétique : où se range Torgersen parmi les trois îles ?": 1}'''),
    ]),
    Paper("12.2", "Min-max et z-score de cinq valeurs", [
        ("a", "a number", "MU_122",
         r'''decimals=2, mistakes={"la moyenne n'est pas la valeur du milieu (la médiane)": 21}'''),
        ("b", "a number", "SD0_122",
         r'''decimals=3, mistakes={"tu as divisé la somme des carrés par n − 1 (ddof = 1) ; l'énoncé demande ddof = 0, comme StandardScaler": round(SD1_122, 3),
          "ton écart-type est calculé avec n − 1 au dénominateur (ddof = 1) ; ici, ddof = 0": round(SD1_122, 2),
          "tu as donné une variance : l'écart-type est sa racine": 16,
          "tu as donné une variance, et avec n − 1 : l'écart-type demandé est la racine de la variance avec n": 20}'''),
        ("c", "a number with 3 decimals", "float(minmax(21, 14, 25))",
         r'''decimals=3, mistakes={"retire le minimum (pas la moyenne) avant de diviser par l'étendue": 0.091,
          "divise par l'étendue max − min, pas par le maximum": 0.28,
          "tu as divisé la valeur par le maximum : retire d'abord le minimum": 0.84}'''),
        ("d", "a number", "(17 - MU_122) / SD0_122",
         r'''decimals=2, mistakes={"17 est sous la moyenne : son z-score est négatif": 0.75,
          "divise l'écart à la moyenne par l'écart-type": -3,
          "ton écart-type est calculé avec n − 1 au dénominateur ; l'énoncé demande ddof = 0": -0.67,
          "tu as divisé l'écart à la moyenne par la variance : divise par l'écart-type": -0.19}'''),
        ("e", "a number", "(25 - MU_122) / SD0_122",
         r'''decimals=2, mistakes={"divise l'écart à la moyenne par l'écart-type": 5,
          "ton écart-type n'est pas celui de l'énoncé : vérifie le dénominateur de la variance (ddof = 0)": 1.12,
          "tu as divisé l'écart à la moyenne par la variance : divise par l'écart-type": 0.31}'''),
        ("f", "a number with 3 decimals", "(25 - MU_122) / SD1_122",
         r'''decimals=3, mistakes={"quel dénominateur as-tu pris pour la variance ? Series.std() divise par n − 1": 1.25}'''),
        ("g", "a number with 3 decimals", "float(minmax(21, 14, 25, -1, 1))",
         r'''decimals=3, mistakes={"vérifie la formule vers [a, b] : as-tu bien pris a = −1 et b = 1 ?": 0.636,
          "il manque la multiplication par b − a = 2": -0.364,
          "il manque un terme : après la multiplication par b − a, la formule ajoute a": 1.273}'''),
    ]),
    Paper("12.3", "Mise à l'échelle univariée ou multivariée", [
        ("a", "a list of 3 numbers", "minmax(X_123, LO_123, HI_123).tolist()",
         r'''decimals=2, mistakes={"retire d'abord le minimum de chaque feature": [0.67, 0.8, 0.0]}'''),
        ("b", "a list of 3 numbers", "minmax(X_123, GLO_123, GHI_123).tolist()",
         r'''decimals=2, mistakes={"en multivarié, le minimum et le maximum sont communs aux trois features : vérifie lesquels tu as pris": [0.5, 0.8, 0.5],
          "retire le minimum global, qui est négatif, avant de diviser": [0.4, 0.08, 0.0],
          "le minimum global vaut −20 : le retirer, c'est ajouter 20": [0.0, -0.32, -0.4],
          "le minimum et le maximum globaux viennent des trois features, pas d'une seule": [1.0, 0.6, 0.5]}'''),
        ("c", "a number", "(HI_123[1] - LO_123[1]) / (GHI_123 - GLO_123)",
         r'''decimals=3, mistakes={"en multivarié, f2 ne remplit plus tout [0, 1] : elle subit la même dilatation que les autres features": 1.0,
          "l'étendue globale va du plus petit minimum au plus grand maximum des trois features": 0.125,
          "l'étendue globale va du plus petit minimum au plus grand maximum des trois : pas seulement ceux de f3": 0.13}'''),
        ("d", "True or False", "True",
         r'''mistakes={"le min-max multivarié applique la même dilatation à toutes les features": False}'''),
        ("e", 'the letter of your choice', '"B"',
         r'''mistakes={"un min-max par colonne donne à chaque magasin son propre minimum : une même valeur transformée ne veut plus dire le même montant": "A"}'''),
    ]),
    Paper("12.4", "Réappliquer la transformation : −10 °C, −50 °C et retour aux voitures", [
        ("a", "a number with 3 decimals", "float(minmax(-10, T_LO, T_HI))",
         r'''decimals=3, mistakes={"le numérateur retire le minimum, pas le maximum": -0.733,
          "le minimum vaut −18 : le retirer, c'est ajouter 18": -0.933,
          "retire d'abord le minimum, puis divise par l'étendue": -0.333,
          "l'étendue vaut max − min = 12 − (−18) : attention au signe": 1.333,
          "l'étendue max − min est positive : 12 − (−18), soit 12 + 18": -1.333}'''),
        ("b", "a number of cars", "C_LO + 0.40 * (C_HI - C_LO)",
         r'''decimals=0, mistakes={"l'inverse ajoute aussi le minimum des comptages": 300,
          "l'inverse multiplie par l'étendue des comptages, max − min, pas par le maximum": 468,
          "deux oublis : l'inverse multiplie par l'étendue (max − min), puis ajoute le minimum": 348,
          "−6 voitures ? Vérifie le minimum et le maximum que tu as utilisés pour revenir en arrière": -6}'''),
        ("c", "a number with 3 decimals", "float(minmax(-50, T_LO, T_HI))",
         r'''decimals=3, mistakes={"retire d'abord le minimum, puis divise par l'étendue": -1.667,
          "attention au signe : cette nuit est bien plus froide que le minimum d'entraînement": 1.067,
          "le minimum vaut −18 : le retirer, c'est ajouter 18": -2.267,
          "sans clip=True, transform ne borne rien : applique la formule telle quelle": 0}'''),
        ("d", "a number with 3 decimals", "float(minmax(20, T_LO, T_HI))",
         r'''decimals=3, mistakes={"retire d'abord le minimum, puis divise par l'étendue": 0.667,
          "le minimum vaut −18 : le retirer, c'est ajouter 18": 0.067,
          "sans clip=True, transform ne borne rien : applique la formule telle quelle": 1.0}'''),
        ("e", "True or False", "False",
         r'''mistakes={"une valeur hors de la plage d'entraînement sort de [0, 1] : relis la fiche §12.5.3": True}'''),
        ("f", "a number of cars", "C_LO + (-0.1) * (C_HI - C_LO)",
         r'''decimals=0, mistakes={"l'inverse s'applique aussi hors de [0, 1] : n'oublie pas d'ajouter le minimum": -75,
          "rien ne borne une prédiction hors de [0, 1] : l'inverse s'applique telle quelle": 120}'''),
        ("g", "a number with 3 decimals", "(-10 - (-3)) / 6",
         r'''decimals=3, mistakes={"−10 est sous la moyenne : z est négatif": 1.167,
          "retire la moyenne avant de diviser par l'écart-type": -1.667,
          "la moyenne vaut −3 : la retirer, c'est ajouter 3": -2.167}'''),
        ("h", "a number of cars", "480 + 0.5 * 150",
         r'''decimals=0, mistakes={"l'inverse multiplie par l'écart-type avant d'ajouter la moyenne": 480.5,
          "n'oublie pas d'ajouter la moyenne": 75}'''),
        ("i", "True or False", "False",
         r'''mistakes={"chaque grandeur a son transformateur : la cible se ramène avec le sien": True}'''),
    ]),
    Paper("12.5", "Trois découpes d'un même tableau : échantillon, feature, élément", [
        ("a", "a list of 3 numbers", "minmax(X_125[1], COL_LO, COL_HI).tolist()",
         r'''decimals=2, mistakes={"par feature : chaque colonne avec son propre minimum et son propre maximum, pas chaque ligne": [0.0, 0.29, 1.0]}'''),
        ("b", "a list of 3 numbers", "row_minmax(X_125[0]).tolist()",
         r'''decimals=3, mistakes={"par échantillon : la ligne avec son propre minimum et son propre maximum, pas chaque colonne": [0.0, 0.0, 0.0]}'''),
        ("c", "a number with 3 decimals", "float(X_125[1, 0] / np.linalg.norm(X_125[1]))",
         r'''decimals=3, mistakes={"divise par la norme (la racine de la somme des carrés), pas par la somme": 0.097,
          "la norme se calcule sur la ligne, pas sur la colonne": 0.507,
          "la norme est la racine de la somme des carrés : as-tu pris la racine ?": 0.006}'''),
        ("d", "a list of 3 numbers", "(0.9 * X_125[2]).tolist()",
         r'''decimals=2, mistakes={"à 0,9 € pour 1 $, on multiplie, on ne divise pas": [5.56, 6.67, 33.33]}'''),
        ("e", 'the letters, in alphabetical order', '"A"',
         r'''mistakes={"une conversion à un taux fixé d'avance n'apprend rien des données": "AD",
          "chaque ligne se transforme avec son propre minimum et son propre maximum : rien n'est appris des autres lignes": "AB",
          "la norme d'une ligne ne dépend que de cette ligne": "AC",
          "B et C sont par échantillon : utilisent-elles les autres lignes de l'entraînement ?": "ABC",
          "toutes ne s'apprennent pas : lesquelles utilisent les autres exemples de l'entraînement ?": "ABCD"}'''),
        ("f", "a list of 3 numbers", "minmax(NEW_125, COL_LO, COL_HI).tolist()",
         r'''decimals=2, mistakes={"par feature, chaque valeur se transforme avec le minimum et le maximum de sa colonne d'entraînement : vérifie lesquels tu as pris": [0.0, 0.0, 1.0],
          "rien ne borne une donnée nouvelle : pas de clip": [0.25, 0.0, 1.0]}'''),
        ("g", "a list of 3 numbers", "row_minmax(NEW_125).tolist()",
         r'''decimals=2, mistakes={"par échantillon, la ligne se transforme avec son propre minimum et son propre maximum : vérifie lesquels tu as pris": [0.25, -0.5, 1.5]}'''),
    ]),
    Paper("12.7", "PCA à la main en 2D : covariance, axe principal, projection", [
        ("a", "a list of 2 numbers", "MU_127.tolist()",
         r'''decimals=2'''),
        ("b", "a list of 2 rows, e.g. [[1, 1], [1, 1]]", "COV_127.tolist()",
         r'''decimals=2, mistakes={"divise par n − 1 (ddof = 1), comme np.cov": [[4.0, 1.6], [1.6, 1.6]],
          "tu as la somme des produits des écarts : divise par n − 1": [[20, 8], [8, 8]],
          "centre d'abord les points : la covariance porte sur les écarts au point moyen": [[25, 17], [17, 13.25]]}'''),
        ("c", "a list of 2 numbers", "(COV_127 @ np.array([2.0, 1.0])).tolist()",
         r'''decimals=2, mistakes={"l'énoncé demande le produit par (2, 1), sans normaliser le vecteur": [5.37, 2.68]}'''),
        ("d", "a number", "LAMBDA_127",
         r'''decimals=2, mistakes={"12 est une coordonnée du produit, pas la valeur propre : relis la définition Σ u = λ u": 12}'''),
        ("e", "a number", "TOTAL_127",
         r'''decimals=2, mistakes={"la variance totale ne compte pas les covariances": 11}'''),
        ("f", "a number with 3 decimals", "LAMBDA_127 / TOTAL_127",
         r'''decimals=3, mistakes={"tu as pris la variance de la première feature ; la composante a la sienne (question d)": 0.714,
          "la variance totale ne compte pas les covariances : revois e)": 0.545}'''),
        ("g", "a list of 5 numbers", "T_127.tolist()",
         r'''decimals=3, mistakes={"centre d'abord les points (retire le point moyen)": [1.342, 4.025, 4.919, 6.708, 7.603],
          "u est unitaire : divise (2, 1) par sa norme": [-8, -2, 0, 4, 6]}'''),
        ("h", "a number", "float(np.var(T_127, ddof=1))",
         r'''decimals=2, mistakes={"ddof = 1 : divise la somme des carrés par n − 1": 4.8}'''),
        ("i", "a list of 2 numbers", "REC_D.tolist()",
         r'''decimals=2, mistakes={"ajoute le point moyen à t u": [1.6, 0.8],
          "la reconstruction reste sur la droite principale : elle ne retrouve pas le point": [5, 5],
          "u est unitaire : la reconstruction ajoute t u, pas t (2, 1)": [7.58, 4.79]}'''),
        ("j", "a number with 3 decimals", "float(np.linalg.norm(D_127 - REC_D))",
         r'''decimals=3, mistakes={"tu as la distance au point moyen, pas à la reconstruction": 2.236,
          "tu as la distance au carré : prends sa racine": 1.8}'''),
        ("k", "a number with 3 decimals", "T_D / math.sqrt(LAMBDA_127)",
         r'''decimals=3, mistakes={"divise par la racine de la valeur propre, pas par la valeur propre": 0.298,
          "le whitening divise encore la coordonnée par l'écart-type de sa composante": 1.789}'''),
        ("l", "a number", "float(np.var(XC_127 @ V_127, ddof=1))",
         r'''decimals=2, mistakes={"ddof = 1 : divise la somme des carrés par n − 1": 0.8,
          "v est unitaire : divise (−1, 2) par sa norme avant de projeter": 5,
          "les points ne sont pas tous sur l'axe principal : calcule leurs coordonnées sur v": 0}'''),
        ("m", "True or False", "False",
         r'''mistakes={"multiplier une feature par 10 multiplie sa variance par 100 : vers où l'axe principal tourne-t-il ?": True}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 12.1 à 12.5 et 12.7)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "plusieurs choix : les lettres collées (`\"BF\"`) ; une suite de codes : les lettres dans l'ordre "
               "(`\"XYZ\"`) ; plusieurs nombres : une liste (`[4, 7]`) ; une matrice : une liste de lignes "
               "(`[[1, 1], [1, 1]]`). Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les "
               "réponses pas encore remplies affichent ⏳. Les questions « dans ta copie », les preuves ∂ 12.6 et "
               "12.8, la réflexion (🗣️ 12.9, ⚖️ 12.10) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 12.11 to 12.33): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "12.11 à 12.15", "diagnostiquer Penguins brut, un CSV à la française, où tombent les données de test "
                           "après un MinMaxScaler, dates, jointures et tableaux croisés avec pandas, une base SQL"),
    ("B", "12.16 à 12.21", "StandardScaler, MinMaxScaler et SimpleImputer dans mylearn.preprocessing, scalers "
                           "face aux points aberrants, bugs de préparation, transformer la cible"),
    ("C", "12.22 à 12.28", "projeter un nuage, variance expliquée, MNIST en composantes, sélection de "
                           "features, OrdinalEncoder, OneHotEncoder et PCA dans mylearn.preprocessing"),
    ("D", "12.29 à 12.33", "échelle et descente de gradient, PCA, t-SNE et UMAP, fuites par le prétraitement, "
                           "une préparation testée, défi Penguins brut"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 12.Q1 à 12.Q11, 12.R1 à 12.R3, 12.1 à 12.5, 12.7 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | "
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
        title = "# 12 · Préparation des données — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch12_preparation/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 12`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 12 · Préparation des données — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Diagnostiquer et nettoyer un dataset brut avec pandas, et l'assembler avec des jointures et SQL.\n"
               "- Écrire et vérifier contre scikit-learn les transformateurs de `mylearn.preprocessing` : "
               "`StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OrdinalEncoder`, `OneHotEncoder` et `PCA`.\n"
               "- Choisir une mise à l'échelle et un encodage, et appliquer la règle d'or : `fit` sur "
               "l'entraînement, `transform` partout.\n"
               "- Réduire et visualiser des données en grande dimension (PCA, t-SNE, UMAP), et mesurer les fuites "
               "dues au prétraitement.\n\n"
               "**Rappel express.** Min-max : $x' = a + (b - a)\\,\\frac{x - \\min}{\\max - \\min}$ ; "
               "standardisation : $z = \\frac{x - \\mu}{\\sigma}$ (ddof = 0), inverse $x = z\\,\\sigma + \\mu$ ; "
               "covariance : $\\boldsymbol{\\Sigma} = \\frac{1}{n - 1}\\mathbf{X}_c^\\top\\mathbf{X}_c$ ; variance des "
               "projections sur $\\mathbf{u}$ unitaire : $\\mathbf{u}^\\top\\boldsymbol{\\Sigma}\\,\\mathbf{u}$ ; PCA : "
               "$\\mathbf{Z} = (\\mathbf{X} - \\boldsymbol{\\mu})\\,\\mathbf{W}_k^\\top$, "
               "$\\hat{\\mathbf{X}} = \\mathbf{Z}\\,\\mathbf{W}_k + \\boldsymbol{\\mu}$, part de variance expliquée "
               "$\\lambda_j / \\sum_l \\lambda_l$. En Python : `sklearn.preprocessing` (`StandardScaler`, "
               "`MinMaxScaler`, `RobustScaler`, `OrdinalEncoder`, `OneHotEncoder`), `sklearn.impute.SimpleImputer`, "
               "`sklearn.decomposition.PCA`, `sklearn.pipeline.make_pipeline`, `np.linalg.svd`.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 12.11 à 12.33 (parties A à D, ta librairie `mylearn.preprocessing`) "
             "seront ajoutés à la prochaine session de génération ; relance alors `python tools/start_chapter.py "
             "12` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dire, pour chaque étape d'une préparation, ce qu'elle apprend des données, et donc où "
               "faire son `fit` ?\n"
               "2. Sais-tu choisir un encodage et une mise à l'échelle selon le type des données et le modèle qui "
               "suit ?\n"
               "3. Sais-tu mener une PCA à la main en dimension 2, et dire ce qu'elle garde et ce qu'elle perd ?\n\n"
               "**Pour aller plus loin** : le guide « Preprocessing data » et la page « Common pitfalls » de "
               "scikit-learn, et le tutoriel de J. Shlens sur la PCA, cités dans la fiche. La suite : le ch. 13 "
               "(classifieurs), où les k plus proches voisins et les SVM dépendront de tes mises à l'échelle, et "
               "pas les arbres de décision." + later)]


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
