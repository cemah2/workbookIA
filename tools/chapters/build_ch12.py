#!/usr/bin/env python
"""Build the two notebooks of chapter 12 from a single source (used by Claude).

    python tools/chapters/build_ch12.py
    python tools/run_all_notebooks.py chapitres/ch12_preparation/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch12_preparation/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q11, of the recalls R1 to R3 and of the ✏️ paper
exercises 12.1, 12.2, 12.3, 12.4, 12.5 and 12.7 (02_exercices.md). The proofs ∂ 12.6 and 12.8, the
reflection (🗣️ 12.9, ⚖️ 12.10) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 12.11 to 12.33): raw data and SQL, scalers and imputation, PCA and encoders, leaks and the
challenge; the 🔨 exercises complete mylearn/preprocessing.py (tests: tests/test_ch12_preprocessing.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

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
    Paper("12.R1", "Ch. 11 : représentation, évaluation, optimisation — où placer la préparation ?", [
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
    Paper("12.R2", "Ch. 9 : pourquoi la pénalité ridge dépend de l'échelle des features", [
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
    Paper("12.R3", "Ch. 5 : descente de gradient dans une vallée très allongée", [
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
# Part A: diagnose and assemble raw data (12.11 to 12.15)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools for the notebook exercises (parts A to D)
import contextlib
import io
import math
import sqlite3
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def fitted(transformer, X):
    """The transformer after transformer.fit(X): no chained call, so that a fit that forgets `return self` (the
    tests report it) does not stop the check cells."""
    transformer.fit(X)
    return transformer


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


TEST_FILE = "tests/test_ch12_preprocessing.py"


def run_mylearn_tests(keyword, impl="learner"):
    """Run the tests of mylearn.preprocessing selected by `keyword` (on YOUR code by default)."""
    command = [sys.executable, "-m", "pytest", TEST_FILE, "-k", keyword, "-q", "-p", "no:cacheprovider",
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
        name, _, reason = line.removeprefix(f"FAILED {TEST_FILE}::").partition(" - ")
        print(f"❌ {name}\n   {reason[:800]}")
    if len(failed) > 8:
        print(f"   ... and {len(failed) - 8} other failed test(s)")
    print("pytest:", lines[-1] if lines else result.stderr.strip()[-300:])


RAW = wb.datasets.load_penguins_raw()    # penguins_raw.csv as it is: 344 rows, 17 columns
print(RAW.shape)'''

SOLUTION_11 = r'''counts_11 = RAW.nunique(dropna=False)                     # NaN counted as a value
constant_11 = sorted(counts_11[counts_11 == 1].index)
n_ids_11 = RAW["Individual ID"].nunique()
by_id_11 = RAW.groupby("Individual ID")
multi_season_11 = int((by_id_11["studyName"].nunique() > 1).sum())
multi_species_11 = int((by_id_11["Species"].nunique() > 1).sum())
n15_only_11 = int((RAW["Delta 15 N (o/oo)"].isna() & RAW["Delta 13 C (o/oo)"].notna()).sum())
print(constant_11, n_ids_11, multi_season_11, multi_species_11, n15_only_11)
print(RAW.loc[RAW["Delta 15 N (o/oo)"].isna() & RAW["Delta 13 C (o/oo)"].notna(), "Comments"].tolist())'''

CHECK_11 = r'''wb.check("12.11a", constant_11)
wb.check("12.11b", n_ids_11)
wb.check("12.11c", multi_season_11)
wb.check("12.11d", multi_species_11)
wb.check("12.11e", n15_only_11)'''

ROWS_11 = 'RAW.groupby("Individual ID").filter(lambda group: group[{col!r}].nunique() > 1).shape[0]'

CSV_12 = r'''CSV_12 = """espece;ile;bec_mm;masse_t
Adelie;Torgersen;39,1;3,75e-3
Adelie;Torgersen;39,5;3,8e-3
Adelie;Torgersen;40,3;3,25e-3
Adelie;Torgersen;?;?
Adelie;Torgersen;36,7;3,45e-3
Adelie;Torgersen;39,3;3,65e-3
Gentoo;Biscoe;46,1;4,5e-3
Gentoo;Biscoe;50;5,7e-3
Gentoo;Biscoe;51,1;6e-3
Gentoo;Biscoe;n.d.;n.d.
Chinstrap;Dream;46,5;3,5e-3
Chinstrap;Dream;50;3,9e-3
"""


def load_buggy_12(text):
    """The colleague's loader: the separator is right, the rest is not."""
    df = pd.read_csv(io.StringIO(text), sep=";")
    for column in ["bec_mm", "masse_t"]:
        df[column] = pd.to_numeric(df[column].str.replace(",", ""), errors="coerce")   # "remove the commas"
    df["masse_g"] = df["masse_t"] * 1000                                                # tonnes -> grams
    return df.drop(columns="masse_t")


buggy_12 = load_buggy_12(CSV_12)
print(buggy_12)
print("mean bill (mm):", round(buggy_12["bec_mm"].mean(), 2), "· mean mass (g):", round(buggy_12["masse_g"].mean(), 2))'''

TODO_12 = r'''def load_12(text):
    """Read the French CSV export `text`: a DataFrame with the columns espece, ile, bec_mm, masse_g (in this order),
    one row per line of the file, bec_mm and masse_g as float64 (bills in mm, masses in GRAMS), missing values NaN."""
    raise NotImplementedError("load_12")'''

SOLUTION_12 = r'''def load_12(text):
    """Read the French CSV export `text`: a DataFrame with the columns espece, ile, bec_mm, masse_g (in this order),
    one row per line of the file, bec_mm and masse_g as float64 (bills in mm, masses in GRAMS), missing values NaN."""
    df = pd.read_csv(io.StringIO(text), sep=";", decimal=",", na_values=["?", "n.d."])
    df["masse_g"] = df["masse_t"] * 1_000_000                 # 1 tonne = 1,000 kg = 1,000,000 g
    return df[["espece", "ile", "bec_mm", "masse_g"]]

'''

CHECK_12 = r'''with wb.attempt("12.12"):
    loaded_12 = load_12(CSV_12)
    if returned("12.12", "load_12", loaded_12):
        if not isinstance(loaded_12, pd.DataFrame) or list(loaded_12.columns) != ["espece", "ile", "bec_mm", "masse_g"]:
            print("❌ Ex 12.12 : load_12 doit renvoyer un DataFrame aux colonnes espece, ile, bec_mm, masse_g, dans cet "
                  f"ordre ; reçu : {list(getattr(loaded_12, 'columns', [])) or type(loaded_12).__name__}.")
        else:
            print(loaded_12)
            numeric_12 = all(loaded_12[column].dtype == np.float64 for column in ["bec_mm", "masse_g"])
            verdict("12.12", numeric_12 and len(loaded_12) == 12, "12 lignes, et deux colonnes de nombres (float64).",
                    f"{len(loaded_12)} lignes (12 attendues) ; types : {loaded_12.dtypes.astype(str).to_dict()} : une "
                    "colonne de texte (object) signale une valeur que pandas n'a pas su lire : une marque de valeur absente "
                    "oubliée dans na_values ?")
            if numeric_12:
                wb.check("12.12a", float(loaded_12["bec_mm"].mean()), computed=True)
                wb.check("12.12b", float(loaded_12["masse_g"].mean()), computed=True)
                wb.check("12.12c", int(loaded_12.isna().sum().sum()), computed=True)'''

EXPERIMENT_13 = r'''from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cal_X_13 = wb.datasets.load_california().drop(columns="MedHouseVal")
names_13 = list(cal_X_13.columns)
out_cells_13, out_splits_13, medians_13 = [], np.zeros(len(names_13), dtype=int), []
for seed in range(200):                                    # 200 random 80/20 splits
    X_tr_13, X_te_13 = train_test_split(cal_X_13.to_numpy(), test_size=0.2, random_state=seed)
    T_13 = MinMaxScaler().fit(X_tr_13).transform(X_te_13)     # fit on the training part only
    outside_13 = (T_13 < 0) | (T_13 > 1)
    out_cells_13.append(outside_13.mean())
    out_splits_13 += outside_13.any(axis=0)
    medians_13.append(np.median(T_13[:, names_13.index("AveOccup")]))
print(f"a) share of the test cells outside [0, 1], mean over 200 splits: {np.mean(out_cells_13):.5%} "
      f"(about {np.mean(out_cells_13) * T_13.size:.1f} cells out of {T_13.size:,})")
print("b) number of splits (out of 200) where each feature leaves [0, 1] at least once:")
for letter, name, count in zip("ABCDEFGH", names_13, out_splits_13):
    print(f"   {letter}  {name:10s} {count:3d}")
print(f"c) median of the scaled AveOccup on the test part: {np.median(medians_13):.4f} (median over the splits)")
fig_13, ax_13 = plt.subplots(figsize=(10, 3.8))
ax_13.boxplot([T_13[:, j] for j in range(len(names_13))], whis=(0, 100))
ax_13.axhspan(0, 1, color="C2", alpha=0.12, label="[0, 1]")
ax_13.set_xticks(range(1, len(names_13) + 1), [f"{l}  {n}" for l, n in zip("ABCDEFGH", names_13)], rotation=20)
ax_13.set(ylabel="scaled value (test)", title="The test part of the last split after MinMaxScaler (whiskers: min and max)")
ax_13.legend(loc="upper left")
plt.show()
wb.check("12.13a", cells_out_13)
wb.check("12.13b", never_out_13)
wb.check("12.13c", aveoccup_13)'''

GIVEN_14 = r'''# A field log invented for the exercise: which team measured each island, each season
TEAMS_14 = pd.DataFrame({
    "Island": ["Biscoe", "Biscoe", "Biscoe", "Dream", "Dream", "Dream", "Torgersen", "Torgersen"],
    "studyName": ["PAL0708", "PAL0809", "PAL0910", "PAL0708", "PAL0809", "PAL0910", "PAL0708", "PAL0809"],
    "team": ["A", "A", "C", "B", "C", "B", "B", "B"],
})
TEAMS_14'''

TODO_14 = r'''dates_14 = ...         # a) the column "Date Egg" of RAW converted to dates (pd.to_datetime)
n_december_14 = ...    # a) number of rows whose egg was laid in December
spread_14 = ...        # b) days between the first and the last egg of each season PAL0708, PAL0809, PAL0910: a list of 3 ints
naive_rows_14 = ...    # c) number of rows of RAW.merge(TEAMS_14, on="Island")
no_team_14 = ...       # d) rows of a left join of RAW and TEAMS_14 on Island AND studyName that get no team
empty_cells_14 = ...   # e) empty cells of the table species x island of the mean body mass (pivot_table)'''

SOLUTION_14 = r'''dates_14 = pd.to_datetime(RAW["Date Egg"])
n_december_14 = int((dates_14.dt.month == 12).sum())
by_season_14 = dates_14.groupby(RAW["studyName"])
spread_14 = [int(days) for days in (by_season_14.max() - by_season_14.min()).dt.days]
naive_rows_14 = len(RAW.merge(TEAMS_14, on="Island"))
joined_14 = RAW.merge(TEAMS_14, on=["Island", "studyName"], how="left", validate="many_to_one")
no_team_14 = int(joined_14["team"].isna().sum())
pivot_14 = RAW.pivot_table(index="Species", columns="Island", values="Body Mass (g)", aggfunc="mean")
empty_cells_14 = int(pivot_14.isna().sum().sum())
print(n_december_14, spread_14, naive_rows_14, len(joined_14), no_team_14, empty_cells_14)
pivot_14.round(0)'''

CHECK_14 = r'''wb.check("12.14a", n_december_14)
wb.check("12.14b", spread_14)
wb.check("12.14c", naive_rows_14)
wb.check("12.14d", no_team_14)
wb.check("12.14e", empty_cells_14)'''

GIVEN_15 = r'''CON_15 = sqlite3.connect(":memory:")                     # a database in memory
PENGUINS_15 = pd.DataFrame({
    "study": RAW["studyName"], "species": RAW["Species"].str.split().str[0], "island": RAW["Island"],
    "sex": RAW["Sex"], "body_mass_g": RAW["Body Mass (g)"],
})
PENGUINS_15.to_sql("penguins", CON_15, index=False, if_exists="replace")
TEAMS_14.rename(columns={"Island": "island", "studyName": "study"}).to_sql("teams", CON_15, index=False,
                                                                          if_exists="replace")
print(pd.read_sql("SELECT * FROM penguins LIMIT 3", CON_15))
print(pd.read_sql("SELECT COUNT(*) AS n FROM teams", CON_15))'''

TODO_15 = r'''QUERY_15A = ...   # a) per island: the number of penguins whose sex is unknown; columns island, n; sorted by island
QUERY_15B = ...   # b) per team: the number of penguins it measured (join on island AND study); columns team, n; sorted by team
QUERY_15C = ...   # c) per species: the mean body mass of the FEMALE penguins; columns species, mean_mass; heaviest first'''

SOLUTION_15 = r'''QUERY_15A = """
SELECT island, COUNT(*) AS n
FROM penguins
WHERE sex IS NULL
GROUP BY island
ORDER BY island
"""
QUERY_15B = """
SELECT t.team, COUNT(*) AS n
FROM penguins AS p
JOIN teams AS t ON p.island = t.island AND p.study = t.study
GROUP BY t.team
ORDER BY t.team
"""
QUERY_15C = """
SELECT species, AVG(body_mass_g) AS mean_mass
FROM penguins
WHERE sex = 'FEMALE'
GROUP BY species
ORDER BY mean_mass DESC
"""
'''

CHECK_15 = r'''def run_query_15(ex_id, query, columns):
    """The result of a query as a DataFrame, or None with a ❌ message (the query fails or has other columns)."""
    if not filled(query):
        print(f"⏳ Ex {ex_id} : pas encore fait (écris la requête).")
        return None
    try:
        result = pd.read_sql(query, CON_15)
    except Exception as error:                            # an SQL syntax error, an unknown column...
        print(f"❌ Ex {ex_id} : la requête échoue : {error}")
        return None
    if list(result.columns) != columns:
        print(f"❌ Ex {ex_id} : colonnes attendues {columns}, reçues {list(result.columns)} (renomme avec AS).")
        return None
    print(result)
    return result


with wb.attempt("12.15"):
    result_15a = run_query_15("12.15a", QUERY_15A, ["island", "n"])
    if result_15a is not None:
        verdict("12.15a", result_15a["island"].tolist() == ["Biscoe", "Dream", "Torgersen"], "îles triées.",
                "les lignes doivent être triées par île, dans l'ordre alphabétique (ORDER BY island).")
        wb.check("12.15a", result_15a["n"].tolist(), computed=True)
        pandas_15a = PENGUINS_15[PENGUINS_15["sex"].isna()].groupby("island").size()
        verdict("12.15a", result_15a["n"].tolist() == pandas_15a.tolist(), "même résultat qu'avec pandas.",
                "ta requête ne donne pas les mêmes nombres que pandas (PENGUINS_15[PENGUINS_15[\"sex\"].isna()]"
                ".groupby(\"island\").size()) : compare les deux.")
with wb.attempt("12.15"):
    result_15b = run_query_15("12.15b", QUERY_15B, ["team", "n"])
    if result_15b is not None:
        verdict("12.15b", result_15b["team"].tolist() == ["A", "B", "C"], "équipes triées.",
                "les lignes doivent être triées par équipe (ORDER BY ...), une ligne par équipe.")
        wb.check("12.15b", result_15b["n"].tolist(), computed=True)
with wb.attempt("12.15"):
    result_15c = run_query_15("12.15c", QUERY_15C, ["species", "mean_mass"])
    if result_15c is not None:
        wb.check("12.15c", result_15c["mean_mass"].tolist(), computed=True)'''

PART_A = Part("A", "Diagnostiquer et assembler des données brutes",
              "Avant toute transformation, on regarde les données et on les assemble. Tu diagnostiques le fichier "
              "brut des manchots (12.11), tu répares la lecture d'un CSV écrit à la française (12.12), tu prédis où "
              "tombent les données de test de California après une normalisation min-max (12.13), puis tu assembles "
              "des tables avec pandas (12.14) et avec SQL (12.15). La cellule suivante définit les outils des parties "
              "A à D et charge `RAW`, le fichier brut des manchots. Ne recopie jamais un nombre à la main : chaque "
              "réponse est calculée par ton code, et pas arrondie.",
              given=PART_A_GIVEN, exercises=[
    Ex("12.11", "📦", 1, 15, "Diagnostic de Penguins brut avec pandas",
       "trouver dans un fichier brut ce qu'aucun résumé ne t'a dit, avec quelques lignes de pandas.",
       "0A (pandas : `nunique`, `groupby`, `isna`) · fiche §12.4 (cinq vérifications)", thread="Penguins",
       tracks="R, C",
       body=r"""`RAW` contient `penguins_raw.csv` tel quel : 344 lignes et 17 colonnes. La fiche, ⚖️ 12.10 et la data card ont déjà compté ses valeurs manquantes par colonne, par espèce et par saison, et signalé que `Individual ID` revient d'une saison à l'autre ; mesure maintenant ce qu'ils ne chiffrent pas. Calcule avec pandas :
a) `constant_11` : la liste des noms des colonnes qui ne prennent qu'**une seule** valeur (une valeur manquante compte comme une valeur) ;
b) `n_ids_11` : le nombre de valeurs distinctes de `Individual ID` ;
c) `multi_season_11` : le nombre d'identifiants qui apparaissent dans plus d'une saison (`studyName`) ;
d) `multi_species_11` : le nombre d'identifiants portés par des oiseaux de plus d'une espèce ;
e) `n15_only_11` : le nombre de lignes où δ15N (`Delta 15 N (o/oo)`) manque alors que δ13C est présent.

Dans tes notes : que t'apprennent c) et d) sur la façon de compter des oiseaux distincts dans ce fichier ? Que dit la colonne `Comments` de la ligne de e) ?""",
       todo=r'''constant_11 = ...       # a) names of the columns with a single value (NaN counts as a value)
n_ids_11 = ...          # b) distinct values of "Individual ID"
multi_season_11 = ...   # c) IDs seen in more than one season (studyName)
multi_species_11 = ...  # d) IDs worn by birds of more than one species
n15_only_11 = ...       # e) rows where Delta 15 N is missing but Delta 13 C is present''',
       check=CHECK_11, solution=SOLUTION_11,
       record=r'''wb.record("12.11a", constant_11, ordered=False,
          mistakes={"il en manque une : chaque colonne qui ne prend qu'une seule valeur compte-t-elle ?": ["Region"]})
wb.record("12.11b", n_ids_11, mistakes={"c'est le nombre de lignes : compte les valeurs distinctes (nunique)": 344,
                                        "c'est le nombre de lignes dont l'identifiant est déjà apparu (duplicated) : compte les valeurs distinctes": 154})
wb.record("12.11c", multi_season_11, mistakes={"tu comptes des lignes : compte des identifiants (un groupe par identifiant)": ''' + ROWS_11.format(col="studyName") + r'''})
wb.record("12.11d", multi_species_11, mistakes={"tu comptes des lignes : compte des identifiants (un groupe par identifiant)": ''' + ROWS_11.format(col="Species") + r''',
                                                "relis la colonne que tu regroupes : la question porte-t-elle sur les saisons ou sur les espèces (Species) ?": multi_season_11})
wb.record("12.11e", n15_only_11, mistakes={"c'est le nombre de lignes sans δ15N : retire celles où δ13C manque aussi": 14,
                                           "c'est le nombre de lignes sans δ13C : on cherche δ15N manquant ET δ13C présent": 13,
                                           "as-tu inversé les deux conditions ? δ15N doit manquer, δ13C être présent": 0})''',
       note="Deux colonnes sont constantes, `Region` (« Anvers ») et `Stage` (« Adult, 1 Egg Stage ») : elles "
            "n'apportent rien à un modèle (§12.6). Les 344 lignes ne portent que 190 identifiants, dont 114 "
            "reviennent d'une saison à l'autre et 94 désignent des oiseaux de plusieurs espèces : un identifiant "
            "désigne un adulte d'un nid (N1A1 et N1A2 sont les deux adultes du nid 1), et les numéros de nid sont "
            "réutilisés d'une saison et d'une espèce à l'autre : un même identifiant peut désigner des oiseaux "
            "différents. On ne compte "
            "donc pas des oiseaux distincts avec lui, et l'on ne s'en sert pas pour regrouper des mesures d'un même "
            "individu. Une "
            "seule ligne a δ13C sans δ15N ; son commentaire, « No delta15N data received from lab. », dit que la "
            "valeur manque pour une raison qui n'a rien à voir avec l'oiseau (MCAR, §12.4)."),

    Ex("12.12", "🐛", 1, 15, "CSV à la française : virgules décimales et « 7e-3 »",
       "lire correctement un CSV exporté par un tableur français, en partant des symptômes.",
       "Ex 12.11 · fiche §12.4 (types et formats ; encadré ⚠️ « NaN et 7e-3 »)", thread="Penguins", tracks="C",
       body=r"""Un laboratoire t'envoie `CSV_12`, l'export d'un tableur réglé en français : point-virgule entre les colonnes, virgule décimale, masses en **tonnes**, souvent en notation scientifique (« 3,75e-3 »), et deux marques de valeur absente. Le chargeur d'un collègue, `load_buggy_12`, ne plante pas, mais ses résultats sont faux : des becs de 391 mm à côté d'autres de 50 mm, des masses de 6 g à 375 g, sans aucune régularité.

Écris `load_12(text)`, qui renvoie un DataFrame aux colonnes `espece`, `ile`, `bec_mm`, `masse_g` (dans cet ordre), une ligne par ligne du fichier, `bec_mm` et `masse_g` en `float64` (masses en **grammes**), les valeurs absentes en `NaN`. La vérification contrôle :
a) la moyenne de `bec_mm` ;
b) la moyenne de `masse_g` ;
c) le nombre total de `NaN` du DataFrame.

Dans tes notes : explique chaque symptôme du chargeur du collègue (pourquoi 391 à côté de 50 ? pourquoi des masses sans régularité ?). Pourquoi `errors="coerce"` est-il dangereux ? « 6e-3 » posait-il un problème ?""",
       given=CSV_12, todo=TODO_12, check=CHECK_12, solution=SOLUTION_12 + CHECK_12.replace('with wb.attempt("12.12"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''reference_12 = load_12(CSV_12)
wb.record("12.12a", float(reference_12["bec_mm"].mean()), decimals=4,
          mistakes={"des becs de plusieurs centaines de mm : la virgule est le séparateur décimal, ne la supprime pas": float(load_buggy_12(CSV_12)["bec_mm"].mean())})
wb.record("12.12b", float(reference_12["masse_g"].mean()), decimals=4,
          mistakes={"une tonne vaut 1 000 kg, soit 1 000 000 g": float(reference_12["masse_g"].mean()) / 1000,
                    "tu as gardé des tonnes : convertis en grammes": float(reference_12["masse_g"].mean()) / 1e6,
                    "des masses sans régularité : la virgule est le séparateur décimal, ne la supprime pas": float(load_buggy_12(CSV_12)["masse_g"].mean())})
wb.record("12.12c", int(reference_12.isna().sum().sum()))''',
       note="Le collègue supprime les virgules au lieu de les lire comme des séparateurs décimaux : « 39,1 » devient "
            "391, mais « 50 » reste 50, d'où des becs dix fois trop grands à côté de becs justes. Pour les masses, "
            "« 3,75e-3 » devient « 375e-3 » (0,375 t) et « 3,8e-3 » « 38e-3 » (0,038 t) : le facteur d'erreur "
            "dépend du nombre de chiffres après la virgule, d'où l'absence de régularité ; « 6e-3 » sans virgule "
            "est lu juste (0,006 t). La conversion multiplie en plus par 1 000 au lieu de 1 000 000. Enfin "
            "`errors=\"coerce\"` transforme en silence tout ce qui ne se lit pas en `NaN` : ici « ? » et « n.d. », "
            "à raison, mais il cacherait aussi une vraie erreur de format. `pd.read_csv(..., sep=\";\", "
            "decimal=\",\", na_values=[\"?\", \"n.d.\"])` lit tout correctement, notation scientifique comprise : la "
            "crainte du livre pour « 7e-3 » ne se vérifie pas, le problème est la virgule."),

    Ex("12.13", "🔮", 1, 10, "Où tombent les données de test après un MinMaxScaler ?",
       "prévoir, avant de le mesurer, comment des données nouvelles se placent par rapport à $[0, 1]$.",
       "Ex 12.4 · fiche §12.5.1 et §12.5.3 · data card de California", thread="California", tracks="R, C",
       hypothesis=True,
       body=r"""On découpe 200 fois les 20 640 districts de California au hasard (80 % d'entraînement, 20 % de test). À chaque fois, un `MinMaxScaler` apprend le minimum et le maximum de chacune des 8 features sur l'entraînement, puis transforme le test. Tu sais déjà (fiche §12.5.3) que des valeurs de test **peuvent** sortir de $[0, 1]$ ; reste à savoir combien, lesquelles, et où tombent les autres. Les features : A `MedInc`, B `HouseAge`, C `AveRooms`, D `AveBedrms`, E `Population`, F `AveOccup`, G `Latitude`, H `Longitude`.

Prédis, **avant** d'exécuter quoi que ce soit :
a) `cells_out_13` : en moyenne sur les 200 découpages, la part des cases du test qui sortent de $[0, 1]$ : `"A"` environ 5 % ; `"B"` environ 0,5 % ; `"C"` environ 0,01 % ; `"D"` exactement 0 ;
b) `never_out_13` : les lettres des features qui ne sortent **jamais** de $[0, 1]$, sur aucun des 200 découpages (`""` si aucune) ;
c) `aveoccup_13` : la médiane des valeurs transformées d'`AveOccup` sur le test : `"A"` environ 0,5 ; `"B"` environ 0,1 ; `"C"` environ 0,01 ; `"D"` moins de 0,005.

Puis exécute l'expérience.""",
       todo=r'''cells_out_13 = ...   # a) "A", "B", "C" or "D"
never_out_13 = ...   # b) the letters of the features, e.g. "CG", or "" if none
aveoccup_13 = ...    # c) "A", "B", "C" or "D"''',
       solution=r'''cells_out_13, never_out_13, aveoccup_13 = "C", "AB", "D"   # the answers, for the record''',
       after=[("code", guarded(EXPERIMENT_13, ["cells_out_13", "never_out_13", "aveoccup_13"],
                               "⏳ Ex 12.13 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : compare avec tes prédictions. Pour chaque résultat, cherche la cause dans les "
                     "données (la data card de California aide). Le dépassement de $[0, 1]$ est-il un problème pour "
                     "la transformation ? Et pour un modèle ?")],
       record=r'''wb.record("12.13a", "C", mistakes={"un district de test sort de [0, 1] seulement s'il dépasse le plus extrême des 16 500 districts d'entraînement : est-ce fréquent ?": "A",
                                       "avec 16 500 districts d'entraînement, combien de districts de test peuvent dépasser leur extrême ?": "B",
                                       "il suffit qu'un district de test dépasse le plus extrême de l'entraînement : est-ce impossible ?": "D"})
wb.record("12.13b", "AB", ordered=False,
          mistakes={"une feature sort de [0, 1] quand son extrême tombe dans le test : relis la data card sur les valeurs plafonnées": ""})
wb.record("12.13c", "D", mistakes={"le maximum d'AveOccup vient d'un point aberrant (data card) : où sont alors les valeurs ordinaires ?": "A",
                                     "le maximum d'AveOccup dépasse 1 000 personnes par ménage (data card) : où tombe un ménage ordinaire ?": "B",
                                     "presque : divise un ménage ordinaire (environ 3 personnes) par l'étendue d'entraînement": "C"})''',
       note="Moins de trois cases sur 33 024 sortent de $[0, 1]$ en moyenne (environ 0,008 % : C). Une valeur de test "
            "ne sort que si elle dépasse le minimum ou le maximum de 16 500 districts, ce qui n'arrive qu'aux "
            "extrêmes des features à longue traîne (`AveRooms`, `AveBedrms`, `Population`, `AveOccup`) ou aux bords "
            "de la carte. `MedInc` et `HouseAge` ne sortent jamais : leurs valeurs sont **plafonnées** (un revenu de "
            "15, soit 150 000 $, et 52 ans, data card), avec 49 districts au plafond de `MedInc` et 1 273 à celui de "
            "`HouseAge`, si bien que l'entraînement en contient toujours. Côté minimum, `MedInc` a aussi un plancher (0,4999, 12 districts) ; "
            "`HouseAge` = 1 ne concerne que 4 districts, tous dans le test une fois sur 600 environ : ici, jamais "
            "sur ces 200 découpages. Enfin, le maximum d'`AveOccup` (1 243 personnes par ménage, une résidence collective) écrase "
            "les valeurs ordinaires : un ménage de 3 personnes tombe vers 0,002 (le district médian, 2,8 personnes, "
            "vers 0,0017). La normalisation min-max est "
            "intacte, mais inutile pour cette feature ; 📦 12.19 compare des mises à l'échelle robustes."),

    Ex("12.14", "📦", 2, 30, "pandas pour préparer : dates (« Date Egg »), jointure merge avec une table des îles, pivot_table",
       "convertir des dates, joindre deux tables en contrôlant les lignes et résumer par un tableau croisé.",
       "Ex 12.11 · fiche « Au-delà du livre » (encadré 🧮 : dates, jointures et tableaux croisés)",
       thread="Penguins", tracks="R, C",
       body=r"""`TEAMS_14` est un carnet de terrain **inventé pour l'exercice** : quelle équipe a mesuré quelle île, chaque saison. Calcule :
a) `dates_14` : la colonne `Date Egg` de `RAW` convertie en dates, puis `n_december_14`, le nombre de lignes dont l'œuf a été pondu en décembre ;
b) `spread_14` : pour chaque saison, dans l'ordre PAL0708, PAL0809, PAL0910, le nombre de jours entre le premier et le dernier œuf (une liste de 3 entiers) ;
c) `naive_rows_14` : le nombre de lignes de `RAW.merge(TEAMS_14, on="Island")` ;
d) `no_team_14` : le nombre de lignes qui ne reçoivent **aucune** équipe dans une jointure à gauche de `RAW` et `TEAMS_14` sur l'île **et** la saison (`studyName`) ;
e) `empty_cells_14` : le nombre de cases vides du tableau croisé espèce × île de la masse moyenne (`pivot_table`).

Dans tes notes : pourquoi c) donne-t-il beaucoup plus de 344 lignes ? Comment `validate="many_to_one"` t'aurait-il prévenu ? Que signifient les cases vides de e), et faut-il les remplir ?""",
       given=GIVEN_14, todo=TODO_14, check=CHECK_14, solution=SOLUTION_14,
       record=r'''wb.record("12.14a", n_december_14, mistakes={"c'est le nombre de lignes de novembre : on demande décembre (mois 12)": 330})
wb.record("12.14b", spread_14, mistakes={"le nombre de jours ENTRE le premier et le dernier œuf est leur différence, sans ajouter 1": [d + 1 for d in spread_14]})
wb.record("12.14c", naive_rows_14, mistakes={"c'est le nombre de lignes de RAW : la jointure sur l'île seule ne garde-t-elle vraiment qu'une équipe par ligne ?": 344})
wb.record("12.14d", no_team_14, mistakes={"aucune ligne sans équipe ? joins-tu bien à gauche (how=\"left\"), sur l'île ET la saison ?": 0})
wb.record("12.14e", empty_cells_14, mistakes={"compte les cases vides (NaN) du tableau entier, pas celles d'une seule ligne": 2})''',
       note="14 œufs pondus en décembre, les autres en novembre ; chaque saison de ponte dure 22 à 24 jours. "
            "La jointure sur l'île seule associe chaque manchot à **toutes** les lignes de son île dans le carnet "
            "(trois saisons pour Biscoe et Dream, deux pour Torgersen) : 168 × 3 + 124 × 3 + 52 × 2 = 980 lignes, "
            "des doublons silencieux. `validate=\"many_to_one\"` aurait levé une `MergeError`, car la clé se répète à "
            "droite. La bonne clé est le couple île-saison ; la jointure à gauche garde les 344 lignes, et les 16 "
            "manchots de Torgersen en 2009-10 n'ont pas d'équipe (le carnet n'a pas de ligne pour eux). Le tableau "
            "croisé a 4 cases vides : dans ce fichier, les Gentoo n'ont été mesurés qu'à Biscoe, et les Chinstrap qu'à Dream. Ces cases ne "
            "sont pas des valeurs manquantes à imputer, mais des combinaisons qui n'existent pas."),

    Ex("12.15", "📦", 2, 30, "Lire des données dans une base SQL : sqlite3, SELECT, JOIN, GROUP BY et pd.read_sql",
       "écrire des requêtes SQL qui filtrent, joignent et agrègent, et récupérer leur résultat dans pandas.",
       "Ex 12.14 · fiche « Au-delà du livre » (encadré 🧮 : SQL en six mots-clés)", thread="Penguins",
       tracks="R, C",
       body=r"""La cellule suivante crée une base SQLite en mémoire avec deux tables : `penguins` (colonnes `study`, `species`, `island`, `sex`, `body_mass_g` : une ligne par manchot de `RAW`, l'espèce réduite à son premier mot) et `teams` (le carnet de 12.14, colonnes `island`, `study`, `team`). Écris trois requêtes, sous forme de chaînes de caractères :
a) `QUERY_15A` : pour chaque île, le nombre de manchots dont le sexe est inconnu ; colonnes `island` et `n`, triées par île ;
b) `QUERY_15B` : pour chaque équipe, le nombre de manchots qu'elle a mesurés (jointure sur l'île **et** la saison) ; colonnes `team` et `n`, triées par équipe ;
c) `QUERY_15C` : pour chaque espèce, la masse moyenne des **femelles** (`sex = 'FEMALE'`) ; colonnes `species` et `mean_mass`, de la plus lourde à la plus légère.

La vérification exécute chaque requête avec `pd.read_sql` et contrôle la colonne des nombres.

Dans tes notes : pourquoi `WHERE sex = NULL` ne renvoie-t-il aucune ligne ? Combien de manchots la jointure de b) laisse-t-elle de côté, et pourquoi ? Écris la requête de c) en pandas, en une ligne.""",
       given=GIVEN_15, todo=TODO_15, check=CHECK_15, solution=SOLUTION_15 + CHECK_15.replace("with wb.attempt(\"12.15\"):\n", "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''wb.record("12.15a", [5, 1, 5], mistakes={"ce sont les manchots dont le sexe est CONNU : la condition s'écrit sex IS NULL": [163, 123, 47]})
naive_15b = pd.read_sql("SELECT t.team, COUNT(*) AS n FROM penguins AS p JOIN teams AS t ON p.island = t.island GROUP BY t.team ORDER BY t.team", CON_15)
wb.record("12.15b", [108, 126, 94], mistakes={"la jointure sur l'île seule multiplie les lignes : joins sur l'île ET la saison": naive_15b["n"].tolist()})
wb.record("12.15c", pd.read_sql(QUERY_15C, CON_15)["mean_mass"].tolist(), decimals=4,
          mistakes={"c'est l'ordre croissant : de la plus lourde à la plus légère, c'est ORDER BY ... DESC": pd.read_sql(QUERY_15C, CON_15)["mean_mass"].tolist()[::-1]})''',
       note="En SQL comme en pandas, une valeur absente (`NULL`) n'est égale à rien, pas même à elle-même : "
            "`sex = NULL` n'est jamais vrai, il faut `sex IS NULL`. Les 11 sexes inconnus se répartissent en 5 à "
            "Biscoe, 1 à Dream et 5 à Torgersen. La jointure de b) est interne : les 16 manchots de Torgersen en "
            "2009-10, sans ligne dans `teams`, disparaissent sans bruit (108 + 126 + 94 = 328) ; un `LEFT JOIN` les "
            "garderait avec une équipe `NULL`. En pandas, c) s'écrit "
            "`PENGUINS_15[PENGUINS_15.sex == \"FEMALE\"].groupby(\"species\")[\"body_mass_g\"].mean()"
            ".sort_values(ascending=False)`."),
])

# ---------------------------------------------------------------------------
# Part B: scalers and imputation (12.16 to 12.21)
# ---------------------------------------------------------------------------
PART_B_GIVEN = r'''# The data of part B: the 333 complete penguins (233 for training, 100 for testing) and California
from sklearn.impute import SimpleImputer as SkSimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler
from sklearn.preprocessing import StandardScaler as SkStandardScaler

PENGUINS = wb.datasets.load_penguins(dropna=True)
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
P_TRAIN, P_TEST = train_test_split(PENGUINS, test_size=100, random_state=1216, stratify=PENGUINS["species"])
CAL = wb.datasets.load_california()
CAL_FEATURES = [column for column in CAL.columns if column != "MedHouseVal"]
C_TRAIN, C_TEST = train_test_split(CAL, test_size=0.2, random_state=1217)
print(len(P_TRAIN), len(P_TEST), len(C_TRAIN), len(C_TEST))'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="12")   # reload your saved file\n'

MYLEARN_HOWTO = ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre "
                 "`mon_travail/mylearn/preprocessing.py` (créé par `python tools/start_chapter.py 12`), lis la "
                 "docstring de la classe et de chaque méthode, remplace les `raise NotImplementedError(...)` par ton "
                 "code et **enregistre**. NumPy seulement : scikit-learn est l'**oracle** des tests. La cellule de "
                 "vérification recharge ta librairie, compare quelques résultats à scikit-learn, puis lance les tests "
                 "de ta classe ; `python -m pytest tests/test_ch12_preprocessing.py -q` les lance tous dans un "
                 "terminal.")

MYLEARN_SHORT = ("> **mylearn** : dans `mon_travail/mylearn/preprocessing.py`, mêmes règles qu'en 12.16 (NumPy "
                 "seulement, scikit-learn pour oracle). Enregistre, puis relance la cellule de vérification.")

CHECK_16 = r'''with checked("12.16"):
    doc_16 = fitted(mylearn.preprocessing.StandardScaler(), np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]]))
    print("docstring example: mean_", doc_16.mean_, "· scale_", doc_16.scale_, "(expected [3. 10.] and [1.633 1.])")
with checked("12.16"):
    X_train_16, X_test_16 = P_TRAIN[MEASURES].to_numpy(), P_TEST[MEASURES].to_numpy()
    mine_16 = fitted(mylearn.preprocessing.StandardScaler(), X_train_16)
    oracle_16 = SkStandardScaler().fit(X_train_16)
    Z_train_16, Z_test_16 = mine_16.transform(X_train_16), mine_16.transform(X_test_16)
    print("training part after transform: means", np.round(Z_train_16.mean(axis=0), 6).tolist(),
          "· standard deviations", np.round(Z_train_16.std(axis=0), 6).tolist())
    print("test part after transform:     means", np.round(Z_test_16.mean(axis=0), 3).tolist(),
          "· standard deviations", np.round(Z_test_16.std(axis=0), 3).tolist())
    verdict("12.16", np.allclose(mine_16.mean_, oracle_16.mean_) and np.allclose(mine_16.scale_, oracle_16.scale_)
            and np.allclose(Z_test_16, oracle_16.transform(X_test_16))
            and np.allclose(mine_16.inverse_transform(Z_test_16), X_test_16),
            "mêmes paramètres et mêmes valeurs transformées que scikit-learn ; l'inverse redonne les mesures.",
            "tes paramètres ou tes valeurs transformées diffèrent de ceux de scikit-learn : les tests ci-dessous "
            "disent où.")
if written(lambda: mylearn.preprocessing.StandardScaler().fit(np.array([[0.0], [1.0]]))):
    run_mylearn_tests("test_standard_scaler_")
else:
    print("⏳ Ex 12.16 : les tests attendent le fit de ton StandardScaler.")'''

CHECK_17 = r'''with checked("12.17"):
    X_doc_17 = np.array([[-1.0, 2.0], [-0.5, 6.0], [0.0, 10.0], [1.0, 18.0]])
    print("docstring example: [2, 2] ->", fitted(mylearn.preprocessing.MinMaxScaler(), X_doc_17).transform(np.array([[2.0, 2.0]])),
          "(expected [[1.5 0. ]]) ; with clip=True ->",
          fitted(mylearn.preprocessing.MinMaxScaler(clip=True), X_doc_17).transform(np.array([[2.0, 2.0]])), "(expected [[1. 0.]])")
with checked("12.17"):
    X_train_17, X_test_17 = C_TRAIN[CAL_FEATURES].to_numpy(), C_TEST[CAL_FEATURES].to_numpy()
    mine_17 = fitted(mylearn.preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0)), X_train_17)
    oracle_17 = SkMinMaxScaler(feature_range=(-1.0, 1.0)).fit(X_train_17)
    T_17 = mine_17.transform(X_test_17)
    outside_17 = (T_17 < -1) | (T_17 > 1)
    print(f"California, towards [-1, 1]: {int(outside_17.sum())} test cell(s) outside [-1, 1], in the columns",
          [CAL_FEATURES[j] for j in np.flatnonzero(outside_17.any(axis=0))])
    clipped_17 = fitted(mylearn.preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0), clip=True), X_train_17)
    T_clip_17 = clipped_17.transform(X_test_17)
    print("with clip=True: smallest and largest value", round(float(T_clip_17.min()), 4), round(float(T_clip_17.max()), 4))
    verdict("12.17", np.allclose(T_17, oracle_17.transform(X_test_17)) and np.allclose(mine_17.inverse_transform(T_17), X_test_17)
            and T_clip_17.min() >= -1 and T_clip_17.max() <= 1,
            "mêmes valeurs que scikit-learn, inverse exacte, et clip=True borne bien les valeurs.",
            "tes valeurs diffèrent de celles de scikit-learn, ton inverse ne redonne pas les données, ou clip=True ne "
            "borne pas : les tests ci-dessous disent où.")
if written(lambda: mylearn.preprocessing.MinMaxScaler().fit(np.array([[0.0], [1.0]]))):
    run_mylearn_tests("test_min_max_scaler_")
else:
    print("⏳ Ex 12.17 : les tests attendent le fit de ton MinMaxScaler.")'''

CHECK_18 = r'''with checked("12.18"):
    doc_18 = fitted(mylearn.preprocessing.SimpleImputer(strategy="mean"), np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 8.0]]))
    print("docstring example: statistics_", doc_18.statistics_, "(expected [2. 6.])")
with checked("12.18"):
    raw_train_18, raw_test_18 = train_test_split(RAW, test_size=100, random_state=1218)
    iso_train_18 = raw_train_18[ISOTOPES_18].to_numpy()
    iso_test_18 = raw_test_18[ISOTOPES_18].to_numpy()
    before_18 = iso_test_18.copy()
    for strategy in ["mean", "median", "most_frequent"]:
        mine_18 = fitted(mylearn.preprocessing.SimpleImputer(strategy=strategy), iso_train_18)
        oracle_18 = SkSimpleImputer(strategy=strategy).fit(iso_train_18)
        filled_18 = mine_18.transform(iso_test_18)
        if not returned("12.18", "transform", filled_18):
            break
        print(f"{strategy:13s} statistics_ {np.round(mine_18.statistics_, 5).tolist()} · NaN left in the test part: "
              f"{int(np.isnan(filled_18).sum())}")
        verdict("12.18", np.allclose(mine_18.statistics_, oracle_18.statistics_) and np.allclose(filled_18, oracle_18.transform(iso_test_18)),
                f"stratégie {strategy} : comme scikit-learn.",
                f"stratégie {strategy} : statistics_ ou valeurs imputées différents de scikit-learn ({np.round(oracle_18.statistics_, 5).tolist()}).")
    verdict("12.18", np.array_equal(before_18, iso_test_18, equal_nan=True), "transform ne modifie pas le tableau reçu.",
            "transform a modifié le tableau reçu : travaille sur une copie.")
if written(lambda: mylearn.preprocessing.SimpleImputer().fit(np.array([[0.0], [np.nan], [1.0]]))):
    run_mylearn_tests("test_simple_imputer_")
else:
    print("⏳ Ex 12.18 : les tests attendent le fit de ton SimpleImputer.")'''

GIVEN_18 = r'''ISOTOPES_18 = ["Delta 15 N (o/oo)", "Delta 13 C (o/oo)"]   # the isotopes of RAW: 14 and 13 missing values'''

TODO_19 = r'''tukey_19 = ...        # a) training districts flagged by Tukey's rule for AveOccup
ratios_19 = ...       # b) IQR(MedInc) / IQR(AveOccup) on the transformed test part, for the 4 scalers of SCALERS_19 (a list)
robust_max_19 = ...   # c) the largest transformed AveOccup of the test part with RobustScaler
choice_19 = ...       # d) "A", "B", "C" or "D"'''

GIVEN_19 = r'''from sklearn.preprocessing import QuantileTransformer, RobustScaler

X_train_19 = C_TRAIN[["MedInc", "AveOccup"]].to_numpy()
X_test_19 = C_TEST[["MedInc", "AveOccup"]].to_numpy()
SCALERS_19 = {"A": SkMinMaxScaler(), "B": SkStandardScaler(), "C": RobustScaler(),
              "D": QuantileTransformer(random_state=0)}     # output: uniform on [0, 1]
print("AveOccup: median", np.median(X_train_19[:, 1]).round(2), "· max", X_train_19[:, 1].max().round(1))'''

SOLUTION_19 = r'''q1_19, q3_19 = np.percentile(X_train_19[:, 1], [25, 75])
iqr_19 = q3_19 - q1_19
tukey_19 = int(((X_train_19[:, 1] < q1_19 - 1.5 * iqr_19) | (X_train_19[:, 1] > q3_19 + 1.5 * iqr_19)).sum())
ratios_19, transformed_19 = [], {}
for letter, scaler in SCALERS_19.items():
    T = scaler.fit(X_train_19).transform(X_test_19)          # fit on the training part only
    transformed_19[letter] = T
    q1, q3 = np.percentile(T, [25, 75], axis=0)
    ratios_19.append(float((q3[0] - q1[0]) / (q3[1] - q1[1])))
robust_max_19 = float(transformed_19["C"][:, 1].max())
choice_19 = "D"
print(tukey_19, np.round(ratios_19, 3).tolist(), round(robust_max_19, 2))
fig_19, axes_19 = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (letter, T) in zip(axes_19, transformed_19.items()):
    ax.scatter(T[:, 0], T[:, 1], s=3, alpha=0.4)
    ax.set(title=f"{letter}: {type(SCALERS_19[letter]).__name__}", xlabel="MedInc (scaled)", ylabel="AveOccup (scaled)")
plt.tight_layout()
plt.show()'''

CHECK_19 = r'''wb.check("12.19a", tukey_19)
wb.check("12.19b", ratios_19)
wb.check("12.19c", robust_max_19)
wb.check("12.19d", choice_19)'''

GIVEN_20 = r'''# The colleague's model: species from the 4 measures (standardized) and the island (one-hot), trained on P_TRAIN
SCALER_20 = SkStandardScaler().fit(P_TRAIN[MEASURES])
ISLANDS_20 = pd.get_dummies(P_TRAIN["island"], dtype=float)
COLUMNS_20 = list(ISLANDS_20.columns)                     # the island columns seen at training
MODEL_20 = LogisticRegression(max_iter=1000).fit(np.hstack([SCALER_20.transform(P_TRAIN[MEASURES]), ISLANDS_20]),
                                                 P_TRAIN["species"])


def predict_buggy_20(batch):
    """The colleague's prediction function, used in production on each batch of new penguins."""
    X = SkStandardScaler().fit_transform(batch[MEASURES])
    islands = pd.get_dummies(batch["island"], dtype=float)
    return MODEL_20.predict(np.hstack([X, islands]))


GENTOO_20 = P_TEST[P_TEST["species"] == "Gentoo"]          # a batch of the day: only Gentoo, all from Biscoe
LONE_20 = P_TEST[(P_TEST["species"] == "Adelie") & (P_TEST["island"] == "Dream")].iloc[[0]]   # one penguin
THREE_20 = pd.concat([P_TEST[P_TEST["island"] == island].iloc[[0]] for island in ["Torgersen", "Dream", "Biscoe"]])
for name, batch in [("the 100 test penguins", P_TEST), ("the Gentoo batch", GENTOO_20), ("one penguin", LONE_20)]:
    try:
        accuracy = np.mean(predict_buggy_20(batch) == batch["species"].to_numpy())
        print(f"{name:22s} accuracy {accuracy:.3f}")
    except Exception as error:
        print(f"{name:22s} {type(error).__name__}: {error}")'''

TODO_20 = r'''def prepare_20(batch):
    """The matrix that MODEL_20 receives for the penguins of `batch` (a DataFrame with the columns of PENGUINS): a float
    array of shape (len(batch), 7), the 4 standardized measures, then the island columns of COLUMNS_20, in this order,
    with exactly the preparation of the training."""
    raise NotImplementedError("prepare_20")'''

SOLUTION_20 = r'''def prepare_20(batch):
    """The matrix that MODEL_20 receives for the penguins of `batch` (a DataFrame with the columns of PENGUINS): a float
    array of shape (len(batch), 7), the 4 standardized measures, then the island columns of COLUMNS_20, in this order,
    with exactly the preparation of the training."""
    X = SCALER_20.transform(batch[MEASURES])                               # the parameters learnt on P_TRAIN
    islands = pd.get_dummies(batch["island"], dtype=float).reindex(columns=COLUMNS_20, fill_value=0.0)
    return np.hstack([X, islands.to_numpy()])

'''

CHECK_20 = r'''def predict_20(batch):
    """The species predicted by MODEL_20 from YOUR preparation, or None with a ❌ message."""
    X = prepare_20(batch)
    if not returned("12.20", "prepare_20", X):
        return None
    X = np.asarray(X, dtype=float)
    if X.shape != (len(batch), 7):
        print(f"❌ Ex 12.20 : prepare_20 doit renvoyer une matrice ({len(batch)}, 7) : 4 mesures et 3 îles ; reçu {X.shape}.")
        return None
    return MODEL_20.predict(X)


with wb.attempt("12.20"):
    predicted_20 = predict_20(P_TEST)
    if predicted_20 is not None:
        test_20 = float(np.mean(predicted_20 == P_TEST["species"].to_numpy()))
        print(f"the 100 test penguins: accuracy {test_20:.3f}")
        wb.check("12.20a", test_20, computed=True)
with wb.attempt("12.20"):
    predicted_20 = predict_20(GENTOO_20)
    if predicted_20 is not None:
        gentoo_20 = float(np.mean(predicted_20 == GENTOO_20["species"].to_numpy()))
        print(f"the Gentoo batch: accuracy {gentoo_20:.3f}")
        wb.check("12.20b", gentoo_20, computed=True)
with wb.attempt("12.20"):
    predicted_20 = predict_20(LONE_20)
    if predicted_20 is not None:
        print("one penguin:", predicted_20)
        wb.check("12.20c", str(predicted_20[0]))
with wb.attempt("12.20"):
    three_20 = prepare_20(THREE_20)
    if returned("12.20", "prepare_20", three_20) and np.shape(three_20) == (3, 7):
        print("prepared matrix of THREE_20:\n", np.round(np.asarray(three_20, dtype=float), 4))
        wb.check("12.20d", np.asarray(three_20, dtype=float).tolist(), computed=True)
with wb.attempt("12.20"):
    shuffled_20 = P_TEST.sample(frac=1.0, random_state=0)[list(reversed(PENGUINS.columns))]   # other row and column order
    again_20 = prepare_20(shuffled_20)
    if returned("12.20", "prepare_20", again_20) and np.shape(again_20) == (len(P_TEST), 7):
        alone_20 = np.vstack([np.asarray(prepare_20(P_TEST.loc[[i]]), dtype=float) for i in shuffled_20.index[:5]])
        same_20 = np.allclose(np.asarray(again_20, dtype=float)[:5], alone_20)
        verdict("12.20", same_20, "la préparation d'un manchot ne dépend ni des autres manchots du batch, ni de l'ordre des colonnes.",
                "la préparation d'un manchot change avec les autres lignes du batch ou l'ordre de ses colonnes : choisis "
                "les colonnes par leur nom, et ne réapprends rien sur le batch.")'''

GIVEN_21 = r'''from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline

rng_21 = np.random.default_rng(1221)
T_21 = rng_21.uniform(-8.0, 16.0, size=120)                     # midnight temperature (°C), 120 mornings
CARS_21 = 1400.0 + 45.0 * T_21 + rng_21.normal(0.0, 120.0, 120)   # cars counted the next morning (invented)
T_NEW_21 = np.array([-2.0, 5.0, 25.0])                          # three new nights
print(f"temperatures {T_21.min():.1f} to {T_21.max():.1f} °C · cars {CARS_21.min():.0f} to {CARS_21.max():.0f}")'''

TODO_21 = r'''def fit_predict_21(T_train, cars_train, T_new):
    """Fit a MinMaxScaler on the temperatures, another one on the cars, and Ridge(alpha=1.0) on the scaled values;
    return the predicted numbers of cars for T_new (an array of shape (len(T_new),), in cars)."""
    raise NotImplementedError("fit_predict_21")


ttr_21 = ...               # b) the same model as one TransformedTargetRegressor (not fitted)
scaled_target_21 = ...     # c) the prediction for T_NEW_21[0] in the scaled units of the target (a number)
outside_21 = ...           # d) how many predictions of a) fall outside the range of CARS_21
target_scaler_matters_21 = ...   # e) True or False: does the scaler of the target change the predictions in cars?'''

SOLUTION_21 = r'''def fit_predict_21(T_train, cars_train, T_new):
    """Fit a MinMaxScaler on the temperatures, another one on the cars, and Ridge(alpha=1.0) on the scaled values;
    return the predicted numbers of cars for T_new (an array of shape (len(T_new),), in cars)."""
    temp_scaler = SkMinMaxScaler().fit(np.reshape(T_train, (-1, 1)))
    cars_scaler = SkMinMaxScaler().fit(np.reshape(cars_train, (-1, 1)))
    model = Ridge(alpha=1.0).fit(temp_scaler.transform(np.reshape(T_train, (-1, 1))),
                                 cars_scaler.transform(np.reshape(cars_train, (-1, 1))).ravel())
    z_hat = model.predict(temp_scaler.transform(np.reshape(T_new, (-1, 1))))
    return cars_scaler.inverse_transform(z_hat.reshape(-1, 1)).ravel()       # the TARGET's transformer


ttr_21 = TransformedTargetRegressor(regressor=make_pipeline(SkMinMaxScaler(), Ridge(alpha=1.0)),
                                    transformer=SkMinMaxScaler())
cars_scaler_21 = SkMinMaxScaler().fit(CARS_21.reshape(-1, 1))
scaled_target_21 = float(cars_scaler_21.transform(fit_predict_21(T_21, CARS_21, T_NEW_21[:1]).reshape(-1, 1))[0, 0])
predicted_21 = fit_predict_21(T_21, CARS_21, T_NEW_21)
outside_21 = int(((predicted_21 < CARS_21.min()) | (predicted_21 > CARS_21.max())).sum())
plain_21 = make_pipeline(SkMinMaxScaler(), Ridge(alpha=1.0)).fit(T_21.reshape(-1, 1), CARS_21).predict(T_NEW_21.reshape(-1, 1))
target_scaler_matters_21 = not np.allclose(plain_21, predicted_21)
print(np.round(predicted_21, 2), np.round(plain_21, 2), scaled_target_21, outside_21, target_scaler_matters_21)
'''

CHECK_21 = r'''wb.check("12.21c", scaled_target_21)
wb.check("12.21d", outside_21)
wb.check("12.21e", target_scaler_matters_21)
with wb.attempt("12.21"):
    if not filled(ttr_21):
        print("⏳ Ex 12.21b : pas encore fait (remplace `...` par ton TransformedTargetRegressor).")
    elif not hasattr(ttr_21, "fit"):
        print("❌ Ex 12.21b : ttr_21 doit être un TransformedTargetRegressor (non entraîné).")
    else:
        via_ttr_21 = ttr_21.fit(T_21.reshape(-1, 1), CARS_21).predict(T_NEW_21.reshape(-1, 1))
        print("TransformedTargetRegressor:", np.round(via_ttr_21, 1).tolist())
        wb.check("12.21b", via_ttr_21.tolist(), computed=True)
with wb.attempt("12.21"):
    predicted_21 = fit_predict_21(T_21, CARS_21, T_NEW_21)
    if returned("12.21", "fit_predict_21", predicted_21):
        try:
            predicted_21 = np.asarray(predicted_21, dtype=float).ravel()
        except (TypeError, ValueError):
            predicted_21 = np.array([])
            print("❌ Ex 12.21 : fit_predict_21 doit renvoyer des nombres, une prédiction par température.")
        if predicted_21.size == 0:
            pass
        elif predicted_21.shape != (3,):
            print(f"❌ Ex 12.21 : fit_predict_21 doit renvoyer une prédiction par température de T_new, forme (3,) ; reçu {predicted_21.shape}.")
        else:
            print("predicted cars at -2, 5 and 25 °C:", np.round(predicted_21, 1).tolist())
            wb.check("12.21a", predicted_21.tolist(), computed=True)
            grid_21 = np.linspace(-15, 30, 200)
            curve_21 = np.asarray(fit_predict_21(T_21, CARS_21, grid_21), dtype=float).ravel()
            plt.scatter(T_21, CARS_21, s=12, label="training mornings")
            plt.plot(grid_21, curve_21, color="C1", label="fit_predict_21")
            plt.axvspan(T_21.min(), T_21.max(), color="C2", alpha=0.08, label="training temperatures")
            plt.xlabel("midnight temperature (°C)")
            plt.ylabel("cars the next morning")
            plt.legend()
            plt.show()'''

PART_B = Part("B", "Mettre à l'échelle et imputer",
              "Tu écris trois classes de `mylearn/preprocessing.py` (`StandardScaler`, `MinMaxScaler`, "
              "`SimpleImputer`), puis tu t'en sers avec celles de scikit-learn : des points aberrants (12.19), un "
              "modèle déployé qui prépare mal ses données (12.20), une cible transformée (12.21). La cellule suivante "
              "charge les 333 manchots complets (233 pour l'entraînement, 100 pour le test) et California (80/20).",
              given=PART_B_GIVEN, exercises=[
    Ex("12.16", "🔨", 2, 25, "Coder StandardScaler (fit, transform, inverse)",
       "écrire un transformateur qui apprend ses paramètres une fois et les réapplique, et le vérifier contre "
       "scikit-learn.",
       "Ex 12.2 · ∂ 12.6 · 0A (classes Python) · fiche §12.5.2 et §12.5.3 (encadré 🧮 : les transformateurs)",
       thread="Penguins", tracks="R, M, C", mylearn="preprocessing.py",
       body=MYLEARN_HOWTO + r"""

Écris la classe `StandardScaler`. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__(with_mean=True, with_std=True)` range ses deux arguments tels quels, sans les convertir ni rien calculer ;
- `fit(X)` apprend `mean_`, `var_` (ddof = 0), `scale_`, `n_features_in_` et `n_samples_seen_`, puis renvoie `self` ; il ne modifie ni `X` ni un hyperparamètre, et n'ajoute que des attributs dont le nom finit par `_` ou commence par `_` ; `mean_` et `var_` sont calculés même avec `with_mean=False` ou `with_std=False` ;
- une colonne **constante** se repère exactement (son maximum égal à son minimum) : son `var_` vaut alors 0 et son `scale_` 1 ; avec `with_std=False`, `scale_` vaut 1 partout ;
- `transform(X)` rend un tableau `float64` sans modifier `X` ; `fit_transform(X)` apprend comme `fit`, puis transforme ; `inverse_transform(Z)` fait le chemin inverse ;
- erreurs : `ValueError` si `X` n'est pas 2-D, contient `NaN` ou une valeur infinie, ou n'a pas le nombre de colonnes appris ; même chose pour `Z` dans `inverse_transform` (2-D, bon nombre de colonnes) ; `RuntimeError` si `transform` ou `inverse_transform` est appelé avant `fit`.

La vérification essaie l'exemple de la docstring, standardise les manchots (`fit` sur les 233 d'entraînement, `transform` sur les 100 de test) en comparant à scikit-learn, puis lance les tests.

Dans tes notes : pourquoi les moyennes du test transformé ne valent-elles pas exactement 0 ? Pourquoi repérer une colonne constante par `max == min` plutôt que par `var_ == 0` ?""",
       check=RELOAD + CHECK_16, solution=CHECK_16.replace('with wb.attempt("12.16"):\n', "if True:\n").replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", keyword="'),
       note="Le test transformé a des moyennes proches de 0 sans y être (de l'ordre de ±0,1) : ses paramètres sont "
            "ceux de l'entraînement, et c'est voulu (§12.5.3). Une colonne constante a une variance mathématiquement "
            "nulle, mais la variance **calculée** en virgule flottante peut valoir $2 \\times 10^{-34}$ (sept valeurs "
            "0,1, fiche §12.5.2) : comparée à 0, elle passerait pour une vraie variance, et la division par sa "
            "racine, $10^{-17}$, ferait exploser le bruit d'arrondi. Comparer le maximum et le minimum est exact."),

    Ex("12.17", "🔨", 2, 20, "Coder MinMaxScaler avec feature_range et clip",
       "écrire la normalisation min-max vers un intervalle quelconque, avec et sans bornage des données nouvelles.",
       "Ex 12.16 · Ex 12.4 · fiche §12.5.1 et §12.8", thread="California", tracks="R, C", mylearn="preprocessing.py",
       body=MYLEARN_SHORT + r"""

Écris la classe `MinMaxScaler`. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__(feature_range=(0.0, 1.0), clip=False)` range ses arguments tels quels ; `fit` lève une `ValueError` si `feature_range[0] >= feature_range[1]` ;
- `fit(X)` apprend `data_min_`, `data_max_`, `data_range_`, `scale_`, `min_` et `n_features_in_`, et renvoie `self` (mêmes règles qu'en 12.16 sur les attributs) ; une colonne constante garde `data_range_ = 0`, mais `scale_` est calculé comme si l'étendue valait 1 ;
- `transform(X)` rend `X * scale_ + min_` en `float64`, borné à `feature_range` si `clip=True` ; il lève une `ValueError` pour un `X` 1-D, contenant une valeur infinie ou de mauvais nombre de colonnes (pour un `NaN` dans `transform`, rien n'est exigé : scikit-learn le laisse passer ; `fit`, lui, refuse les `NaN`, comme le dit sa docstring) ;
- `inverse_transform(Z)` rend `(Z - min_) / scale_` et ne borne **jamais**, même avec `clip=True` ;
- `RuntimeError` avant `fit`, comme en 12.16.

La vérification essaie les exemples de la docstring, normalise California vers $[-1, 1]$ (`fit` sur l'entraînement, `transform` sur le test), compte les cases qui sortent de l'intervalle, essaie `clip=True`, puis lance les tests.

Dans tes notes : avec `clip=True`, `inverse_transform(transform(X))` redonne-t-il `X` ? Pourquoi l'inverse ne doit-il pas borner ?""",
       check=RELOAD + CHECK_17, solution=CHECK_17.replace('with wb.attempt("12.17"):\n', "if True:\n").replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", keyword="'),
       note="Vers $[a, b]$, `scale_` vaut $(b - a)/(\\max - \\min)$ et `min_` vaut $a - \\min \\times$ `scale_` : "
            "le minimum d'entraînement va en $a$, le maximum en $b$. Avec `clip=True`, une valeur bornée a perdu "
            "l'information de son dépassement : `inverse_transform` rend la borne d'origine (le minimum ou le maximum "
            "d'entraînement), pas la valeur vraie. Borner aussi dans l'inverse ferait mentir la transformation sur "
            "des valeurs que l'utilisateur fournit lui-même dans l'espace transformé, comme la prédiction d'un modèle "
            "(12.21) : scikit-learn ne le fait pas."),

    Ex("12.18", "🔨", 2, 25, "Coder SimpleImputer (moyenne, médiane, mode, constante)",
       "remplacer les valeurs manquantes par une statistique apprise sur l'entraînement, colonne par colonne.",
       "Ex 12.16 · Ex 12.11 · fiche §12.4 (supprimer ou imputer ?)", thread="Penguins", tracks="C",
       mylearn="preprocessing.py",
       body=MYLEARN_SHORT + r"""

Écris la classe `SimpleImputer`, pour des tableaux de nombres où `np.nan` marque une valeur manquante. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__(strategy="mean", fill_value=None)` range ses arguments tels quels ; `fit` lève une `ValueError` pour une stratégie inconnue ;
- `fit(X)` apprend `statistics_` (une valeur par colonne, calculée sur ses valeurs **observées**) et `n_features_in_`, et renvoie `self` : `"mean"` la moyenne ; `"median"` la médiane (pour un nombre pair de valeurs, la moyenne des deux du milieu) ; `"most_frequent"` la valeur la plus fréquente, la plus petite en cas d'égalité ; `"constant"` la valeur `fill_value` (`0.0` si elle vaut `None`) ;
- une colonne **entièrement** manquante lève une `ValueError` avec `"mean"`, `"median"` et `"most_frequent"` (scikit-learn la supprimerait, avec un simple avertissement) ; avec `"constant"`, elle est acceptée ;
- `transform(X)` remplace chaque `NaN` de la colonne $j$ par `statistics_[j]`, sans modifier `X`, et rend un tableau `float64` ; `ValueError` pour un `X` 1-D ou de mauvais nombre de colonnes ; `RuntimeError` avant `fit` ;
- `fit_transform(X)` apprend, puis transforme. Les valeurs infinies ne sont pas testées.

La vérification essaie l'exemple de la docstring, impute les isotopes de `RAW` (`fit` sur 244 lignes, `transform` sur 100 autres) avec trois stratégies, en comparant à scikit-learn, puis lance les tests.

Dans tes notes : pour δ15N, la moyenne et la médiane diffèrent-elles beaucoup ? Imputer la moyenne change-t-il la moyenne de la colonne ? Et son écart-type ?""",
       given=GIVEN_18, check=RELOAD + CHECK_18, solution=CHECK_18.replace('with wb.attempt("12.18"):\n', "if True:\n").replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", keyword="'),
       note="`np.nanmean` et `np.nanmedian` calculent sur les valeurs observées ; pour la valeur la plus fréquente, "
            "`np.unique(colonne_observée, return_counts=True)` donne les valeurs triées et leurs effectifs, et "
            "`np.argmax` des effectifs renvoie le **premier** maximum, donc la plus petite valeur parmi les ex aequo. "
            "Sur des mesures continues comme les isotopes, presque toutes les valeurs sont uniques : la « plus "
            "fréquente » est alors la plus petite observée, un mauvais choix pour un nombre. Imputer la moyenne ne "
            "change pas la moyenne des valeurs observées, mais diminue l'écart-type (des valeurs ajoutées pile au "
            "centre) : la colonne paraît plus sûre qu'elle n'est (§12.4)."),

    Ex("12.19", "📦", 2, 25, "Scalers de scikit-learn face aux points aberrants de California",
       "repérer des points aberrants, et comparer quatre mises à l'échelle sur des features qui en contiennent.",
       "Ex 12.17 · fiche §12.4 (règle de Tukey), §12.5 et encadré 🕰️ « des mises à l'échelle robustes »",
       thread="California", tracks="R, C",
       body=r"""On reprend deux features de California, le revenu médian `MedInc` et l'occupation moyenne des logements `AveOccup` (le découpage 80/20 de la partie B). `SCALERS_19` contient quatre transformateurs de scikit-learn : A `MinMaxScaler`, B `StandardScaler`, C `RobustScaler` (médiane et écart interquartile), D `QuantileTransformer` (rangs, sortie uniforme sur $[0, 1]$). Calcule :
a) `tukey_19` : le nombre de districts **d'entraînement** que la règle de Tukey signale pour `AveOccup` (quartiles de `np.percentile`) ;
b) `ratios_19` : pour chacun des quatre transformateurs, dans l'ordre A, B, C, D, ajusté sur l'entraînement et appliqué au test, le rapport de l'écart interquartile de `MedInc` transformé à celui d'`AveOccup` transformé (une liste de 4 nombres) : un rapport proche de 1 veut dire que les deux features occupent des étendues comparables ;
c) `robust_max_19` : la plus grande valeur d'`AveOccup` transformée par `RobustScaler` sur le test ;
d) `choice_19` : la lettre du transformateur qui donne aux deux features des étendues comparables **et** garde toutes les valeurs du test entre deux bornes fixes.

Trace aussi, pour chaque transformateur, le nuage du test transformé (`MedInc` en abscisse, `AveOccup` en ordonnée).

Dans tes notes : faut-il supprimer les districts signalés en a) ? Que garde `RobustScaler` du point aberrant, et que perd `QuantileTransformer` ?""",
       given=GIVEN_19, todo=TODO_19, check=CHECK_19, solution=SOLUTION_19,
       record=r'''wb.record("12.19a", tukey_19, mistakes={"compte les deux côtés de l'intervalle de Tukey, au-dessus et en dessous": int((X_train_19[:, 1] > q3_19 + 1.5 * iqr_19).sum())})
def refit_ratio_19(scaler):
    T = scaler.fit_transform(X_test_19)                     # the classic error: fitted on the test part
    q1, q3 = np.percentile(T, [25, 75], axis=0)
    return float((q3[0] - q1[0]) / (q3[1] - q1[1]))


wb.record("12.19b", ratios_19, decimals=1,
          mistakes={"c'est le rapport inverse : l'écart interquartile de MedInc divisé par celui d'AveOccup": [1 / r for r in ratios_19],
                    "les transformateurs s'ajustent sur l'entraînement, puis s'appliquent au test": [refit_ratio_19(sc) for sc in SCALERS_19.values()]})
wb.record("12.19c", robust_max_19, decimals=2,
          mistakes={"c'est la plus grande valeur d'entraînement : transforme le TEST": float(RobustScaler().fit(X_train_19).transform(X_train_19)[:, 1].max()),
                    "RobustScaler s'ajuste sur l'entraînement, puis s'applique au test": float(RobustScaler().fit_transform(X_test_19)[:, 1].max())})
wb.record("12.19d", choice_19, mistakes={"le min-max laisse les valeurs ordinaires d'AveOccup écrasées près de 0 : relis le rapport de b)": "A",
                                         "la standardisation ne borne rien et laisse AveOccup bien plus étroite que MedInc": "B",
                                         "les étendues sont comparables, mais les valeurs du test sont-elles bornées ? relis c)": "C"})''',
       note="La règle de Tukey signale plusieurs centaines de districts d'entraînement pour `AveOccup` : une longue "
            "traîne, pas quelques erreurs. Les supprimer serait une erreur : ce sont de vrais districts (résidences "
            "étudiantes, foyers), que le modèle rencontrera en service ; on choisit plutôt une mise à l'échelle qui "
            "les supporte. Après le min-max, l'étendue interquartile de `MedInc` est environ 200 fois celle "
            "d'`AveOccup` ; après la standardisation, environ 15 fois ; après `RobustScaler` et "
            "`QuantileTransformer`, à peu près la même. Mais `RobustScaler` ne borne rien : le district le plus "
            "extrême du test reste à 267 écarts interquartiles de la médiane. `QuantileTransformer` envoie "
            "chaque valeur sur son rang : tout tombe dans $[0, 1]$ (D), au prix des écarts eux-mêmes (un ménage de "
            "500 personnes et un de 20 deviennent presque égaux, tout en haut)."),

    Ex("12.20", "🐛", 2, 20, "Le scaler réentraîné sur le test et les get_dummies désalignés",
       "réparer la préparation d'un modèle déployé en partant de ses symptômes.",
       "Ex 12.16 · Ex 12.1 · fiche §12.2, §12.3.1 et §12.5.3", thread="Penguins", tracks="R, C",
       body=r"""Un collègue a entraîné `MODEL_20`, une régression logistique qui prédit l'espèce à partir des quatre mesures standardisées et de l'île en one-hot, sur les 233 manchots d'entraînement. En production, il prépare chaque **batch** de nouveaux manchots avec `predict_buggy_20`. La cellule suivante montre les symptômes : les 100 manchots de test passent, mais le batch du jour (des Gentoo, tous de Biscoe) et un manchot seul font planter la fonction.

Écris `prepare_20(batch)`, qui renvoie la matrice que reçoit `MODEL_20` pour les manchots du batch : un tableau de nombres de forme `(len(batch), 7)`, les quatre mesures standardisées puis les colonnes d'îles de `COLUMNS_20`, dans cet ordre, avec **exactement** la préparation de l'entraînement (elle est dans la cellule suivante : ne la modifie pas, et ne réentraîne rien). La vérification prédit avec `MODEL_20.predict(prepare_20(batch))` et contrôle :
a) l'accuracy sur les 100 manchots de test ;
b) l'accuracy sur le batch de Gentoo ;
c) l'espèce prédite pour le manchot seul (une chaîne) ;
d) la matrice préparée de `THREE_20`, trois manchots de test (un de Torgersen, un de Dream, un de Biscoe) ;
puis que la préparation d'un manchot ne dépend ni des autres manchots du batch, ni de l'ordre de ses lignes et de ses colonnes.

Dans tes notes : pourquoi la fonction du collègue passait-elle sur les 100 manchots de test ? Si l'on corrige seulement les colonnes d'îles, que devient l'accuracy sur le batch de Gentoo, et pourquoi ?""",
       given=GIVEN_20, todo=TODO_20, check=CHECK_20, solution=SOLUTION_20 + CHECK_20.replace('with wb.attempt("12.20"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''def half_fixed_20(batch):
    """Only the island columns fixed: the scaler is still refitted on the batch."""
    X = SkStandardScaler().fit_transform(batch[MEASURES])
    islands = pd.get_dummies(batch["island"], dtype=float).reindex(columns=COLUMNS_20, fill_value=0.0)
    return MODEL_20.predict(np.hstack([X, islands.to_numpy()]))


wb.record("12.20a", float(np.mean(predict_20(P_TEST) == P_TEST["species"].to_numpy())), decimals=4)
wb.record("12.20b", float(np.mean(predict_20(GENTOO_20) == GENTOO_20["species"].to_numpy())), decimals=4,
          mistakes={"le scaler est réappris sur le batch : ses Gentoo, centrés sur eux-mêmes, ressemblent à des manchots moyens": float(np.mean(half_fixed_20(GENTOO_20) == GENTOO_20["species"].to_numpy()))})
wb.record("12.20c", str(predict_20(LONE_20)[0]), mistakes={"le scaler est réappris sur un seul manchot : ses mesures deviennent toutes 0": str(half_fixed_20(LONE_20)[0])})
right_20 = prepare_20(THREE_20)
reversed_20 = np.hstack([right_20[:, :4], right_20[:, 4:][:, ::-1]])
ddof1_20 = np.hstack([right_20[:, :4] * np.std(P_TRAIN[MEASURES].to_numpy(), axis=0) / np.std(P_TRAIN[MEASURES].to_numpy(), axis=0, ddof=1),
                      right_20[:, 4:]])
all_20 = np.hstack([SkStandardScaler().fit(PENGUINS[MEASURES]).transform(THREE_20[MEASURES]), right_20[:, 4:]])
wb.record("12.20d", right_20.tolist(), decimals=6,
          mistakes={"les colonnes d'îles ne sont pas dans l'ordre de COLUMNS_20": reversed_20.tolist(),
                    "les écarts-types diffèrent un peu : StandardScaler divise par n (ddof = 0), pandas .std() par n − 1": ddof1_20.tolist(),
                    "les mesures sont standardisées avec des paramètres appris sur tous les manchots : seul P_TRAIN compte": all_20.tolist()})''',
       note="Deux bugs. (1) `StandardScaler().fit_transform(batch[...])` réapprend la moyenne et l'écart-type sur le "
            "batch : un batch de Gentoo, centré sur lui-même, ressemble à un batch de manchots moyens, et ses mesures ne "
            "veulent plus rien dire pour le modèle ; un manchot seul est transformé en zéros. Il faut "
            "`SCALER_20.transform`, avec les paramètres de l'entraînement. (2) `pd.get_dummies` sur le batch ne crée "
            "que les colonnes des îles présentes : un batch d'une seule île donne une colonne au lieu de trois, d'où "
            "l'erreur de forme ; un batch de deux îles en donnerait deux, décalées. On réaligne sur les colonnes de "
            "l'entraînement (`reindex(columns=COLUMNS_20, fill_value=0)`), ou mieux, on garde un `OneHotEncoder` "
            "appris à l'entraînement. Sur les 100 manchots de test, les trois espèces et les trois îles sont "
            "présentes et les moyennes du batch ressemblent à celles de l'entraînement : les deux bugs se cachent. "
            "Corriger seulement les îles laisse le premier bug : les Gentoo, recentrés sur eux-mêmes, sont alors "
            "classés en partie comme Adélie."),

    Ex("12.21", "📦", 2, 25, "Trafic et température : transformer la cible et revenir aux voitures",
       "entraîner un modèle sur une cible transformée et ramener ses prédictions dans les unités d'origine.",
       "Ex 12.17 · ch. 9 (Ridge) · fiche §12.8", thread="synthétique", tracks="C",
       body=r"""Des données **inventées** : 120 matins, la température de la nuit précédente (de −8 à 16 °C) et le nombre de voitures comptées le matin. Écris :
- a) `fit_predict_21(T_train, cars_train, T_new)` : un `MinMaxScaler` pour la température, un autre pour les voitures, `Ridge(alpha=1.0)` entraîné sur les valeurs transformées ; la fonction renvoie les nombres de voitures prédits pour `T_new`, **en voitures** (un tableau de forme `(len(T_new),)`) ;
- b) `ttr_21` : le même modèle en un seul objet, un `TransformedTargetRegressor` (non entraîné), dont le régresseur est un pipeline `MinMaxScaler` puis `Ridge(alpha=1.0)` et le transformateur un `MinMaxScaler` ;
- c) `scaled_target_21` : la prédiction pour −2 °C, dans les unités **transformées** de la cible ;
- d) `outside_21` : le nombre de prédictions de a), parmi les trois nuits de `T_NEW_21`, qui tombent hors de l'étendue des voitures d'entraînement ;
- e) `target_scaler_matters_21` : le transformateur de la **cible** change-t-il les prédictions en voitures d'une Ridge ? (`True` ou `False` ; compare avec le même pipeline entraîné sur les voitures brutes).

La vérification contrôle c), d), e), b) les prédictions de `ttr_21`, puis a) celles de `fit_predict_21` pour −2, 5 et 25 °C, et trace le modèle.

Dans tes notes : que donnerait la prédiction de −2 °C ramenée avec le transformateur de la **température** ? Explique le résultat de e) : pourquoi le transformateur de la cible change-t-il, ou non, les prédictions, et celui de la feature ? Que penses-tu de la prédiction pour 25 °C ?""",
       given=GIVEN_21, todo=TODO_21, check=CHECK_21, solution=SOLUTION_21 + CHECK_21.replace('with wb.attempt("12.21"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''temp_scaler_21 = SkMinMaxScaler().fit(T_21.reshape(-1, 1))
wrong_21 = temp_scaler_21.inverse_transform(np.array([[scaled_target_21]]))[0, 0]
wb.record("12.21a", predicted_21.tolist(), decimals=4,
          mistakes={"ces nombres sont dans les unités transformées de la cible : ramène-les en voitures avec le transformateur des voitures": cars_scaler_21.transform(predicted_21.reshape(-1, 1)).ravel().tolist()})
wb.record("12.21b", ttr_21.fit(T_21.reshape(-1, 1), CARS_21).predict(T_NEW_21.reshape(-1, 1)).tolist(), decimals=4)
wb.record("12.21c", scaled_target_21, decimals=4,
          mistakes={"ce nombre est en voitures : on demande la prédiction dans les unités transformées de la cible": float(predicted_21[0])})
wb.record("12.21d", outside_21, mistakes={"compare chaque prédiction au minimum ET au maximum des voitures d'entraînement": 0})
wb.record("12.21e", target_scaler_matters_21, mistakes={"compare les deux pipelines sur les mêmes températures : leurs prédictions en voitures diffèrent-elles ?": True})''',
       note="La prédiction pour −2 °C vaut environ 1 311 voitures ; dans les unités transformées de la cible, environ "
            "0,32. Ramenée avec le transformateur de la température, elle donnerait une température (environ −0,3 °C) "
            "prise pour un nombre de voitures, un non-sens que rien ne signale. `TransformedTargetRegressor` fait "
            "exactement les mêmes calculs que la fonction écrite à la main. Le transformateur de la cible ne change "
            "rien ici : une Ridge est linéaire en $y$ et n'en pénalise pas l'ordonnée à l'origine, si bien que "
            "transformer $y$ par une fonction affine, puis revenir, redonne les mêmes prédictions. Celui de la "
            "feature compte, car la pénalité porte sur le poids, qui dépend de l'unité de la température (🔁 "
            "12.R2). 25 °C est hors de la plage d'entraînement : la droite extrapole au-delà de toutes les voitures "
            "vues ; une prédiction à signaler plutôt qu'à croire."),
])

# ---------------------------------------------------------------------------
# Part C: PCA, feature selection and encoders (12.22 to 12.28)
# ---------------------------------------------------------------------------
PART_C_GIVEN = r'''# MNIST for part C: the training images flattened (784 pixels from 0 to 255) and their labels
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder

MNIST_X, MNIST_Y = wb.datasets.load_mnist("train", flatten=True)
print(MNIST_X.shape, MNIST_X.dtype)'''

GIVEN_22 = r'''rng_22 = np.random.default_rng(1222)
angle_22_rad = np.radians(30)
rotation_22 = np.array([[np.cos(angle_22_rad), -np.sin(angle_22_rad)], [np.sin(angle_22_rad), np.cos(angle_22_rad)]])
CLOUD_22 = (rng_22.normal(size=(300, 2)) * [3.0, 1.0]) @ rotation_22.T + [12.0, 40.0]   # 300 points, raw units
print("means", CLOUD_22.mean(axis=0).round(2), "· standard deviations", CLOUD_22.std(axis=0).round(2))'''

TODO_22 = r'''def project_22(X, u):
    """The coordinates of the rows of X (n, 2) on the line through the origin of direction u (not necessarily unit):
    an array of shape (n,)."""
    raise NotImplementedError("project_22")


cloud_s_22 = ...       # the standardized cloud (each column: mean 0, standard deviation 1)
u1_22 = ...            # the unit direction of maximal variance of cloud_s_22, first coordinate positive
share_x_22 = ...       # a) share of the total variance of cloud_s_22 kept by the projection on the horizontal axis
share_pc1_22 = ...     # b) the same share for the projection on u1_22
angle_22 = ...         # c) the angle of u1_22 with the horizontal axis, in degrees, between 0 and 180'''

SOLUTION_22 = r'''def project_22(X, u):
    """The coordinates of the rows of X (n, 2) on the line through the origin of direction u (not necessarily unit):
    an array of shape (n,)."""
    u = np.asarray(u, dtype=float)
    return np.asarray(X, dtype=float) @ (u / np.linalg.norm(u))


cloud_s_22 = (CLOUD_22 - CLOUD_22.mean(axis=0)) / CLOUD_22.std(axis=0)
values_22, vectors_22 = np.linalg.eigh(np.cov(cloud_s_22, rowvar=False))   # eigenvalues in increasing order
u1_22 = vectors_22[:, -1] * np.sign(vectors_22[0, -1])
total_22 = np.var(cloud_s_22, axis=0).sum()
share_x_22 = float(np.var(project_22(cloud_s_22, [1.0, 0.0])) / total_22)
share_pc1_22 = float(np.var(project_22(cloud_s_22, u1_22)) / total_22)
angle_22 = float(np.degrees(np.arctan2(u1_22[1], u1_22[0])) % 180)
print(u1_22.round(4), round(share_x_22, 4), round(share_pc1_22, 4), round(angle_22, 2))

fig_22, axes_22 = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (title, u) in zip(axes_22[:2], [("(a) on the horizontal axis", np.array([1.0, 0.0])), ("(b) on the principal axis", u1_22)]):
    t = project_22(cloud_s_22, u)
    feet = np.outer(t, u / np.linalg.norm(u))                  # the projected points, on the line
    ax.scatter(*cloud_s_22.T, s=8, color="C0", alpha=0.5, label="standardized cloud")
    ax.plot(*np.array([-4 * u, 4 * u]).T, color="C3", lw=1.5, label="line")
    for point, foot in zip(cloud_s_22[::10], feet[::10]):
        ax.plot(*np.array([point, foot]).T, color="gray", lw=0.6)
    ax.scatter(*feet.T, s=8, color="C3", label="projections")
    ax.set(title=title, xlim=(-4, 4), ylim=(-4, 4), aspect="equal")
    ax.legend(fontsize=8, loc="upper left")
axes_22[2].hist([project_22(cloud_s_22, [1.0, 0.0]), project_22(cloud_s_22, u1_22)], bins=30,
                label=[f"horizontal axis ({share_x_22:.0%})", f"principal axis ({share_pc1_22:.0%})"], color=["C0", "C3"])
axes_22[2].set(title="(c) the two clouds in 1D", xlabel="coordinate")
axes_22[2].legend(fontsize=8)
plt.tight_layout()
plt.show()'''

CHECK_22 = r'''with wb.attempt("12.22"):
    probe_22 = project_22(np.array([[2.0, 3.0], [1.0, 0.0]]), [3.0, 4.0])
    if returned("12.22", "project_22", probe_22):
        probe_22 = np.asarray(probe_22, dtype=float)
        if probe_22.shape != (2,):
            print(f"❌ Ex 12.22 : project_22 doit renvoyer un tableau de forme (n,), une coordonnée par point ; reçu {probe_22.shape}.")
        else:
            verdict("12.22", np.allclose(probe_22, [3.6, 0.6]), "project_22 divise bien la direction par sa norme.",
                    f"project_22([[2, 3], [1, 0]], [3, 4]) renvoie {np.round(probe_22, 4)} au lieu de [3.6 0.6]"
                    + (" : u doit être unitaire." if np.allclose(probe_22, [18.0, 3.0]) else "."))
wb.check("12.22a", share_x_22)
wb.check("12.22b", share_pc1_22)
wb.check("12.22c", angle_22)'''

GIVEN_23 = r'''ones_23 = MNIST_X[MNIST_Y == 1][:2000] / 255.0        # 2,000 images of 1
eights_23 = MNIST_X[MNIST_Y == 8][:2000] / 255.0      # 2,000 images of 8
ratios_23 = {digit: SkPCA().fit(images).explained_variance_ratio_ for digit, images in [("1", ones_23), ("8", eights_23)]}
fig_23, axes_23 = plt.subplots(1, 2, figsize=(14, 4.8))
for (digit, ratio), color in zip(ratios_23.items(), ["C0", "C1"]):
    k = np.arange(1, 41)
    axes_23[0].plot(k[:20], ratio[:20], marker="o", ms=4, color=color, label=f"the {digit}s")
    axes_23[1].plot(k, np.cumsum(ratio)[:40], marker="o", ms=3, color=color, label=f"the {digit}s")
axes_23[0].set(xlabel="component k", ylabel="explained variance ratio", title="(a) each component", xticks=range(1, 21))
axes_23[1].set(xlabel="number of components k", ylabel="cumulative explained variance ratio", title="(b) cumulative",
               xticks=range(0, 41, 2), yticks=np.arange(0.1, 1.01, 0.1), ylim=(0.1, 1.0))
axes_23[1].set_yticks(np.arange(0.1, 1.0, 0.05), minor=True)
for ax in axes_23:
    ax.grid(alpha=0.4)
    ax.grid(which="minor", alpha=0.15)
    ax.legend()
plt.tight_layout()
plt.show()'''

TODO_23 = r'''first_ones_23 = ...    # a) "A", "B", "C" or "D"
k60_ones_23 = ...      # b) an integer read on the figure
at20_eights_23 = ...   # c) "A", "B", "C" or "D"
ones_90_23 = ...       # d) True or False'''

CHECK_23 = r'''wb.check("12.23a", first_ones_23)
wb.check("12.23b", k60_ones_23)
wb.check("12.23c", at20_eights_23)
wb.check("12.23d", ones_90_23)'''

EXPERIMENT_24 = r'''N_24 = wb.by_mode(fast=5000, full=20000)
images_24 = MNIST_X[np.random.default_rng(1224).choice(len(MNIST_X), N_24, replace=False)] / 255.0
start_24 = time.perf_counter()
cumulative_24 = np.cumsum(SkPCA().fit(images_24).explained_variance_ratio_)
standardized_24 = SkStandardScaler().fit_transform(images_24)
cumulative_std_24 = np.cumsum(SkPCA().fit(standardized_24).explained_variance_ratio_)
print(f"{N_24} images, two PCA in {time.perf_counter() - start_24:.1f} s")
for share in (0.5, 0.9, 0.95, 0.99):
    print(f"to keep more than {share:.0%} of the variance: {int(np.argmax(cumulative_24 > share)) + 1} components "
          f"(standardized pixels: {int(np.argmax(cumulative_std_24 > share)) + 1})")
plt.plot(np.arange(1, 785), cumulative_24, label="pixels in [0, 1], centred")
plt.plot(np.arange(1, 785), cumulative_std_24, label="standardized pixels")
plt.axhline(0.9, color="gray", lw=0.8, ls="--")
plt.xscale("log")
plt.xlabel("number of components k (log scale)")
plt.ylabel("cumulative explained variance ratio")
plt.legend()
plt.show()
wb.check("12.24a", k90_24)
wb.check("12.24b", k99_24)
wb.check("12.24c", standardized_more_24)'''

GIVEN_25 = r'''from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score

NUMERIC_25 = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)",
              "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"]
rows_25 = RAW.dropna(subset=NUMERIC_25)                          # 330 penguins with the 6 measures
SEL_25 = pd.concat([rows_25[NUMERIC_25], (rows_25["Clutch Completion"] == "Yes").astype(float).rename("Clutch"),
                    pd.get_dummies(rows_25[["Island", "Region", "Stage"]], dtype=float)], axis=1)
Y_25 = rows_25["Species"].str.split().str[0]
CV_25 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1225)
print(SEL_25.shape)
SEL_25.head(3)'''

TODO_25 = r'''constant_25 = ...      # a) names of the columns of SEL_25 removed by VarianceThreshold()
n_removed_25 = ...     # b) number of columns of SEL_25 removed by VarianceThreshold(threshold=1.0)
kg_kept_25 = ...       # c) True or False: with the mass in kg, would VarianceThreshold(threshold=1.0) keep it?
best2_25 = ...         # d) names of the 2 columns of NUMERIC_25 kept by SelectKBest(f_classif, k=2) for Y_25
cv_all_25 = ...        # e) mean accuracy (CV_25) of StandardScaler + LogisticRegression(max_iter=1000) on NUMERIC_25
cv_best2_25 = ...      # e) the same with SelectKBest(f_classif, k=2) between the two, inside the pipeline'''

SOLUTION_25 = r'''from sklearn.pipeline import make_pipeline

constant_25 = list(SEL_25.columns[~VarianceThreshold().fit(SEL_25).get_support()])
n_removed_25 = int((~VarianceThreshold(threshold=1.0).fit(SEL_25).get_support()).sum())
kg_kept_25 = bool(np.var(SEL_25["Body Mass (g)"] / 1000) > 1.0)
best2_25 = list(np.array(NUMERIC_25)[SelectKBest(f_classif, k=2).fit(SEL_25[NUMERIC_25], Y_25).get_support()])
cv_all_25 = float(cross_val_score(make_pipeline(SkStandardScaler(), LogisticRegression(max_iter=1000)),
                                  SEL_25[NUMERIC_25], Y_25, cv=CV_25).mean())
cv_best2_25 = float(cross_val_score(make_pipeline(SkStandardScaler(), SelectKBest(f_classif, k=2),
                                                  LogisticRegression(max_iter=1000)),
                                    SEL_25[NUMERIC_25], Y_25, cv=CV_25).mean())
print(constant_25, n_removed_25, kg_kept_25, best2_25, round(cv_all_25, 4), round(cv_best2_25, 4))
print(SEL_25.var(ddof=0).round(3).to_dict())'''

CHECK_25 = r'''wb.check("12.25a", constant_25)
wb.check("12.25b", n_removed_25)
wb.check("12.25c", kg_kept_25)
wb.check("12.25d", best2_25)
wb.check("12.25e", [cv_all_25, cv_best2_25] if filled(cv_all_25, cv_best2_25) else ...)'''

CHECK_26 = r'''with checked("12.26"):
    X_doc_26 = [["red", "S"], ["blue", "M"], ["red", "L"]]
    ordinal_26 = fitted(mylearn.preprocessing.OrdinalEncoder(), X_doc_26)
    print("OrdinalEncoder, docstring example: categories_", [c.tolist() for c in ordinal_26.categories_],
          "· codes", ordinal_26.transform(X_doc_26).tolist())
    sizes_26 = fitted(mylearn.preprocessing.OrdinalEncoder(categories=[["S", "M", "L"]]), [["S"], ["L"], ["M"]])
    print("with the order S < M < L:", sizes_26.transform([["S"], ["L"], ["M"]]).ravel().tolist(), "(expected [0, 2, 1])")
with checked("12.26"):
    cats_26 = PENGUINS[["island", "sex"]].to_numpy(dtype=object)
    for options in [{}, {"drop": "first"}, {"drop": "if_binary"}]:
        mine_26 = fitted(mylearn.preprocessing.OneHotEncoder(**options), cats_26)
        oracle_26 = SkOneHotEncoder(sparse_output=False, **options).fit(cats_26)
        same_26 = (np.array_equal(mine_26.transform(cats_26), oracle_26.transform(cats_26))
                   and list(mine_26.get_feature_names_out(["island", "sex"])) == list(oracle_26.get_feature_names_out(["island", "sex"])))
        verdict("12.26", same_26, f"OneHotEncoder({options}) : mêmes colonnes et mêmes noms que scikit-learn "
                f"{list(oracle_26.get_feature_names_out(['island', 'sex']))}.",
                f"OneHotEncoder({options}) : colonnes ou noms différents de ceux de scikit-learn "
                f"{list(oracle_26.get_feature_names_out(['island', 'sex']))}.")
if (written(lambda: mylearn.preprocessing.OrdinalEncoder().fit([["a"], ["b"]]))
        or written(lambda: mylearn.preprocessing.OneHotEncoder().fit([["a"], ["b"]]))):
    run_mylearn_tests("test_ordinal_encoder_ or test_one_hot_encoder_")
else:
    print("⏳ Ex 12.26 : les tests attendent le fit de ton OrdinalEncoder ou de ton OneHotEncoder.")'''

CHECK_27 = r'''with checked("12.27"):
    X_doc_27 = np.array([[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]])
    doc_27 = fitted(mylearn.preprocessing.PCA(n_components=2), X_doc_27)
    print("docstring example: explained_variance_ratio_", np.round(doc_27.explained_variance_ratio_, 4),
          "(expected [0.9924 0.0076])")
with checked("12.27"):
    rng_27 = np.random.default_rng(1227)
    X_27 = rng_27.normal(size=(200, 5)) @ rng_27.normal(size=(5, 5)) + rng_27.normal(0, 3, size=5)
    mine_27 = fitted(mylearn.preprocessing.PCA(), X_27)
    oracle_27 = SkPCA(svd_solver="full").fit(X_27)
    print("explained_variance_ratio_:", np.round(mine_27.explained_variance_ratio_, 4).tolist())
    same_27 = (np.allclose(mine_27.explained_variance_ratio_, oracle_27.explained_variance_ratio_)
               and np.allclose(np.abs(mine_27.components_), np.abs(oracle_27.components_))
               and np.allclose(mine_27.inverse_transform(mine_27.transform(X_27)), X_27))
    verdict("12.27", same_27, "mêmes parts de variance et mêmes composantes (au signe près) que scikit-learn ; "
            "avec toutes les composantes, la reconstruction est exacte.",
            "parts de variance, composantes ou reconstruction différentes de celles de scikit-learn : les tests "
            "ci-dessous disent où.")
    k90_27 = fitted(mylearn.preprocessing.PCA(n_components=0.9), X_27).n_components_
    print("n_components=0.9 keeps", k90_27, "components (scikit-learn:", SkPCA(n_components=0.9, svd_solver="full").fit(X_27).n_components_, ")")
if written(lambda: mylearn.preprocessing.PCA().fit(np.eye(3))):
    run_mylearn_tests("test_pca_")
else:
    print("⏳ Ex 12.27 : les tests attendent le fit de ta PCA.")'''

GIVEN_28 = r'''TRAIN_28 = wb.datasets.load_mnist("train", n=5000, flatten=True, seed=1228)[0] / 255.0   # the PCA learns here
TEST_28 = wb.datasets.load_mnist("test", n=1000, flatten=True, seed=1228)[0] / 255.0     # never seen by the PCA
GRID_28 = [5, 10, 20, 50, 100, 200, 400]
print(TRAIN_28.shape, TEST_28.shape)'''

TODO_28 = r'''def errors_28(k):
    """Fit SkPCA(n_components=k, svd_solver="full") on TRAIN_28; return (train_error, test_error): the mean, over the images and the
    pixels, of the squared difference between an image and its reconstruction with k components."""
    raise NotImplementedError("errors_28")


def baseline_28():
    """The same test error with k = 0 components: every test image "reconstructed" by the mean image of TRAIN_28."""
    raise NotImplementedError("baseline_28")'''

SOLUTION_28 = r'''def errors_28(k):
    """Fit SkPCA(n_components=k, svd_solver="full") on TRAIN_28; return (train_error, test_error): the mean, over the images and the
    pixels, of the squared difference between an image and its reconstruction with k components."""
    pca = SkPCA(n_components=k, svd_solver="full").fit(TRAIN_28)
    return tuple(float(np.mean((pca.inverse_transform(pca.transform(images)) - images) ** 2))
                 for images in (TRAIN_28, TEST_28))


def baseline_28():
    """The same test error with k = 0 components: every test image "reconstructed" by the mean image of TRAIN_28."""
    return float(np.mean((TEST_28 - TRAIN_28.mean(axis=0)) ** 2))

'''

CHECK_28 = r'''with wb.attempt("12.28"):
    table_28 = {k: errors_28(k) for k in GRID_28}
    if all(returned("12.28", "errors_28", value) for value in table_28.values()):
        if not all(isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2 for value in table_28.values()):
            print("❌ Ex 12.28 : errors_28 doit renvoyer un couple (erreur d'entraînement, erreur de test).")
        else:
            for k, (train_error, test_error) in table_28.items():
                print(f"k = {k:3d}: train {train_error:.5f} · test {test_error:.5f} · test / train {test_error / train_error:.3f}")
            first_28 = next((k for k, (tr, te) in table_28.items() if te > tr), None)
            wb.check("12.28a", first_28)
            wb.check("12.28b", table_28[400][1] / table_28[400][0], computed=True)
            base_28 = baseline_28()
            if returned("12.28", "baseline_28", base_28):
                print(f"k = 0 (the mean image): test error {base_28:.5f}")
                wb.check("12.28c", 1 - table_28[50][1] / base_28, computed=True)
with wb.attempt("12.28"):
    pca_28 = SkPCA(n_components=200, svd_solver="full").fit(TRAIN_28)
    fig_28, axes_28 = plt.subplots(5, 8, figsize=(12, 7.8))
    axes_28[0, 0].imshow(pca_28.mean_.reshape(28, 28), cmap="gray")
    axes_28[0, 0].set_title("mean", fontsize=9)
    for j in range(7):
        limit = np.abs(pca_28.components_[j]).max()
        axes_28[0, j + 1].imshow(pca_28.components_[j].reshape(28, 28), cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes_28[0, j + 1].set_title(f"component {j + 1}", fontsize=9)
    for row, k in zip(range(1, 4), (10, 50, 200)):
        Z = pca_28.transform(TEST_28[:8])
        Z[:, k:] = 0                                            # keep the first k coordinates only
        for j, image in enumerate(pca_28.inverse_transform(Z)):
            axes_28[row, j].imshow(image.reshape(28, 28), cmap="gray")
            axes_28[row, j].set_title(f"k = {k}", fontsize=9)
    for j in range(8):
        axes_28[4, j].imshow(TEST_28[j].reshape(28, 28), cmap="gray")
        axes_28[4, j].set_title("original", fontsize=9)
    for ax in axes_28.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()'''

PART_C = Part("C", "Réduire la dimension, sélectionner et encoder",
              "La PCA d'abord à la main sur un nuage en 2D (12.22), puis sur des chiffres manuscrits (12.23, 12.24, "
              "12.28) ; entre les deux, la sélection de features (12.25) et les deux encodeurs (12.26) et la PCA "
              "(12.27) de ta librairie. La cellule suivante charge les 60 000 images d'entraînement de MNIST, "
              "aplaties en 784 pixels.",
              given=PART_C_GIVEN, exercises=[
    Ex("12.22", "🎨", 2, 25, "Projeter un nuage 2D : axe horizontal contre axe de variance maximale",
       "voir et mesurer ce que garde une projection, selon la droite choisie.",
       "Ex 12.7 · fiche §12.7.1 (encadrés 🧮 : projection, vecteurs propres)", thread="synthétique", tracks="R, M, C",
       body=r"""`CLOUD_22` est un nuage **inventé** de 300 points en deux features, aux unités différentes. Comme le livre avec son nuage en guitare (p. 452-458), standardise-le d'abord, puis compare deux projections. Écris :
- `project_22(X, u)` : les coordonnées des lignes de `X` sur la droite passant par l'origine de direction `u` (un vecteur **pas forcément unitaire**) ;
- `cloud_s_22` : le nuage standardisé (chaque colonne de moyenne 0 et d'écart-type 1) ;
- `u1_22` : la direction unitaire de variance maximale de `cloud_s_22` (un vecteur propre de sa matrice de covariance, `np.linalg.eigh`), de première coordonnée positive ;
- a) `share_x_22` : la part de la variance totale de `cloud_s_22` que garde la projection sur l'axe horizontal (la variance des coordonnées divisée par la somme des variances des deux features) ;
- b) `share_pc1_22` : la même part pour la projection sur `u1_22` ;
- c) `angle_22` : l'angle de `u1_22` avec l'axe horizontal, en degrés, entre 0 et 180.

Puis trace la figure en trois panneaux : (a) le nuage, l'axe horizontal et les points projetés ; (b) la même chose avec l'axe principal ; (c) les deux nuages obtenus à une dimension (deux histogrammes).

Dans tes notes : a) et c) sont des nombres ronds ; pourquoi ? Pour quels nuages standardisés l'angle de c) serait-il différent ? Que vaudraient a), b) et c) sans standardiser ?""",
       given=GIVEN_22, todo=TODO_22, check=CHECK_22, solution=SOLUTION_22 + "\n\n" + CHECK_22.replace('with wb.attempt("12.22"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''raw_values_22, raw_vectors_22 = np.linalg.eigh(np.cov(CLOUD_22, rowvar=False))
raw_angle_22 = float(np.degrees(np.arctan2(raw_vectors_22[1, -1], raw_vectors_22[0, -1])) % 180)
wb.record("12.22a", share_x_22, decimals=4,
          mistakes={"c'est la part du nuage brut : standardise d'abord": float(np.var(CLOUD_22[:, 0]) / np.var(CLOUD_22, axis=0).sum())})
wb.record("12.22b", share_pc1_22, decimals=3,
          mistakes={"c'est la part du nuage brut : standardise d'abord": float(raw_values_22[-1] / raw_values_22.sum()),
                    "c'est la part de la direction de plus petite variance : prends le vecteur propre de la plus grande valeur propre": 1 - share_pc1_22})
wb.record("12.22c", angle_22, decimals=2,
          mistakes={"c'est l'angle de l'axe principal du nuage brut : standardise d'abord": raw_angle_22,
                    "c'est l'angle de la direction de plus petite variance : np.linalg.eigh range les valeurs propres par ordre croissant": 135.0})''',
       note="Après une standardisation, les deux features ont la même variance : l'axe horizontal garde exactement "
            "la moitié de la variance totale (a : 0,5). La matrice de covariance s'écrit alors "
            "$\\begin{pmatrix} 1 & r \\\\ r & 1 \\end{pmatrix}$ ($r$ la corrélation), dont les vecteurs propres sont "
            "toujours $(1, 1)/\\sqrt{2}$ et $(1, -1)/\\sqrt{2}$, de valeurs propres $1 + r$ et $1 - r$ : l'axe "
            "principal est à 45° pour toute corrélation positive (c), à 135° pour une corrélation négative, et il "
            "garde la part $(1 + r)/2$ (b : environ 0,90, avec $r \\approx 0{,}80$). En dimension 2, une PCA de "
            "features standardisées ne choisit donc que le signe de la corrélation ; c'est la valeur de $r$ qui dit "
            "combien elle garde. Sans standardiser, la direction dépend des unités (environ 28,5° ici), et l'axe "
            "horizontal garde la part de la feature la plus étendue (environ 73 %)."),

    Ex("12.23", "📈", 2, 15, "Lire une courbe de variance expliquée cumulée",
       "lire sur des courbes de variance expliquée la place qu'occupent les premières composantes, et combien il en "
       "faut.",
       "Ex 12.22 · fiche §12.7.1 (combien de composantes ?)", thread="MNIST", tracks="R, M, C",
       body=r"""La cellule suivante calcule la PCA de 2 000 images de **1** et celle de 2 000 images de **8**, puis trace (a) la part de variance expliquée par chacune des 20 premières composantes, (b) la part cumulée des 40 premières. Lis **la figure** (sans calculer) :
a) `first_ones_23` : la part de variance de la première composante des 1 : `"A"` environ 13 % ; `"B"` environ 35 % ; `"C"` environ 60 % ; `"D"` environ 90 % ;
b) `k60_ones_23` : le plus petit nombre de composantes qui garde plus de 60 % de la variance des 1 ;
c) `at20_eights_23` : la part de variance des 8 gardée par 20 composantes : `"A"` environ 50 % ; `"B"` environ 65 % ; `"C"` environ 80 % ; `"D"` environ 95 % ;
d) `ones_90_23` : les 1 dépassent-ils 90 % avec au plus 40 composantes ? (`True` ou `False`)

Dans tes notes : pourquoi les 1 se résument-ils avec moins de composantes que les 8 ? À quoi pourrait ressembler la première composante des 1 (pense à ce qui varie le plus d'un 1 à l'autre) ?""",
       given=GIVEN_23, todo=TODO_23, check=CHECK_23,
       solution=r'''first_ones_23, k60_ones_23, at20_eights_23, ones_90_23 = "B", 4, "B", True
for digit, ratio in ratios_23.items():
    cumulative = np.cumsum(ratio)
    print(f"the {digit}s: first component {ratio[0]:.3f} · cumulative at k = 3, 4, 20, 40: "
          f"{cumulative[2]:.3f}, {cumulative[3]:.3f}, {cumulative[19]:.3f}, {cumulative[39]:.3f}")''',
       record=r'''wb.record("12.23a", "B", mistakes={"c'est la part de la première composante des 8 : lis la courbe des 1": "A",
                                       "lis la valeur de la première composante sur le panneau (a), pas un cumul": "C"})
wb.record("12.23b", 4, mistakes={"à cette valeur, la courbe des 1 dépasse-t-elle vraiment 60 % sur le panneau (b) ?": 3})
wb.record("12.23c", "B", mistakes={"c'est la part cumulée des 1 à k = 20 : lis la courbe des 8": "C",
                                     "à quelle abscisse lis-tu la courbe cumulée des 8 ? la question porte sur k = 20": "A"})
wb.record("12.23d", True, mistakes={"lis la fin de la courbe des 1 sur le panneau (b), à k = 40": False})''',
       note="La première composante des 1 explique à elle seule 35 % de leur variance ; il en faut 4 pour dépasser "
            "60 % (57 % avec 3, 62 % avec 4). Les 8 sont plus divers : 20 composantes n'en gardent que 66 %, et "
            "40 composantes 80 %, quand les 1 dépassent 90 % dès 35. Un 1 varie surtout par son inclinaison et son "
            "épaisseur : quelques directions suffisent. La première composante des 1 oppose d'ailleurs un 1 droit à un 1 "
            "penché vers la droite. Un 8 varie par la taille relative de ses "
            "boucles, leur position, leur inclinaison, leurs ouvertures : plus de directions."),

    Ex("12.24", "🔮", 2, 20, "Combien de composantes pour 90 % de la variance de MNIST ?",
       "prévoir la dimension utile d'un dataset d'images, avant de la mesurer.",
       "Ex 12.23 · fiche §12.7.1 et §12.7.2", thread="MNIST", tracks="M, C",
       hypothesis=True,
       body=r"""On fait la PCA de 5 000 images de MNIST tirées au hasard parmi les 60 000 (20 000 hors FAST_MODE), tous chiffres confondus, pixels ramenés dans $[0, 1]$ et centrés. Une image a 784 pixels.

Prédis, **avant** d'exécuter quoi que ce soit :
a) `k90_24` : le nombre de composantes qu'il faut pour garder plus de 90 % de la variance : `"A"` environ 10 ; `"B"` environ 30 ; `"C"` environ 90 ; `"D"` environ 300 ;
b) `k99_24` : le nombre de composantes pour plus de 99 % : `"A"` environ 100 ; `"B"` environ 330 ; `"C"` environ 600 ; `"D"` les 784 ;
c) `standardized_more_24` : si l'on **standardise** chaque pixel avant la PCA, comme le livre, faut-il plus de composantes pour garder 90 % ? (`True` ou `False`)

Puis exécute l'expérience.""",
       todo=r'''k90_24 = ...                 # a) "A", "B", "C" or "D"
k99_24 = ...                 # b) "A", "B", "C" or "D"
standardized_more_24 = ...   # c) True or False''',
       solution=r'''k90_24, k99_24, standardized_more_24 = "C", "B", True   # the answers, for the record''',
       after=[("code", guarded(EXPERIMENT_24, ["k90_24", "k99_24", "standardized_more_24"],
                               "⏳ Ex 12.24 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : compare avec tes prédictions. Que représentent les dernières composantes, celles "
                     "qu'il faut ajouter pour passer de 90 % à 99 % ? Comment expliques-tu l'effet de la "
                     "standardisation des pixels ? Un modèle entraîné sur les coordonnées de la PCA a-t-il besoin "
                     "des 99 % ?")],
       record=r'''wb.record("12.24a", "C", mistakes={"10 composantes gardent environ la moitié de la variance des 8 seuls (12.23) : tous les chiffres ensemble en demandent-ils moins ?": "A",
                                       "40 composantes gardent 80 % de la variance des 8 seuls (12.23) : tous les chiffres ensemble en demandent-ils moins ?": "B",
                                       "un dixième de l'image suffit-il, ou en faut-il plus du tiers ? la courbe cumulée monte vite, puis lentement": "D"})
wb.record("12.24b", "B", mistakes={"de 90 % à 99 %, la courbe cumulée monte de plus en plus lentement : faut-il seulement quelques composantes de plus ?": "A",
                                     "beaucoup de pixels du bord sont presque toujours noirs : combien de directions portent vraiment de la variance ?": "C",
                                     "les pixels du bord, presque toujours noirs, n'ont presque pas de variance : faut-il vraiment les 784 directions ?": "D"})
wb.record("12.24c", True, mistakes={"standardiser donne la même variance 1 à chaque pixel : que devient la variance d'un pixel du bord, presque toujours noir ?": False})''',
       note="Il faut 86 composantes pour garder 90 % de la variance (87 avec 20 000 images, comme avec les 60 000), et "
            "321 pour 99 % (330 avec 20 000 images) : la courbe cumulée monte vite, puis très lentement. Les composantes de 90 % à 99 % "
            "décrivent des détails fins et du bruit d'écriture ; un classifieur sur les coordonnées de la PCA s'en "
            "passe très bien. Avec des pixels standardisés, il en faut 181 pour 90 % (212 avec 20 000 images, 236 avec "
            "les 60 000) : un pixel du bord, "
            "presque toujours noir, voit sa minuscule variance amplifiée jusqu'à 1, et ces pixels de bruit "
            "réclament leurs propres directions. C'est l'encadré ⚠️ du §12.7.2 : pour des pixels de même unité, on "
            "centre sans réduire. Le réglage de la PCA (centrer ou standardiser) change la dimension qu'elle trouve."),

    Ex("12.25", "📦", 2, 25, "Sélection de features : colonnes constantes, VarianceThreshold, SelectKBest",
       "retirer des colonnes sans intérêt, puis choisir des features selon leur lien avec la cible, sans fuite.",
       "Ex 12.11 · ch. 2 (variance) · fiche §12.6", thread="Penguins", tracks="C",
       body=r"""`SEL_25` contient, pour les 330 manchots de `RAW` qui ont les six mesures, ces six mesures (`NUMERIC_25`), `Clutch` (1 si la couvée était complète) et le one-hot de `Island`, `Region` et `Stage` : 12 colonnes. La cible `Y_25` est l'espèce. Calcule :
a) `constant_25` : les noms des colonnes que retire `VarianceThreshold()` (seuil 0) ;
b) `n_removed_25` : le nombre de colonnes que retire `VarianceThreshold(threshold=1.0)` ;
c) `kg_kept_25` : si la masse était en kilogrammes, `VarianceThreshold(threshold=1.0)` la garderait-il ? (`True` ou `False`) ;
d) `best2_25` : les noms des 2 colonnes de `NUMERIC_25` que garde `SelectKBest(f_classif, k=2)` pour prédire l'espèce ;
e) `cv_all_25` et `cv_best2_25` : l'accuracy moyenne, en validation croisée `CV_25`, d'un pipeline `StandardScaler` puis `LogisticRegression(max_iter=1000)` sur les six mesures, puis du même pipeline avec `SelectKBest(f_classif, k=2)` entre les deux.

Dans tes notes : que t'apprennent b) et c) sur le seuil de `VarianceThreshold` ? Pourquoi `SelectKBest` doit-il être **dans** le pipeline en e) ? Que perd-on à ne garder que deux mesures ?""",
       given=GIVEN_25, todo=TODO_25, check=CHECK_25, solution=SOLUTION_25,
       record=r'''wb.record("12.25a", constant_25, ordered=False,
          mistakes={"une seule colonne à variance nulle ? une colonne one-hot est constante quand la colonne d'origine ne prend qu'une valeur : relis 12.11 a)": ["Region_Anvers"]})
wb.record("12.25b", n_removed_25, mistakes={"compte les colonnes retirées, pas celles qui restent": SEL_25.shape[1] - n_removed_25,
                                            "le seuil est 1.0 : seules les colonnes constantes ont une variance nulle, mais lesquelles ont une variance sous 1 ?": 2})
wb.record("12.25c", kg_kept_25, mistakes={"en kilogrammes, la variance est divisée par 1 000² : compare-la au seuil": True})
wb.record("12.25d", best2_25, ordered=False,
          mistakes={"SelectKBest garde les k plus GRANDS scores de f_classif": list(np.array(NUMERIC_25)[np.argsort(SelectKBest(f_classif, k=2).fit(SEL_25[NUMERIC_25], Y_25).scores_)[:2]])})
wb.record("12.25e", [cv_all_25, cv_best2_25], decimals=4)''',
       note="`Region` et `Stage` sont constantes : leurs colonnes one-hot ont une variance nulle. Avec un seuil de 1, "
            "`VarianceThreshold` retire 8 colonnes : les deux isotopes (variances 0,3 et 0,6), `Clutch` et toutes les "
            "colonnes 0/1 (une variance d'au plus 0,25). Le seuil compare des variances exprimées dans des unités "
            "différentes, ce qui n'a pas de sens : la masse en kilogrammes (variance environ 0,65) serait retirée, "
            "en grammes elle reste. On n'applique donc un seuil de variance qu'à des colonnes de même unité, ou pour "
            "retirer les colonnes constantes. `f_classif` classe les mesures par le rapport entre la variance des "
            "moyennes par espèce et la variance dans chaque espèce : la nageoire et la longueur du bec gagnent. "
            "Dans le pipeline, la sélection est refaite à chaque fold, sur l'entraînement seul : faite avant, elle "
            "aurait vu les labels de la validation (8.25). Elle n'est d'ailleurs pas stable : dans un des cinq folds, "
            "la profondeur du bec passe devant la longueur (scores 276 contre 274), et ce fold tombe à 0,80 (les "
            "quatre autres vont de 0,92 à 1) : 0,92 en moyenne, contre 1 avec les six mesures."),

    Ex("12.26", "🔨", 3, 45, "Coder OrdinalEncoder et OneHotEncoder",
       "écrire les deux encodeurs de catégories, avec un ordre imposé, des colonnes retirées et des catégories "
       "inconnues.",
       "Ex 12.1 · Ex 12.16 · fiche §12.3 et §12.3.1", thread="Penguins", tracks="C", mylearn="preprocessing.py",
       body=MYLEARN_SHORT + r"""

Écris les classes `OrdinalEncoder` et `OneHotEncoder`. **Contrat** commun (les docstrings, plus ces précisions que vérifient les tests) : `__init__` range ses arguments tels quels ; `fit(X)` renvoie `self` et n'ajoute que des attributs `nom_` ou `_nom` ; `X` est une liste de listes ou un tableau (chaînes ou nombres) ; un `X` qui n'est pas 2-D lève une `ValueError` ; `RuntimeError` avant `fit`.

`OrdinalEncoder(categories="auto")` :
- `categories_` est une **liste** de tableaux NumPy, un par colonne : avec `"auto"`, les valeurs distinctes vues par `fit`, dans l'ordre de `sorted` et de `np.unique` (majuscules avant minuscules, nombres par valeur) ; sinon, les listes données, dans leur ordre (`ValueError` s'il n'y a pas une liste par colonne, ou si une valeur de `X` manque à la liste de sa colonne) ;
- `transform` rend des codes `float64` (la position dans `categories_[j]`), `ValueError` pour une catégorie inconnue ou un mauvais nombre de colonnes ; `inverse_transform` rend un tableau d'objets, `ValueError` pour un code qui n'est pas une position valide.

`OneHotEncoder(categories="auto", drop=None, handle_unknown="error")` :
- `fit` lève une `ValueError` si `drop` n'est pas `None`, `"first"` ou `"if_binary"`, si `handle_unknown` n'est pas `"error"` ou `"ignore"`, ou si une valeur de `X` manque à la liste donnée de sa colonne ; `categories_` comme ci-dessus ;
- `drop_idx_` vaut `None` si `drop` est `None` ; sinon c'est un tableau d'**objets** de forme `(n_features,)` : 0 pour une colonne dont on retire la première catégorie, `None` pour une colonne où rien n'est retiré (`"if_binary"` sur une colonne qui n'a pas deux catégories) ;
- `transform` rend un tableau `float64` de 0 et de 1, bloc par bloc, sans les colonnes retirées ; une catégorie inconnue lève une `ValueError`, ou donne un bloc de zéros avec `handle_unknown="ignore"` ;
- `inverse_transform` lit chaque bloc : la position du 1 donne la catégorie ; un bloc **entièrement nul** donne la catégorie retirée de **sa** colonne s'il y en a une (lue dans `drop_idx_`), sinon `None` ;
- `get_feature_names_out(input_features=None)` rend un tableau NumPy de noms `"<colonne>_<catégorie>"`, la catégorie écrite avec `str()` (`"x0_1.0"` pour le nombre 1.0), sans les catégories retirées ; `ValueError` si `input_features` n'a pas le bon nombre de noms.

La vérification essaie les exemples des docstrings, encode l'île et le sexe des manchots avec trois réglages de `drop` en comparant à scikit-learn, puis lance les tests des deux classes.

Dans tes notes : pourquoi `drop="first"` est-il utile à une régression linéaire sans pénalité, et inutile à un arbre ? Avec `handle_unknown="ignore"`, que devient l'inverse d'une île inconnue ?""",
       check=RELOAD + CHECK_26, solution=CHECK_26.replace('with wb.attempt("12.26"):\n', "if True:\n").replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", keyword="'),
       note="Les deux encodeurs partagent leur `fit` : pour chaque colonne, `np.unique` (avec `\"auto\"`) ou la liste "
            "donnée, vérifiée contre les valeurs de `X`. `OrdinalEncoder.transform` cherche la position de chaque "
            "valeur dans sa liste (un dictionnaire `{catégorie: code}` le fait vite). `OneHotEncoder.transform` compare "
            "chaque colonne à ses catégories (`colonne[:, None] == categories_[j]` donne d'un coup le bloc de 0/1), "
            "retire la colonne de `drop_idx_[j]` si elle existe, puis concatène les blocs. Retirer une colonne ne "
            "sert qu'aux modèles dont les poids ne sont pas uniques quand les colonnes d'un bloc somment à 1, comme "
            "une régression linéaire sans pénalité avec une ordonnée à l'origine ; un arbre de décision n'a pas ce "
            "problème. Avec `\"ignore\"`, une île inconnue donne un bloc de zéros, et l'inverse rend `None` : "
            "l'information « inconnue » est conservée, sa valeur non."),

    Ex("12.27", "🔨", 3, 60, "Coder la PCA (SVD, variance expliquée, whitening, reconstruction)",
       "écrire une PCA complète par SVD et la vérifier contre scikit-learn, au signe près.",
       "Ex 12.7 · ∂ 12.8 · Ex 12.16 · fiche §12.7.1 (encadrés 🧮 et pseudo-code)", thread="synthétique", tracks="M, C",
       mylearn="preprocessing.py",
       body=MYLEARN_SHORT + r"""

Écris la classe `PCA`. **Contrat** (la docstring, plus ces précisions que vérifient les tests) :
- `__init__(n_components=None, whiten=False)` range ses arguments tels quels ; `fit(X)` renvoie `self` et n'ajoute que des attributs `nom_` ou `_nom` ;
- `fit` apprend `mean_`, `components_` (une direction unitaire par ligne, rangées par variance décroissante), `explained_variance_` (ddof = 1), `explained_variance_ratio_` (la variance de chaque composante gardée divisée par la variance **totale**, celle de toutes les composantes), `singular_values_`, `n_components_` et `n_features_in_` ;
- le **signe** de chaque composante est fixé : sa coordonnée de plus grande valeur absolue est positive (les tests l'éprouvent sur des composantes dont la somme des coordonnées est négative) ;
- `n_components` : `None` garde $\min(n, p)$ composantes ; un entier $k$ entre 1 et $\min(n, p)$ en garde $k$ ; un réel strictement entre 0 et 1 garde le plus petit $k$ dont la part cumulée est **strictement** supérieure à ce réel ; toute autre valeur lève une `ValueError` (refuser un booléen n'est pas exigé) ; les tests utilisent toujours au moins 2 exemples ;
- `transform(X)` rend $(X - \mu)\,W_k^\top$, divisé colonne par colonne par $\sqrt{\lambda_j}$ si `whiten=True` ; `inverse_transform(Z)` défait le whitening puis rend $Z\,W_k + \mu$ ; `ValueError` si `X` n'a pas `n_features_in_` colonnes ou si `Z` n'a pas `n_components_` colonnes ; `RuntimeError` avant `fit`. Le whitening d'une composante de variance nulle n'est pas testé.

La vérification essaie l'exemple de la docstring, compare ta PCA à celle de scikit-learn sur un nuage de 200 points en 5 dimensions, essaie `n_components=0.9`, puis lance les tests.

Dans tes notes : pourquoi faut-il fixer le signe des composantes ? Avec `n_components=0.9`, pourquoi « strictement supérieure » compte-t-il ? Que vérifie le fait que `inverse_transform(transform(X))` redonne `X` avec toutes les composantes ?""",
       check=RELOAD + CHECK_27, solution=CHECK_27.replace('with wb.attempt("12.27"):\n', "if True:\n").replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", keyword="'),
       note="Le cœur tient en quelques lignes : centrer, `U, S, Vt = np.linalg.svd(Xc, full_matrices=False)`, "
            "fixer le signe de chaque ligne de `Vt` (multiplier par le signe de sa coordonnée de plus grande valeur "
            "absolue, et la colonne de `U` avec si on la garde), `explained_variance_ = S**2 / (n - 1)`, puis la part "
            "par rapport à la somme de **toutes** les variances avant de couper à $k$. Le signe d'un vecteur propre "
            "est arbitraire, et deux bibliothèques (ou deux versions de LAPACK) peuvent rendre $-\\mathbf{u}$ : sans "
            "convention, les coordonnées changeraient de signe d'une machine à l'autre. Avec `n_components=0.9`, si "
            "la part cumulée vaut exactement 0,9 pour un $k$, scikit-learn prend le $k$ suivant ; « supérieure ou "
            "égale » donnerait une composante de moins. Une reconstruction exacte avec toutes les composantes "
            "vérifie que `components_` forme une base orthonormée et que la moyenne est bien rajoutée."),

    Ex("12.28", "📦", 3, 40, "Chiffres propres : PCA sur MNIST et reconstructions",
       "voir les composantes d'une PCA d'images, et juger ses reconstructions sur des images qu'elle n'a pas vues.",
       "Ex 12.23 · fiche §12.7.2 (encadré ⚠️ : les reconstructions du livre)", thread="MNIST", tracks="R, C",
       body=r"""Le livre juge ses reconstructions de huskies sur les images mêmes qui ont servi à la PCA (fiche §12.7.2). Fais mieux : une PCA (`SkPCA`) apprise sur `TRAIN_28` (5 000 images d'entraînement), jugée aussi sur `TEST_28` (1 000 images de test). Écris :
- `errors_28(k)` : le couple (erreur d'entraînement, erreur de test) d'une PCA à $k$ composantes (`SkPCA(n_components=k, svd_solver="full")` : le calcul exact ; le réglage par défaut tire au hasard pour aller plus vite) ajustée sur `TRAIN_28`, l'erreur étant la moyenne, sur les images et les pixels, du carré de l'écart entre une image et sa reconstruction ;
- `baseline_28()` : l'erreur de test sans aucune composante, chaque image « reconstruite » par l'image moyenne de `TRAIN_28`.

La vérification calcule les erreurs pour $k$ dans `GRID_28` (5, 10, 20, 50, 100, 200, 400) et contrôle :
a) la première valeur de $k$ de la grille pour laquelle l'erreur de test dépasse l'erreur d'entraînement ;
b) le rapport de l'erreur de test à l'erreur d'entraînement pour $k = 400$ ;
c) la part de l'erreur de `baseline_28()` que supprime la PCA à 50 composantes : $1 - \text{erreur de test}(50) / \text{erreur}(0)$.

Puis elle affiche l'image moyenne, les sept premières composantes et des chiffres de test reconstruits avec 10, 50 et 200 composantes.

Dans tes notes : pourquoi l'erreur de test dépasse-t-elle l'erreur d'entraînement à partir d'un certain $k$, et pas avant ? Que montre la composante 1 ? À partir de quel $k$ reconnais-tu chaque chiffre ?""",
       given=GIVEN_28, todo=TODO_28, check=CHECK_28, solution=SOLUTION_28 + CHECK_28.replace('with wb.attempt("12.28"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''table_ref_28 = table_28                     # the table printed above: never recompute a recorded value
wb.record("12.28a", next(k for k, (tr, te) in table_ref_28.items() if te > tr),
          mistakes={"c'est la première valeur où l'erreur de test est PLUS PETITE : on cherche la première où elle dépasse celle d'entraînement": 5})
wb.record("12.28b", table_ref_28[400][1] / table_ref_28[400][0], decimals=4,
          mistakes={"c'est le rapport inverse : l'erreur de test divisée par l'erreur d'entraînement": table_ref_28[400][0] / table_ref_28[400][1]})
wb.record("12.28c", 1 - table_ref_28[50][1] / baseline_28(), decimals=4,
          mistakes={"c'est la part qui reste : on demande la part supprimée, 1 − erreur(50) / erreur(0)": table_ref_28[50][1] / baseline_28()})''',
       note="Avec peu de composantes, l'erreur de test est un peu **plus petite** que celle d'entraînement (les "
            "1 000 images de test sont un peu plus simples, en moyenne) : les grandes directions de variance se "
            "généralisent sans peine. À partir de 50 composantes, l'erreur de test dépasse celle d'entraînement, et "
            "l'écart grandit : à 400 composantes, l'erreur de test est environ 27 % plus grande. Les dernières "
            "composantes ont appris des détails des 5 000 images d'entraînement, qui ne se retrouvent pas ailleurs : "
            "c'est de l'overfitting de la PCA, invisible si l'on ne juge que l'entraînement, comme le livre. Avec "
            "50 composantes, la PCA supprime environ 83 % de l'erreur de l'image moyenne. La composante 1 oppose "
            "un trait vertical (un 1) à un rond (un 0) ; on reconnaît la plupart des chiffres dès 50 composantes, "
            "sur 784 pixels."),
])

# ---------------------------------------------------------------------------
# Part D: scale and gradient descent, 2D maps, leaks, a tested preparation, the challenge (12.29 to 12.33)
# ---------------------------------------------------------------------------
GIVEN_29 = r'''idx_29 = np.random.default_rng(1229).choice(len(CAL), 2000, replace=False)
X_29 = CAL[["MedInc", "HouseAge"]].to_numpy()[idx_29]          # 2,000 districts, two features in their units
Y_29 = CAL["MedHouseVal"].to_numpy()[idx_29]
LRS_29 = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
design_29 = np.hstack([X_29, np.ones((len(X_29), 1))])
BEST_29 = float(np.mean((design_29 @ np.linalg.lstsq(design_29, Y_29, rcond=None)[0] - Y_29) ** 2))   # least squares


def mse_grad_29(X, y, w, b):
    """The gradient of the MSE (1/n) * sum((X @ w + b - y) ** 2) with respect to w and to b, at (w, b)."""
    residual = X @ w + b - y
    return 2 * X.T @ residual / len(y), 2 * residual.mean()


print(f"smallest possible MSE (least squares): {BEST_29:.5f}")'''

TODO_29 = r'''def gd_29(X, y, lr, n_steps):
    """Gradient descent on the MSE of the linear model X @ w + b, from w = 0 and b = 0: at each step, both gradients
    from mse_grad_29 at the current (w, b), then w <- w - lr * grad_w and b <- b - lr * grad_b.
    Returns the list of the n_steps + 1 MSE values: before the first step, then after each step."""
    raise NotImplementedError("gd_29")


X_std_29 = ...          # X_29 standardized (fit on X_29)
lr_raw_29 = ...         # a) largest lr of LRS_29 that does not diverge on X_29 in 2,000 steps
lr_std_29 = ...         # b) the same on X_std_29
steps_29 = ...          # c) [steps on X_29 with lr_raw_29, steps on X_std_29 with lr_std_29] to reach 1.001 * BEST_29
same_optimum_29 = ...   # d) True or False: after 20,000 steps (with lr_raw_29 and lr_std_29), the same MSE?'''

SOLUTION_29 = r'''def gd_29(X, y, lr, n_steps):
    """Gradient descent on the MSE of the linear model X @ w + b, from w = 0 and b = 0: at each step, both gradients
    from mse_grad_29 at the current (w, b), then w <- w - lr * grad_w and b <- b - lr * grad_b.
    Returns the list of the n_steps + 1 MSE values: before the first step, then after each step."""
    w, b = np.zeros(X.shape[1]), 0.0
    losses = [float(np.mean((X @ w + b - y) ** 2))]
    with np.errstate(over="ignore", invalid="ignore"):         # a divergent run overflows: inf, then nan
        for _ in range(n_steps):
            grad_w, grad_b = mse_grad_29(X, y, w, b)
            w, b = w - lr * grad_w, b - lr * grad_b
            losses.append(float(np.mean((X @ w + b - y) ** 2)))
    return losses


def largest_stable_29(X):
    for lr in LRS_29:                                          # from the largest to the smallest
        losses = gd_29(X, Y_29, lr, 2000)
        if np.isfinite(losses[-1]) and losses[-1] < losses[0]:
            return lr


def steps_to_29(X, lr, target, n_steps=20_000):
    losses = gd_29(X, Y_29, lr, n_steps)
    return next(t for t, loss in enumerate(losses) if loss <= target)


X_std_29 = SkStandardScaler().fit_transform(X_29)
lr_raw_29, lr_std_29 = largest_stable_29(X_29), largest_stable_29(X_std_29)
steps_29 = [steps_to_29(X_29, lr_raw_29, 1.001 * BEST_29), steps_to_29(X_std_29, lr_std_29, 1.001 * BEST_29)]
same_optimum_29 = bool(np.isclose(gd_29(X_29, Y_29, lr_raw_29, 20_000)[-1], gd_29(X_std_29, Y_29, lr_std_29, 20_000)[-1],
                                  rtol=1e-4))
print(lr_raw_29, lr_std_29, steps_29, same_optimum_29)
'''

CHECK_29 = r'''with wb.attempt("12.29"):
    probe_29 = gd_29(np.array([[1.0], [2.0]]), np.array([2.0, 4.0]), 0.1, 2)
    if returned("12.29", "gd_29", probe_29):
        if not isinstance(probe_29, (list, tuple, np.ndarray)) or np.ndim(probe_29) != 1 or len(probe_29) != 3:
            print("❌ Ex 12.29 : gd_29 doit renvoyer la LISTE des n_steps + 1 MSE (ici 3 nombres), pas "
                  f"{type(probe_29).__name__} de forme {np.shape(probe_29)}.")
        elif not all(isinstance(value, (int, float, np.number)) for value in probe_29):
            print("❌ Ex 12.29 : gd_29 doit renvoyer une liste de nombres (les MSE).")
        elif np.allclose(probe_29, [10.0, 1.06, 0.1732]):
            print("✅ Ex 12.29 : gd_29 rend la loss initiale puis celle de chaque pas.")
        elif np.allclose(probe_29, [10.0, 1.69, 0.306], atol=1e-3):
            print("❌ Ex 12.29 : w et b se mettent à jour ENSEMBLE, avec les deux gradients du même point (b est "
                  "calculé ici avec le w déjà modifié).")
        else:
            print(f"❌ Ex 12.29 : gd_29([[1], [2]], [2, 4], lr=0.1, n_steps=2) rend {probe_29} au lieu de "
                  "[10, 1.06, 0.1732] : pars de w = 0 et b = 0, et garde la loss avant le premier pas.")
wb.check("12.29a", lr_raw_29)
wb.check("12.29b", lr_std_29)
wb.check("12.29c", steps_29)
wb.check("12.29d", same_optimum_29)
with wb.attempt("12.29"):
    if not filled(X_std_29, lr_raw_29, lr_std_29):
        raise NotImplementedError("X_std_29, lr_raw_29, lr_std_29")
    probe_29 = globals().get("probe_29")
    if not isinstance(probe_29, (list, tuple, np.ndarray)) or np.ndim(probe_29) != 1 or len(probe_29) != 3:
        print("⏳ Ex 12.29 : la courbe attend une gd_29 qui renvoie la liste des MSE.")
    else:
        for label, X, lr in [("raw features", X_29, lr_raw_29), ("standardized", X_std_29, lr_std_29)]:
            losses = np.asarray(gd_29(X, Y_29, float(lr), 3000), dtype=float)
            plt.plot(np.arange(len(losses)), losses / BEST_29 - 1, label=f"{label}, lr = {lr}")
        plt.xscale("symlog", linthresh=10)
        plt.yscale("log")
        plt.axhline(0.001, color="gray", lw=0.8, ls="--")
        plt.xlabel("step")
        plt.ylabel("MSE / smallest MSE - 1")
        plt.legend()
    plt.show()'''

GIVEN_30 = r'''with warnings.catch_warnings():
    warnings.simplefilter("ignore")        # umap imports tqdm, which warns about Jupyter widgets
    import umap
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

N_30 = wb.by_mode(fast=2000, full=5000)
IMAGES_30, LABELS_30 = wb.datasets.load_mnist("train", n=N_30, flatten=True, seed=1230)
IMAGES_30 = IMAGES_30 / 255.0
print(IMAGES_30.shape)'''

TODO_30 = r'''def embed_30(method, X):
    """A 2D map of the rows of X: method is "PCA" (2 components), "t-SNE" (perplexity 30, init="pca",
    random_state=0, on the first 50 components of a PCA) or "UMAP" (n_neighbors=15, random_state=0).
    Returns an array of shape (len(X), 2)."""
    raise NotImplementedError("embed_30")'''

SOLUTION_30 = r'''def embed_30(method, X):
    """A 2D map of the rows of X: method is "PCA" (2 components), "t-SNE" (perplexity 30, init="pca",
    random_state=0, on the first 50 components of a PCA) or "UMAP" (n_neighbors=15, random_state=0).
    Returns an array of shape (len(X), 2)."""
    if method == "PCA":
        return SkPCA(n_components=2).fit_transform(X)
    if method == "t-SNE":
        return TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
            SkPCA(n_components=50).fit_transform(X))
    if method == "UMAP":
        return umap.UMAP(n_neighbors=15, random_state=0).fit_transform(X)
    raise ValueError(f"unknown method: {method}")

'''

CHECK_30 = r'''with wb.attempt("12.30"):
    maps_30, scores_30 = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # UMAP warns that random_state disables parallelism
        for method in ["PCA", "t-SNE", "UMAP"]:
            start = time.perf_counter()
            embedding = embed_30(method, IMAGES_30)
            if not returned("12.30", "embed_30", embedding):
                break
            embedding = np.asarray(embedding)
            if embedding.shape != (len(IMAGES_30), 2):
                print(f"❌ Ex 12.30 : embed_30(\"{method}\", ...) doit rendre un tableau ({len(IMAGES_30)}, 2) ; reçu {embedding.shape}.")
                break
            seconds = time.perf_counter() - start
            accuracy = cross_val_score(KNeighborsClassifier(n_neighbors=5), embedding, LABELS_30, cv=5).mean()
            trust = trustworthiness(IMAGES_30, embedding, n_neighbors=10)
            maps_30[method], scores_30[method] = embedding, (accuracy, trust)
            print(f"{method:6s} {seconds:5.1f} s · 5-NN accuracy in 2D {accuracy:.3f} · trustworthiness {trust:.3f}")
    if len(maps_30) == 3:
        fig_30, axes_30 = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (method, embedding) in zip(axes_30, maps_30.items()):
            ax.scatter(*embedding.T, c=LABELS_30, cmap="tab10", s=4)
            for digit in range(10):
                ax.text(*np.median(embedding[LABELS_30 == digit], axis=0), str(digit), fontsize=14, weight="bold")
            ax.set(title=f"{method}: 5-NN {scores_30[method][0]:.2f}", xticks=[], yticks=[])
        plt.show()
        verdict("12.30", min(scores_30["t-SNE"][0], scores_30["UMAP"][0]) > scores_30["PCA"][0] + 0.3,
                "t-SNE et UMAP gardent bien mieux les voisinages que la PCA à deux composantes.",
                "les accuracies 5-NN ne montrent pas l'écart attendu entre la PCA et les deux autres méthodes : "
                "vérifie les réglages de embed_30.")'''

GIVEN_31 = r'''from sklearn.compose import make_column_transformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

rng_31 = np.random.default_rng(1231)
values_31 = rng_31.normal(size=(300, 4)) * [1.0, 10.0, 100.0, 0.1] + [0.0, 50.0, 1000.0, 0.0]
Y_31 = (rng_31.random(300) < 1 / (1 + np.exp(-(0.8 * values_31[:, 0] - 0.08 * (values_31[:, 1] - 50.0))))).astype(int)
values_31[rng_31.random((300, 4)) < 0.15] = np.nan                       # 15 % of missing values
NUM_31 = ["x0", "x1", "x2", "x3"]
DATA_31 = pd.DataFrame(values_31, columns=NUM_31)
DATA_31["code"] = [f"c{c:03d}" for c in rng_31.integers(0, 150, size=300)]   # 150 codes, unrelated to the target
CV_31 = StratifiedKFold(n_splits=5, shuffle=True, random_state=1231)


def numeric_31():
    """The preparation of the numeric columns: imputation by the mean, then standardization."""
    return make_pipeline(SkSimpleImputer(), SkStandardScaler())


def encoder_31():
    """The encoder of the column "code": each code replaced by a shrunk mean of the target in that code."""
    return TargetEncoder(random_state=0)


def cv_auc_31(model, X):
    """Mean ROC AUC of model on X (a DataFrame) and Y_31, with the folds CV_31."""
    return float(cross_val_score(model, X, Y_31, cv=CV_31, scoring="roc_auc").mean())


print(DATA_31.head(3), "\nshare of positives:", Y_31.mean().round(3), "· distinct codes:", DATA_31["code"].nunique())'''

TODO_31 = r'''def auc_31(variant):
    """The mean ROC AUC (cv_auc_31) of LogisticRegression() after the preparation of `variant`, "A" to "D"."""
    raise NotImplementedError("auc_31")


largest_31 = ...          # b) the letter of the most optimistic variant
scaler_small_31 = ...     # c) True or False: is |AUC(B) - AUC(A)| below 0.01?
crossfit_small_31 = ...   # d) True or False: is |AUC(D) - AUC(A)| below 0.01?'''

SOLUTION_31 = r'''def auc_31(variant):
    """The mean ROC AUC (cv_auc_31) of LogisticRegression() after the preparation of `variant`, "A" to "D"."""
    if variant == "A":                                   # everything inside the cross-validation
        prep = make_column_transformer((numeric_31(), NUM_31), (encoder_31(), ["code"]))
        return cv_auc_31(make_pipeline(prep, LogisticRegression()), DATA_31)
    if variant == "B":                                   # imputer and scaler fitted on all the rows first
        X = pd.DataFrame(numeric_31().fit_transform(DATA_31[NUM_31]), columns=NUM_31)
        X["code"] = DATA_31["code"].to_numpy()
        prep = make_column_transformer(("passthrough", NUM_31), (encoder_31(), ["code"]))
        return cv_auc_31(make_pipeline(prep, LogisticRegression()), X)
    if variant in ("C", "D"):                            # the encoder fitted on all the rows first
        codes = DATA_31[["code"]]
        encoded = (encoder_31().fit(codes, Y_31).transform(codes) if variant == "C"
                   else encoder_31().fit_transform(codes, Y_31))
        X = DATA_31[NUM_31].copy()
        X["code_te"] = encoded.ravel()
        prep = make_column_transformer((numeric_31(), NUM_31), ("passthrough", ["code_te"]))
        return cv_auc_31(make_pipeline(prep, LogisticRegression()), X)
    raise ValueError(f"unknown variant: {variant}")


aucs_ref_31 = {variant: auc_31(variant) for variant in "ABCD"}
largest_31 = max(aucs_ref_31, key=aucs_ref_31.get)
scaler_small_31 = abs(aucs_ref_31["B"] - aucs_ref_31["A"]) < 0.01
crossfit_small_31 = abs(aucs_ref_31["D"] - aucs_ref_31["A"]) < 0.01
print({variant: round(auc, 4) for variant, auc in aucs_ref_31.items()}, largest_31, scaler_small_31, crossfit_small_31)
'''

CHECK_31 = r'''with wb.attempt("12.31"):
    aucs_31 = [auc_31(variant) for variant in "ABCD"]
    if not all(returned("12.31", "auc_31", auc) for auc in aucs_31):
        pass
    elif not all(np.ndim(auc) == 0 and np.issubdtype(np.asarray(auc).dtype, np.number) for auc in aucs_31):
        print("❌ Ex 12.31 : auc_31 doit renvoyer UN nombre, la moyenne des AUC des cinq folds (reçu : "
              f"{type(aucs_31[0]).__name__} de forme {np.shape(aucs_31[0])}).")
    else:
        for variant, auc in zip("ABCD", aucs_31):
            print(f"{variant}: mean ROC AUC {auc:.4f}")
        wb.check("12.31a", [float(auc) for auc in aucs_31], computed=True)
wb.check("12.31b", largest_31)
wb.check("12.31c", scaler_small_31)
wb.check("12.31d", crossfit_small_31)'''

GIVEN_32 = r'''from sklearn.impute import SimpleImputer                      # the names used by prepare_32 and its tests
from sklearn.preprocessing import OneHotEncoder, StandardScaler

P_RAW_32 = wb.datasets.load_penguins(dropna=False)                   # 344 penguins, with missing values
TRAIN_32, TEST_32 = train_test_split(P_RAW_32, test_size=100, random_state=1232)
PREAMBLE_32 = """import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_32 = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_32 = ["island", "sex"]"""
exec(PREAMBLE_32)                  # the same names in the notebook and in the test file


def prepare_bug_1(train, other):
    both = pd.concat([train, other])[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
    imputer = SimpleImputer(strategy="median").fit(both)
    scaler = StandardScaler().fit(imputer.transform(both))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["island", "sex"]].fillna("unknown"))
    prep = lambda df: np.hstack([scaler.transform(imputer.transform(df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])),
                                 encoder.transform(df[["island", "sex"]].fillna("unknown"))])
    return prep(train), prep(other)


def prepare_bug_2(train, other):
    def fit_and_prepare(df):
        numeric = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
        imputer = SimpleImputer(strategy="median").fit(numeric)
        scaler = StandardScaler().fit(imputer.transform(numeric))
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["island", "sex"]].fillna("unknown"))
        return np.hstack([scaler.transform(imputer.transform(numeric)), encoder.transform(df[["island", "sex"]].fillna("unknown"))])
    return fit_and_prepare(train), fit_and_prepare(other)


def prepare_bug_3(train, other):
    imputer = SimpleImputer(strategy="median").fit(train[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
    scaler = StandardScaler().fit(imputer.transform(train[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]))
    prep = lambda df: np.hstack([scaler.transform(imputer.transform(df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])),
                                 pd.get_dummies(df[["island", "sex"]].fillna("unknown"), dtype=float).to_numpy()])
    return prep(train), prep(other)


def prepare_bug_4(train, other):
    imputer = SimpleImputer(strategy="mean").fit(train[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
    scaler = StandardScaler().fit(imputer.transform(train[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["island", "sex"]].fillna("unknown"))
    prep = lambda df: np.hstack([scaler.transform(imputer.transform(df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])),
                                 encoder.transform(df[["island", "sex"]].fillna("unknown"))])
    return prep(train), prep(other)


def prepare_bug_5(train, other):
    train["sex"] = train["sex"].fillna("unknown")          # changes the caller's DataFrame
    other["sex"] = other["sex"].fillna("unknown")
    imputer = SimpleImputer(strategy="median").fit(train[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])
    scaler = StandardScaler().fit(imputer.transform(train[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["island", "sex"]])
    prep = lambda df: np.hstack([scaler.transform(imputer.transform(df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]])),
                                 encoder.transform(df[["island", "sex"]])])
    return prep(train), prep(other)


BUGS_32 = [prepare_bug_1, prepare_bug_2, prepare_bug_3, prepare_bug_4, prepare_bug_5]'''

TODO_32 = r'''def prepare_32(train, other):
    """Write the docstring: what the function does, its parameters, what it returns."""
    raise NotImplementedError("prepare_32")


# Write your tests here: each one calls prepare_32(...) on small DataFrames that you build, and checks one property
# with assert. Then list them in TESTS_32, e.g. TESTS_32 = [test_one_property, test_another_property].

TESTS_32 = ...'''

SOLUTION_32 = r'''def prepare_32(train, other):
    """Prepare penguins for a model, with parameters learnt on `train` only.

    The 4 measures (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) are imputed by their median,
    then standardized; island and sex (a missing sex becomes the category "unknown") are one-hot encoded, an
    unknown category giving a block of zeros. Every statistic and every category comes from `train`; `other` is
    only transformed, row by row. Neither DataFrame is modified.

    Parameters
    ----------
    train : DataFrame with the columns of penguins.csv
        The training penguins: the parameters are learnt here.
    other : DataFrame with the same columns
        Penguins to transform with the parameters of train (validation, test, production).

    Returns
    -------
    (X_train, X_other) : two float arrays with the same columns
        The 4 standardized measures, then the one-hot columns of island and sex.
    """
    measures = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    imputer = SimpleImputer(strategy="median").fit(train[measures])
    scaler = StandardScaler().fit(imputer.transform(train[measures]))
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["island", "sex"]].fillna("unknown"))

    def prepare(df):
        numeric = scaler.transform(imputer.transform(df[measures]))
        return np.hstack([numeric, encoder.transform(df[["island", "sex"]].fillna("unknown"))])

    return prepare(train), prepare(other)


def small_32(rows):
    """A small DataFrame of penguins from (bill, depth, flipper, mass, island, sex) tuples."""
    return pd.DataFrame(rows, columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "island", "sex"])


def test_shapes_and_no_missing_values():
    train = small_32([(40, 18, 190, 3800, "Dream", "male"), (46, None, 210, 5000, "Biscoe", None), (38, 17, 185, None, "Dream", "female")])
    X_train, X_other = prepare_32(train, train.iloc[:2])
    assert X_train.shape[0] == 3 and X_other.shape[0] == 2 and X_train.shape[1] == X_other.shape[1]
    assert not np.isnan(X_train).any() and not np.isnan(X_other).any()


def test_train_measures_are_standardized():
    train = small_32([(40, 18, 190, 3800, "Dream", "male"), (46, 15, 210, 5000, "Biscoe", "female"), (38, 17, 185, 3500, "Dream", "female")])
    X_train, _ = prepare_32(train, train)
    assert np.allclose(X_train[:, :4].mean(axis=0), 0) and np.allclose(X_train[:, :4].std(axis=0), 1)


def test_train_part_does_not_depend_on_other():
    train = small_32([(40, 18, 190, 3800, "Dream", "male"), (46, 15, 210, 5000, "Biscoe", "female"), (38, 17, 185, 3500, "Dream", "female")])
    other = small_32([(50, 14, 230, 6000, "Biscoe", "male")])
    huge = small_32([(5000, 1400, 23000, 600000, "Biscoe", "male")])
    assert np.allclose(prepare_32(train, other)[0], prepare_32(train, huge)[0])


def test_a_row_of_other_does_not_depend_on_the_other_rows():
    train = small_32([(40, 18, 190, 3800, "Dream", "male"), (46, 15, 210, 5000, "Biscoe", "female"), (38, 17, 185, 3500, "Dream", "female")])
    other = small_32([(50, 14, 230, 6000, "Biscoe", "male"), (39, 19, 188, 3600, "Dream", "female")])
    assert np.allclose(prepare_32(train, other)[1][:1], prepare_32(train, other.iloc[:1])[1])


def test_missing_value_gets_the_train_median():
    train = small_32([(30, 18, 190, 3800, "Dream", "male"), (40, 15, 210, 5000, "Biscoe", "female"), (80, 17, 185, 3500, "Dream", "female")])
    other = small_32([(None, 18, 190, 3800, "Dream", "male")])
    X_train, X_other = prepare_32(train, other)
    assert np.isclose(X_other[0, 0], X_train[1, 0])           # the median of 30, 40, 80 is the bill of row 1


def test_columns_are_aligned_when_a_category_is_missing():
    train = small_32([(40, 18, 190, 3800, "Dream", "male"), (46, 15, 210, 5000, "Biscoe", "female"), (38, 17, 185, 3500, "Torgersen", "female")])
    X_train, X_other = prepare_32(train, train.iloc[[0]])
    assert np.allclose(X_other, X_train[[0]])


def test_inputs_are_not_modified():
    train = small_32([(40, 18, 190, 3800, "Dream", None), (46, 15, 210, 5000, "Biscoe", "female"), (38, None, 185, 3500, "Dream", "female")])
    other = small_32([(50, 14, 230, None, "Biscoe", None)])
    train_copy, other_copy = train.copy(), other.copy()
    prepare_32(train, other)
    pd.testing.assert_frame_equal(train, train_copy)
    pd.testing.assert_frame_equal(other, other_copy)


TESTS_32 = [test_shapes_and_no_missing_values, test_train_measures_are_standardized, test_train_part_does_not_depend_on_other,
            test_a_row_of_other_does_not_depend_on_the_other_rows, test_missing_value_gets_the_train_median,
            test_columns_are_aligned_when_a_category_is_missing, test_inputs_are_not_modified]

'''

CHECK_32 = r'''with wb.attempt("12.32"):
    X_train_32, X_test_32 = prepare_32(TRAIN_32.copy(), TEST_32.copy())
    print("shapes:", np.shape(X_train_32), np.shape(X_test_32))
    doc_32 = (prepare_32.__doc__ or "").strip()
    verdict("12.32", len(doc_32.splitlines()) >= 3 and ("Returns" in doc_32 or "Renvoie" in doc_32),
            "prepare_32 a une docstring qui dit ce qu'elle renvoie.",
            "écris une docstring de plusieurs lignes, avec une section Returns (ou Renvoie).")
    if not filled(TESTS_32):
        raise NotImplementedError
    tests_32 = list(TESTS_32) if isinstance(TESTS_32, (list, tuple)) else [TESTS_32]
    names_32 = [getattr(test, "__name__", repr(test)) for test in tests_32]
    good_32 = None
    if len(tests_32) < 4:
        print(f"❌ Ex 12.32 : écris au moins quatre tests ; TESTS_32 n'en contient que {len(tests_32)}.")
    if not all(callable(test) for test in tests_32):
        print("❌ Ex 12.32 : TESTS_32 doit être une liste de fonctions de test, pas de leurs noms ni de leurs résultats.")
    elif not all(name.startswith("test_") for name in names_32):
        print(f"❌ Ex 12.32 : pytest ne lance que les fonctions dont le nom commence par test_ : renomme "
              f"{[name for name in names_32 if not name.startswith('test_')]}.")
    else:
        helpers_32 = [obj for name, obj in list(globals().items()) if name.startswith("small_32") and callable(obj)]
        good_32 = wb.run_pytest(tests_32, subject=prepare_32, name="prepare_32", extra=helpers_32,
                                preamble=PREAMBLE_32, quiet=True)
        if "NameError" in good_32.output:
            print("❌ Ex 12.32 : un nom est inconnu dans le fichier de test : il ne contient que math, numpy (np), "
                  "pandas (pd), pytest, SimpleImputer, OneHotEncoder, StandardScaler, NUMERIC_32, CATEGORICAL_32, ta "
                  "fonction, tes tests et tes fonctions dont le nom commence par small_32.")
        verdict("12.32", good_32.ok, f"tes {good_32.passed} tests passent sur ta fonction.",
                f"un de tes tests échoue sur ta fonction : le test ou la fonction est faux ({good_32.summary}).")
    if good_32 is not None and not good_32.ok:
        print(good_32.output[-1500:])
    elif good_32 is not None:
        for bug_32 in BUGS_32:
            caught_32 = wb.run_pytest(tests_32, subject=bug_32, name="prepare_32", extra=helpers_32,
                                      preamble=PREAMBLE_32, quiet=True)
            verdict("12.32", not caught_32.ok, f"{bug_32.__name__} attrapée ({caught_32.failed} test(s) en échec).",
                    f"{bug_32.__name__} passe tous tes tests : compare son code à ta fonction, et ajoute le test d'une "
                    "propriété qu'elle viole.")'''

GIVEN_33 = r'''from sklearn.model_selection import StratifiedKFold as SkStratifiedKFold

PUBLIC_33 = [0, 1, 2]                      # 3 repetitions of a 5-fold cross-validation: as often as you like
HIDDEN_33 = list(range(3300, 3310))        # 10 other repetitions: revealed once per session (READY_33 = True)
REVEALED_33 = globals().get("REVEALED_33", {})   # kept when this cell runs again
DATA_33 = (RAW.drop(columns=["Sample Number", "Individual ID", "Comments", "Date Egg"])   # numbering, comments, the
           .sample(frac=1.0, random_state=1233).reset_index(drop=True))   # egg date of the nest, the row order: not measures
print(list(DATA_33.columns))


def leak_checks_33(prepare):
    """The rules of the challenge, checked on DATA_33; returns the list of the broken rules (empty if all hold)."""
    problems = []
    train, new = DATA_33.iloc[:250], DATA_33.iloc[250:].drop(columns="Sex")
    train_copy, new_copy = train.copy(), new.copy()
    try:
        X_train, X_new = (np.asarray(part, dtype=float) for part in prepare(train, new))
    except Exception as error:
        return [f"prepare_33 lève {type(error).__name__} : {error}"]
    if X_train.ndim != 2 or X_new.ndim != 2 or X_train.shape[1] != X_new.shape[1]:
        return ["prepare_33 doit rendre deux tableaux 2-D qui ont le même nombre de colonnes"]
    if len(X_train) != len(train) or len(X_new) != len(new):
        problems.append("une ligne par ligne reçue, sans en retirer aucune")
    if not (np.isfinite(X_train).all() and np.isfinite(X_new).all()):
        problems.append("aucune valeur manquante ni infinie dans ce que rend prepare_33")
    if not (train.equals(train_copy) and new.equals(new_copy)):
        problems.append("prepare_33 ne modifie pas les DataFrames reçus")
    if not problems:
        try:
            other = DATA_33.iloc[300:].drop(columns="Sex")
            if not np.allclose(X_train, np.asarray(prepare(train, other)[0], dtype=float)):
                problems.append("ce qui est appris ne dépend que de raw_train : X_train change avec raw_new")
            alone = [0, 1, 2] + [i for i in range(len(new)) if new.iloc[i].isna().any()]   # rows with missing values too
            if not all(np.allclose(X_new[i], np.asarray(prepare(train, new.iloc[[i]])[1], dtype=float)[0]) for i in alone):
                problems.append("chaque ligne de raw_new se prépare seule : elle change avec les autres lignes du batch")
        except Exception as error:
            problems.append("chaque ligne de raw_new se prépare seule, avec ce qui est appris sur raw_train : prepare_33 "
                            f"lève {type(error).__name__} sur un autre batch ({error})")
    return problems


def cv_accuracy_33(prepare, seeds):
    """Mean accuracy of LogisticRegression(max_iter=1000) on the sex, over 5-fold cross-validations of DATA_33 (one per
    seed); the penguins of unknown sex are prepared, but neither learnt nor scored."""
    sex = DATA_33["Sex"].fillna("unknown")
    scores = []
    for seed in seeds:
        for train_idx, new_idx in SkStratifiedKFold(5, shuffle=True, random_state=seed).split(DATA_33, sex):
            train, new = DATA_33.iloc[train_idx], DATA_33.iloc[new_idx]
            X_train, X_new = (np.asarray(part, dtype=float) for part in prepare(train, new.drop(columns="Sex")))
            known_train, known_new = train["Sex"].notna().to_numpy(), new["Sex"].notna().to_numpy()
            model = LogisticRegression(max_iter=1000).fit(X_train[known_train], train["Sex"][known_train])
            scores.append(np.mean(model.predict(X_new[known_new]) == new["Sex"][known_new].to_numpy()))
    return float(np.mean(scores))


def grade_33(prepare, reveal=False):
    """The rules, the public score at every call, the hidden score once per session (reveal=True)."""
    problems = leak_checks_33(prepare)
    for problem in problems:
        print(f"❌ Ex 12.33 : règle non respectée : {problem}.")
    if problems:
        return
    print("✅ Ex 12.33 : les règles du défi sont respectées.")
    print(f"public score (3 x 5 folds): accuracy {cv_accuracy_33(prepare, PUBLIC_33):.4f}")
    if not reveal:
        print("the hidden score (10 other repetitions) is revealed once per session, when READY_33 = True")
        return
    if "hidden" in REVEALED_33:
        print(f"hidden score already revealed in this session: {REVEALED_33['hidden']:.4f} (frozen)")
        return
    REVEALED_33["hidden"] = hidden = cv_accuracy_33(prepare, HIDDEN_33)
    print(f"hidden score (10 x 5 folds): accuracy {hidden:.4f}")
    verdict("12.33", hidden >= 0.92, "objectif atteint : au moins 0,92 sur le score caché.",
            "objectif manqué (au moins 0,92) : retravaille ta préparation sur le score public, sans plus regarder "
            "le score caché.")
    if hidden >= 0.93:
        print("🌟 palier atteint : au moins 0,93.")'''

TODO_33 = r'''def prepare_33(raw_train, raw_new):
    """Your preparation: (raw_train, raw_new) -> (X_train, X_new), two float arrays, one row per row received.
    raw_train has all the columns of DATA_33; raw_new the same, without "Sex". The starting point: the 4 measures,
    imputed by their median and standardized, with parameters learnt on raw_train."""
    measures = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)"]
    imputer = SkSimpleImputer(strategy="median").fit(raw_train[measures])
    scaler = SkStandardScaler().fit(imputer.transform(raw_train[measures]))
    prepare = lambda df: scaler.transform(imputer.transform(df[measures]))
    return prepare(raw_train), prepare(raw_new)


READY_33 = False       # True once your preparation is chosen on the public score: reveals the hidden score (once)'''

SOLUTION_33 = r'''def prepare_33(raw_train, raw_new):
    """Measures and isotopes standardized WITHIN each species (median imputation, mean and standard deviation of
    the species, learnt on raw_train), then the species in one-hot: a male is heavy for HIS species."""
    columns = ["Culmen Length (mm)", "Culmen Depth (mm)", "Flipper Length (mm)", "Body Mass (g)",
               "Delta 15 N (o/oo)", "Delta 13 C (o/oo)"]
    species = sorted(raw_train["Species"].unique())
    stats = {}
    for name, group in raw_train.groupby("Species"):
        median = group[columns].median()
        filled = group[columns].fillna(median)
        stats[name] = (median, filled.mean(), filled.std(ddof=0))
    overall = raw_train[columns].median()

    def prepare(df):
        rows = []
        for name, values in zip(df["Species"], df[columns].itertuples(index=False)):
            median, mean, std = stats.get(name, (overall, overall, pd.Series(1.0, index=columns)))
            x = pd.Series(values, index=columns).fillna(median)
            rows.append(((x - mean) / std).to_numpy(dtype=float))
        one_hot = (df["Species"].to_numpy()[:, None] == np.array(species)[None, :]).astype(float)
        return np.hstack([np.array(rows), one_hot])

    return prepare(raw_train), prepare(raw_new)


READY_33 = True

'''

CHECK_33 = r'''with wb.attempt("12.33"):
    grade_33(prepare_33, reveal=READY_33)'''

PART_D = Part("D", "Échelle, cartes 2D, fuites, et le défi",
              "Ce que coûte une mauvaise échelle à une descente de gradient (12.29), trois façons de voir MNIST en "
              "2D (12.30), puis la règle d'or appliquée jusqu'au bout : mesurer les fuites dues au prétraitement "
              "(12.31), écrire une préparation documentée et le test qui la protège (12.32), et préparer le fichier "
              "brut des manchots pour un modèle, sans fuite (12.33).",
              exercises=[
    Ex("12.29", "🔬", 3, 40, "Échelle des features et descente de gradient",
       "mesurer ce que la standardisation change pour une descente de gradient, et ce qu'elle ne change pas.",
       "Ex 12.16 · ch. 5 (descente de gradient, learning rate) · ch. 9 (MSE) · 🔁 12.R3 · fiche §12.5",
       thread="California", tracks="M, C",
       body=r"""Une régression linéaire de la valeur des logements (`MedHouseVal`) sur deux features de 2 000 districts : `MedInc` (en dizaines de milliers de dollars, autour de 4) et `HouseAge` (en années, autour de 29). `BEST_29` est la plus petite MSE possible (moindres carrés), et `mse_grad_29` donne le gradient de la MSE (tu n'as pas à le dériver). Écris :
- `gd_29(X, y, lr, n_steps)` : la descente de gradient sur la MSE de `X @ w + b`, partant de `w = 0` et `b = 0` ; à chaque pas, les deux gradients au point courant, puis les deux mises à jour **ensemble** ; la fonction renvoie la liste des `n_steps + 1` MSE (avant le premier pas, puis après chaque pas) ;
- `X_std_29` : `X_29` standardisé ;
- a) `lr_raw_29` : le plus grand learning rate de `LRS_29` pour lequel la descente sur `X_29` ne **diverge** pas en 2 000 pas (une dernière MSE finie et plus petite que la première) ;
- b) `lr_std_29` : la même chose sur `X_std_29` ;
- c) `steps_29` : la liste [pas sur `X_29` avec `lr_raw_29`, pas sur `X_std_29` avec `lr_std_29`] qu'il faut pour descendre sous `1.001 * BEST_29` (au plus 20 000 pas) ;
- d) `same_optimum_29` : après 20 000 pas chacune (avec `lr_raw_29` et `lr_std_29`), les deux descentes arrivent-elles à la même MSE, à $10^{-4}$ près en relatif (`np.isclose(..., rtol=1e-4)`) ? (`True` ou `False`)

La vérification essaie `gd_29` sur un tout petit exemple, contrôle a) à d), puis trace les deux courbes.

Dans tes notes : relie a) et b) à la forme de la surface de la loss (🔁 12.R3). Pourquoi la standardisation ne change-t-elle pas la plus petite MSE possible ? Que deviennent les poids de `X_29` quand on les compare à ceux de `X_std_29` ?""",
       given=GIVEN_29, todo=TODO_29, check=CHECK_29, solution=SOLUTION_29 + CHECK_29.replace('with wb.attempt("12.29"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''wb.record("12.29a", lr_raw_29, decimals=4, choices=LRS_29,
          mistakes={"avec ce learning rate, la MSE explose : prends le plus grand qui ne diverge pas": 0.003})
wb.record("12.29b", lr_std_29, decimals=4, choices=LRS_29,
          mistakes={"avec ce learning rate, la MSE explose : prends le plus grand qui ne diverge pas": 1.0})
wb.record("12.29c", steps_29, mistakes={"les deux nombres sont dans l'ordre [features brutes, standardisées]": steps_29[::-1]})
wb.record("12.29d", same_optimum_29, mistakes={"une standardisation est une transformation affine de chaque feature : un modèle linéaire peut-il la compenser ?": False})''',
       note="Sur les features brutes, la surface de la MSE est une vallée très allongée : sa courbure est environ "
            "1 960 dans la direction de `HouseAge` (des nombres autour de 29, au carré) et 0,18 dans la plus plate, un "
            "rapport de 10 000. Le learning rate doit rester sous $2/1\\,960 \\approx 0{,}00102$ pour ne pas diverger dans la "
            "direction raide ; avec lui, la direction plate avance d'un pas minuscule, et il faut environ 2 300 pas "
            "pour approcher la plus petite MSE à 0,1 % près. Standardisées, les features donnent une cuvette presque "
            "ronde (courbures 1,8 à 2,2) : un learning rate de 0,3 convient, et 6 pas suffisent. La plus petite MSE "
            "ne change pas : une standardisation est une transformation affine de chaque feature, qu'un modèle "
            "linéaire compense exactement par ses poids et son ordonnée à l'origine (les poids de `X_29` valent "
            "ceux de `X_std_29` divisés par les écarts-types). La standardisation ne change pas le modèle optimal, "
            "seulement la difficulté de le trouver."),

    Ex("12.30", "📦", 3, 40, "PCA, t-SNE ou UMAP pour voir MNIST en 2D",
       "comparer trois cartes en 2D d'un dataset d'images, et mesurer ce qu'elles gardent des voisinages.",
       "Ex 12.28 · fiche « Au-delà du livre » (t-SNE et UMAP)", thread="MNIST", tracks="C",
       body=r"""Écris `embed_30(method, X)`, qui rend une carte en 2D (un tableau de forme `(len(X), 2)`) des lignes de `X` :
- `"PCA"` : les deux premières composantes (`SkPCA`) ;
- `"t-SNE"` : `TSNE(n_components=2, perplexity=30, init="pca", random_state=0)`, appliqué aux 50 premières composantes d'une PCA (la réduction préalable que conseille scikit-learn) ;
- `"UMAP"` : `umap.UMAP(n_neighbors=15, random_state=0)`.

La vérification construit les trois cartes de 2 000 images (5 000 hors FAST_MODE), mesure le temps de calcul, l'accuracy d'un classifieur des 5 plus proches voisins **dans la carte** (validation croisée à 5 folds) et la *trustworthiness* (`sklearn.manifold.trustworthiness`, de 0 à 1 : à peu près la part des voisins d'un point sur la carte qui étaient déjà ses voisins dans l'espace des 784 pixels, les intrus étant pénalisés selon leur rang), puis trace les trois cartes. Cet exercice se corrige avec `05_solutions.md`.

Dans tes notes : quelle carte sépare le mieux les chiffres ? Quels chiffres restent mêlés sur la carte de la PCA, et lesquels se touchent sur les deux autres ? Peut-on dire, sur la carte de t-SNE, que deux groupes éloignés sont « très différents » ? Pourquoi ne pas entraîner un classifieur sur les coordonnées de t-SNE pour prédire de nouvelles images ?""",
       given=GIVEN_30, todo=TODO_30, check=CHECK_30, solution=SOLUTION_30 + CHECK_30.replace('with wb.attempt("12.30"):\n', "if True:\n"),
       note="La PCA à deux composantes ne garde que 17 % de la variance : les chiffres se superposent, et un "
            "classifieur des 5 plus proches voisins sur la carte ne fait qu'environ 40 %. t-SNE et UMAP gardent les "
            "voisinages (trustworthiness de 0,95 à 0,97) : 86 à 90 % d'accuracy en 2D, des groupes nets, avec "
            "les 4, 7 et 9 souvent proches, et des 3, 5 et 8 qui se touchent. Mais la taille des groupes et les "
            "distances entre groupes n'y ont pas de sens précis : deux groupes éloignés ne sont pas forcément plus "
            "différents que deux groupes voisins. t-SNE n'a pas de `transform` : une image nouvelle ne se place pas "
            "sur une carte existante, et la carte change avec les données et la graine ; UMAP a un `transform`, "
            "mais ses coordonnées restent faites pour regarder, pas pour mesurer."),

    Ex("12.31", "🔬", 3, 45, "Fuite de données par le prétraitement : scaler, imputation et TargetEncoder ajustés avant le découpage",
       "mesurer la fuite due à trois prétraitements ajustés avant la validation croisée, et reconnaître le plus "
       "dangereux.",
       "Ex 12.16 · ch. 8 (validation croisée, fuite de données) · fiche §12.10 et encadré 🕰️ « l'encodage des "
       "catégories »", thread="synthétique", tracks="R, C",
       body=r"""Des données **inventées** : 300 exemples, une cible binaire `Y_31` qui dépend de `x0` et `x1`, quatre colonnes numériques à 15 % de valeurs manquantes (`NUM_31`), et une colonne `code` à 150 modalités **sans aucun lien** avec la cible (un identifiant de magasin, par exemple). La préparation complète : imputation et standardisation des colonnes numériques (`numeric_31()`), encodage de `code` par la cible (`encoder_31()`, un `TargetEncoder`), puis `LogisticRegression()`. On la juge par l'AUC moyenne d'une validation croisée (`cv_auc_31`, avec les folds `CV_31`).

Écris `auc_31(variant)`, qui rend cette AUC pour quatre variantes :
- `"A"` : toute la préparation **dans** la validation croisée (un pipeline : `make_column_transformer((numeric_31(), NUM_31), (encoder_31(), ["code"]))`, puis la régression) ;
- `"B"` : l'imputation et la standardisation ajustées sur **toutes** les lignes avant la validation croisée (`numeric_31().fit_transform`), l'encodeur restant dans le pipeline (`"passthrough"` pour les colonnes déjà préparées) ;
- `"C"` : l'encodeur ajusté sur toutes les lignes avant la validation croisée, par `encoder_31().fit(...)` puis `.transform(...)`, l'imputation et la standardisation restant dans le pipeline ;
- `"D"` : comme C, mais avec `encoder_31().fit_transform(...)`.

La vérification contrôle a) la liste des quatre AUC, dans l'ordre A, B, C, D, puis :
b) `largest_31` : la lettre de la variante la plus optimiste ;
c) `scaler_small_31` : l'écart entre B et A est-il inférieur à 0,01 en valeur absolue ? (`True` ou `False`) ;
d) `crossfit_small_31` : l'écart entre D et A est-il inférieur à 0,01 en valeur absolue ?

Dans tes notes : pourquoi la fuite de B est-elle si petite, et celle de C si grande, alors que la colonne `code` n'apporte **rien** ? Pourquoi D fuit-il beaucoup moins que C (relis l'encadré 🕰️ du §12.3.1) ? Une fuite minuscule, comme celle de B, est-elle pour autant acceptable ?""",
       given=GIVEN_31, todo=TODO_31, check=CHECK_31, solution=SOLUTION_31 + CHECK_31.replace('with wb.attempt("12.31"):\n', "if True:\n").replace("wb.check(", "print_answer("),
       record=r'''wb.record("12.31a", [aucs_ref_31[variant] for variant in "ABCD"], decimals=4)
wb.record("12.31b", largest_31, mistakes={"compare les quatre AUC : la plus grande n'est pas celle de la préparation honnête": "A",
                                          "compare l'écart de chaque variante à A : laquelle s'en éloigne le plus ?": "B",
                                          "regarde de combien chaque variante s'écarte de A : laquelle est la plus loin ?": "D"})
wb.record("12.31c", scaler_small_31, mistakes={"calcule |AUC(B) − AUC(A)| et compare-le à 0,01": False})
wb.record("12.31d", crossfit_small_31, mistakes={"calcule |AUC(D) − AUC(A)| et compare-le à 0,01": False})''',
       note="L'AUC honnête (A) vaut environ 0,74. Ajuster l'imputation et la standardisation sur toutes les lignes (B) "
            "ne change presque rien (+0,0004) : la moyenne et l'écart-type de 300 lignes ressemblent à ceux de 240, "
            "et ils ne disent rien de la cible. L'encodeur par la cible ajusté sur toutes les lignes, puis appliqué "
            "(C), fait grimper l'AUC vers 0,90 : avec un ou deux exemples par code, la moyenne de la cible dans un "
            "code **est** presque la cible de la ligne elle-même. La colonne devient un recopiage déguisé du label, y "
            "compris pour les lignes de validation, et le modèle l'exploite. `fit_transform` (D) encode chaque ligne "
            "avec des moyennes calculées sur les **autres** folds de son cross fitting interne : la ligne ne voit pas "
            "sa propre cible, et la fuite tombe à +0,006 (les lignes de validation ont tout de même servi à encoder "
            "celles d'entraînement). Une petite fuite reste une fuite : sur d'autres données (peu de lignes, une "
            "transformation plus souple), la même erreur peut coûter cher, et un `Pipeline` l'empêche par "
            "construction."),

    Ex("12.32", "🛠️", 3, 30, "Une fonction de préparation documentée et son test anti-fuite",
       "écrire une préparation réutilisable, avec sa docstring, et des tests qui attrapent les fuites et les "
       "décalages de colonnes.",
       "Ex 12.26 · Ex 12.18 · ch. 8 (fuites) · 0A (pytest) · fiche §12.2", thread="Penguins", tracks="C",
       body=r"""Une préparation qu'on recopie d'un notebook à l'autre finit toujours par fuir. On l'écrit une fois, dans une fonction documentée et testée. Écris :
- `prepare_32(train, other)` : `train` et `other` sont deux DataFrames aux colonnes de `penguins.csv` ; la fonction rend `(X_train, X_other)`, deux tableaux de nombres aux mêmes colonnes : les quatre mesures, imputées par leur **médiane** puis standardisées, puis le one-hot de l'île et du sexe (un sexe manquant devient la catégorie `"unknown"` ; une catégorie inconnue de `train` donne un bloc de zéros). Tout ce qui est appris vient de `train` ; aucun des deux DataFrames n'est modifié. Utilise les classes de scikit-learn `SimpleImputer`, `StandardScaler` et `OneHotEncoder` (sous ces noms) et donne à la fonction une docstring qui dit ce qu'elle fait, ses paramètres et ce qu'elle rend ;
- au moins **quatre** tests `test_...`, rangés dans `TESTS_32`, qui appellent `prepare_32` sur de petits DataFrames que tu construis toi-même. Le fichier de test ne contient que `math`, `numpy` (`np`), `pandas` (`pd`), `pytest`, les trois classes ci-dessus, les listes `NUMERIC_32` et `CATEGORICAL_32`, ta fonction, tes tests et tes fonctions dont le nom commence par `small_32` (des aides pour construire les petits DataFrames).

La vérification lance tes tests sur ta fonction (ils doivent passer), puis sur cinq versions boguées, `prepare_bug_1` à `prepare_bug_5` (dans la cellule suivante) : chacune doit faire échouer au moins un de tes tests. Elles fuient, réapprennent sur `other`, décalent des colonnes, se trompent de statistique ou modifient leurs arguments.

Dans tes notes : pour chaque version boguée, le test qui l'attrape et la propriété qu'il vérifie. Pourquoi un test « le score est bon » n'attraperait-il presque aucune d'elles ?""",
       given=GIVEN_32, todo=TODO_32, check=CHECK_32, solution=SOLUTION_32 + CHECK_32.replace('with wb.attempt("12.32"):\n', "if True:\n"),
       note="Les tests utiles vérifient des **propriétés** de la préparation, sur de petits exemples dont on connaît "
            "la réponse : `X_train` ne change pas quand `other` change (attrape `prepare_bug_1`, qui ajuste "
            "l'imputation et la standardisation sur les deux) ; une ligne de `other` se prépare de la même façon seule ou dans un batch "
            "(attrape `prepare_bug_2`, qui réapprend sur `other`) ; un `other` d'une seule ligne a les colonnes de "
            "`train` (attrape `prepare_bug_3` et ses `get_dummies`) ; une valeur manquante reçoit la médiane de "
            "`train`, sur un exemple où la moyenne diffère (attrape `prepare_bug_4`) ; les DataFrames reçus sont "
            "intacts après l'appel (attrape `prepare_bug_5`). Un test « le score est bon » passerait presque "
            "toujours : sur des données qui ressemblent à l'entraînement, une fuite ou un réapprentissage change à "
            "peine le score (12.31, B) ; c'est en production, sur un batch différent, qu'elle coûte."),

    Ex("12.33", "🏆", 4, 100, "Défi : Penguins brut prêt pour l'entraînement, sans fuite",
       "préparer un fichier brut pour un modèle, en respectant la règle d'or, et gagner en accuracy par la seule "
       "préparation.",
       "Ex 12.32 · Ex 12.11 · ch. 8 (validation croisée, règle du test unique) · fiche §12.2 à §12.5",
       thread="Penguins", tracks="C",
       body=r"""Le modèle est **imposé** : `LogisticRegression(max_iter=1000)`, qui prédit le **sexe** des manchots. Tu ne joues que sur la préparation : une fonction `prepare_33(raw_train, raw_new)` qui reçoit deux morceaux de `DATA_33` (toutes les colonnes pour `raw_train` ; les mêmes, **sans** `Sex`, pour `raw_new`) et rend `(X_train, X_new)`, deux tableaux de nombres. `DATA_33` est `RAW` sans `Sample Number`, `Individual ID`, `Comments` et `Date Egg`, lignes mélangées : des numéros de saisie, des remarques de terrain et la date de ponte, commune aux deux adultes d'un nid, pas des mesures ; dans ce fichier, le numéro d'un adulte dans son nid (A1 ou A2) suit le plus souvent le sexe, et l'ordre des lignes aussi, une régularité de la collecte qu'un modèle ne doit pas exploiter.

Les règles, vérifiées à chaque notation :
1. une ligne rendue par ligne reçue, sans en retirer aucune (des manchots sans mesure compris), et aucune valeur manquante ni infinie ;
2. tout ce qui s'apprend (médianes, moyennes, catégories…) s'apprend sur `raw_train` seul : `X_train` ne change pas quand `raw_new` change ;
3. chaque ligne de `raw_new` se prépare seule : son résultat ne dépend pas des autres lignes du batch ;
4. les DataFrames reçus ne sont pas modifiés.

La note : l'accuracy moyenne en validation croisée à 5 folds sur `DATA_33`, les manchots de sexe inconnu étant préparés mais ni appris ni notés. Le score **public** (3 répétitions) se recalcule autant que tu veux ; le score **caché** (10 autres répétitions) n'est révélé qu'**une fois** par session, quand tu mets `READY_33 = True`.

**Objectif : au moins 0,92 sur le score caché.** 🌟 Palier : au moins 0,93. Le point de départ (les quatre mesures, imputées et standardisées) fait environ 0,894. Pistes : la fiche (§12.3 à §12.5), 12.11, et une question : un manchot lourd est-il un mâle ?

Dans tes notes : ta préparation, ses scores public et caché, ce qui a compté, et ce que tu as essayé sans succès. Pourquoi le modèle reçoit-il l'espèce, alors qu'on ne prédit que le sexe ?""",
       given=GIVEN_33, todo=TODO_33, check=CHECK_33, solution=SOLUTION_33 + CHECK_33.replace('with wb.attempt("12.33"):\n', "if True:\n"),
       note="Le corrigé standardise les six mesures (les quatre mesures et les deux isotopes) **à l'intérieur de "
            "chaque espèce**, avec des médianes, des moyennes et des écarts-types appris sur `raw_train`, et ajoute "
            "l'espèce en one-hot : score public 0,927, caché 0,925 (objectif atteint ; le palier 🌟 demande un peu "
            "plus). Un mâle est lourd **pour son espèce** : un Gentoo femelle pèse plus qu'un Adélie mâle, si "
            "bien que la masse brute mélange l'espèce et le sexe ; centrée dans l'espèce, elle ne parle plus que du "
            "sexe. Les isotopes aident un peu (les sexes ne s'alimentent pas tout à fait de la même façon). Pour le "
            "palier, des interactions entre l'espèce et les mesures standardisées (un modèle linéaire par espèce, en "
            "somme), ajoutées aux colonnes du corrigé, atteignent 0,933 en public et 0,932 sur le score caché. Les règles 2 et 3 sont la règle d'or : une préparation qui apprendrait sur `raw_new` "
            "ou d'un batch à l'autre ne serait pas celle qui sert en production."),
])
PARTS = [part for part in (globals().get("PART_A"), globals().get("PART_B"), globals().get("PART_C"),
                           globals().get("PART_D")) if part is not None]


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
    if kind == "exercise":
        title = "# 12 · Préparation des données — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est "
               "pas fait affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la "
               "section « Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch12_preparation/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 12`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
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
               "pas les arbres de décision.")]


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
