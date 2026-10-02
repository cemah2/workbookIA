#!/usr/bin/env python
"""Build the two notebooks of chapter 9 from a single source (used by Claude).

    python tools/chapters/build_ch09.py
    python tools/run_all_notebooks.py chapitres/ch09_overfitting/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch09_overfitting/03_notebook.ipynb

Part 0 checks the short answers of the quizzes (all but Q11), of the recalls R2 and R3, of the
✏️ paper exercises 9.1, 9.4, 9.5 and 9.7, of ∂ 9.2, 9.3 and 9.6 and of the 📈 reading 9.9.
Parts A to D (exercises 9.12 to 9.31) come with the second generation session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Paper, badge, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "9"
FOLDER = "chapitres/ch09_overfitting"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np

# R2: probabilities given to the right class of four examples
P_R2 = np.array([0.9, 0.6, 0.25, 0.8])
CE_R2 = float(-np.mean(np.log(P_R2)))                          # nats

# R3: four measurements
S_R3 = np.array([3.0, 5.0, 6.0, 10.0])

# 9.1: five targets and five predictions
Y_91 = np.array([3.0, 5.0, 4.0, 8.0, 10.0])
YHAT_91 = np.array([2.0, 5.5, 5.0, 7.5, 9.0])
R_91 = Y_91 - YHAT_91
SST_91 = float(np.sum((Y_91 - Y_91.mean()) ** 2))
Y2_91 = Y_91.copy()
Y2_91[4] = 20.0                                                # the 5th target becomes 20
R2_91 = Y2_91 - YHAT_91

# 9.2: five points and their least squares line
X_92 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
Y_92 = np.array([1.0, 3.0, 2.0, 5.0, 7.0])
A_92 = float(np.sum((X_92 - X_92.mean()) * (Y_92 - Y_92.mean())) / np.sum((X_92 - X_92.mean()) ** 2))
B_92 = float(Y_92.mean() - A_92 * X_92.mean())
RES_92 = Y_92 - (A_92 * X_92 + B_92)

# 9.3 and 9.6: one feature, no intercept
X_93 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
Y_93 = np.array([-3.0, -1.0, 0.0, 2.0, 2.0])
SXY_93, SXX_93 = float(X_93 @ Y_93), float(X_93 @ X_93)


def ridge_1d(lam):
    """The weight of the 1-D ridge regression without intercept (∂ 9.3)."""
    return SXY_93 / (SXX_93 + lam)


def soft_threshold(z, gamma):
    """S(z, gamma) = sign(z) max(|z| - gamma, 0)."""
    return math.copysign(max(abs(z) - gamma, 0.0), z) + 0.0


N_96 = len(X_93)
RHO_96, Z_96 = SXY_93 / N_96, SXX_93 / N_96

# 9.4: four models at three points, and the ideal curve
P_94 = np.array([[1.5, 2.0, 2.0], [0.5, 3.0, 2.5], [1.0, 2.5, 1.5], [1.0, 2.5, 2.0]])
F_94 = np.array([1.0, 2.0, 3.0])
AVG_94 = P_94.mean(axis=0)
BIAS2_94 = float(np.mean((AVG_94 - F_94) ** 2))
VAR_94 = float(np.mean(P_94.var(axis=0)))                       # ddof=0: divide by the 4 models

# 9.5: the validation losses of 15 epochs
VAL_95 = [0.90, 0.72, 0.61, 0.55, 0.53, 0.54, 0.52, 0.53, 0.55, 0.54, 0.56, 0.57, 0.58, 0.60, 0.61]


def early_stopping(losses, patience, min_delta=0.0):
    """(epoch of the stop, epoch whose weights are kept), epochs numbered from 1 (the rule of the fiche)."""
    best, kept, wait = math.inf, None, 0
    for epoch, loss in enumerate(losses, start=1):
        if loss < best - min_delta - 1e-12:                    # a strict improvement (1e-12: float rounding)
            best, kept, wait = loss, epoch, 0
        else:
            wait += 1
            if wait == patience:
                return epoch, kept
    return len(losses), kept


# 9.7: nine lines (a, b), a prior with weights 4, 2 and 1, a coarse likelihood of the gap r
LINES_97 = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
PRIOR_97 = {line: {0: 4, 1: 2, 2: 1}[abs(line[0]) + abs(line[1])] / 16 for line in LINES_97}


def likelihood_97(line, point):
    gap = abs(point[1] - (line[0] * point[0] + line[1]))
    return {0: 1.0, 1: 0.5, 2: 0.1}.get(gap, 0.0)


def normalised(weights):
    total = sum(weights.values())
    return {line: weight / total for line, weight in weights.items()}


POST1_97 = normalised({line: PRIOR_97[line] * likelihood_97(line, (1, 1)) for line in LINES_97})
POST2_97 = normalised({line: POST1_97[line] * likelihood_97(line, (-1, 0)) for line in LINES_97})'''

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("9.Q1", "Sur- ou sous-apprentissage ? Définitions et symptômes", [
        ("a", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"18 % sur la validation contre 0,5 % sur l'entraînement : regarde l'écart": "A",
          "l'erreur d'entraînement est très basse : le modèle réussit ses propres exemples": "C"}'''),
        ("b", 'the letter of your choice', '"C"',
         r'''mistakes={"les deux erreurs sont proches, mais compare leur niveau à celui d'un humain": "A",
          "l'écart est petit : ce n'est pas le symptôme de l'overfitting": "B"}'''),
        ("c", 'the letter of your choice', '"A"',
         r'''mistakes={"un écart d'un point n'est pas le symptôme de l'overfitting": "B",
          "3 % contre 2 % pour un humain, et un seul point d'écart avec la validation : est-ce le portrait d'un modèle qui n'a pas appris ?": "C"}'''),
        ("d", "True or False", "False",
         r'''mistakes={"cherche, parmi les trois modèles, un contre-exemple à cette phrase": True}'''),
    ]),
    Paper("9.Q2", "Walter et sa moustache : qu'est-ce qui a été mal appris ?", [
        ("a", 'the letter of your choice', '"C"',
         r'''mistakes={"les prénoms ont bien été retenus : on les a même ressortis": "A",
          "c'est le lien entre l'apparence et le prénom qui a trompé, pas la mémoire de l'apparence": "B",
          "l'histoire montre une règle qui trompe, pas un oubli": "D"}'''),
        ("b", 'the letter of your choice', '"B"',
         r'''mistakes={"pendant le mariage, chaque invité déjà rencontré était reconnu": "A"}'''),
        ("c", "True or False", "True",
         r'''mistakes={"relis le §9.5 du livre : ce qu'on aurait pu remarquer d'autre chez Walter": False}'''),
        ("d", 'the letter of your choice', '"D"',
         r'''mistakes={"le learning rate règle la taille des pas, pas ce que le modèle remarque": "A",
          "un test plus grand ne changerait pas ce qui a été appris": "B",
          "le problème vient de ce qui permet de reconnaître les exemples, pas de la loss": "C"}'''),
    ]),
    Paper("9.Q3", "Sous-apprentissage : les vrais remèdes", [
        ("a", 'the letter of your choice', '"D"',
         r'''mistakes={"une fuite donnerait un score trop beau, pas un score bas": "A",
          "l'écart entre entraînement et validation est minuscule": "B",
          "un modèle plus souple fait bien mieux sur la même validation": "C"}'''),
        ("b", 'the letters of the useful remedies, in alphabetical order, e.g. "BD"', '"ACE"',
         r'''mistakes={"plus d'exemples du même genre rend-il une régression linéaire plus souple ? (fiche §9.2.2)": "ABCE",
          "augmenter la régularisation rend le modèle encore plus rigide": "ACDE",
          "une pénalité plus faible donne-t-elle plus ou moins de liberté au modèle ?": "AE",
          "relis le diagnostic de a) : le problème est-il la variance ?": "B",
          "un modèle plus souple atteint 0,80 sur la même validation : ce remède-là ne serait-il pas utile ?": "AC",
          "ajouter des puissances et des produits de features, c'est aussi rendre le modèle plus souple": "CE"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"les deux R² sont déjà presque égaux : sur quoi plus d'exemples agiraient-ils ?": True}'''),
    ]),
    Paper("9.Q4", "Courbes d'erreur : où commence le surapprentissage ?", [
        ("a", "a whole number (an epoch of the table)", "20",
         r'''mistakes={"c'est l'epoch de la plus basse erreur d'entraînement : la question porte sur la validation": 40,
          "lis toute la ligne de validation : la question demande sa plus petite valeur, pas le moment où elle se met à remonter": 25}'''),
        ("b", "True or False", "True",
         r'''mistakes={"regarde la ligne « entraînement » au-delà de cette epoch": False}'''),
        ("c", 'the letter of your choice', '"A"',
         r'''mistakes={"à l'epoch 35, l'erreur d'entraînement est basse et l'écart avec la validation est grand": "B",
          "la validation suffit pour ce diagnostic ; le test est gardé pour l'évaluation finale": "C"}'''),
        ("d", "True or False", "False",
         r'''mistakes={"l'erreur de validation est mesurée sur un échantillon de données nouvelles : relis l'encadré ⚠️ de la fiche §9.3": True}'''),
    ]),
    Paper("9.Q5", "Un point isolé : frontière tordue ou frontière simple ?", [
        ("a", 'the letter of your choice', '"D"',
         r'''mistakes={"un centroïde est le centre d'un groupe de points": "A",
          "une fuite fait passer une information du test vers le modèle": "B",
          "ce point est remarquable par sa position au milieu de l'autre classe (les vecteurs de support viendront avec les SVM, ch. 13)": "C"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"compte les points d'entraînement mal classés par chacune des deux frontières": False}'''),
        ("c", 'the letter of your choice', '"A"',
         r'''mistakes={"regarde de quel côté de la frontière simple tombe tout ce voisinage": "B"}'''),
        ("d", "True or False", "False",
         r'''mistakes={"un point aberrant peut être une vraie valeur, rare : il faut d'abord savoir d'où il vient": True}'''),
    ]),
    Paper("9.Q6", "Early stopping : quand s'arrêter, et pourquoi c'est délicat", [
        ("a", 'the letter of your choice', '"D"',
         r'''mistakes={"décider sur le test, c'est déjà s'en servir pour régler le modèle (ch. 8)": "A",
          "l'erreur d'entraînement baisse presque toujours : elle ne signale pas l'overfitting": "B",
          "le learning rate est un réglage, pas une mesure de la généralisation": "C"}'''),
        ("b", 'the letter of your choice', '"B"',
         r'''mistakes={"l'erreur d'entraînement continue en général de baisser": "A",
          "le test n'est pas consulté pendant l'entraînement": "C",
          "28 epochs est la valeur d'une figure du livre, pas une règle": "D"}'''),
        ("c", "a whole number", "5",
         r'''fractional="un nombre d'epochs est entier",
          mistakes={"compte les epochs sans amélioration nécessaires pour que l'attente atteigne la patience": 4,
          "l'epoch de la meilleure loss ne compte pas dans l'attente": 6}'''),
        ("d", 'the letter of your choice', '"A"',
         r'''mistakes={"la dernière epoch suit plusieurs epochs sans amélioration : ses poids ne sont pas les meilleurs": "B",
          "moyenner des poids est une autre technique, qui n'est pas l'early stopping": "C",
          "la première epoch n'a presque rien appris": "D"}'''),
    ]),
    Paper("9.Q7", "Régularisation : ce que change λ", [
        ("a", 'the letter of your choice', '"C"',
         r'''mistakes={"la pénalité grandit avec la taille des poids : que fait la minimisation quand elle pèse plus lourd ?": "A",
          "la pénalité dépend des poids : elle les change": "B",
          "la pénalité ne rend pas les poids égaux": "D"}'''),
        ("b", 'the letter of your choice', '"A"',
         r'''mistakes={"la loss sur les données est la plus basse sans pénalité : que se passe-t-il quand la solution s'en éloigne ?": "B"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"si l'on minimisait la loss d'entraînement en λ, quelle valeur obtiendrait-on toujours ?": True}'''),
        ("d", 'the letter of your choice', '"A"',
         r'''mistakes={"C joue le rôle de l'inverse de λ (encadré 💼 de la fiche §9.5)": "B"}'''),
        ("e", "True or False", "True",
         r'''mistakes={"avec des poids écrasés à 0, que reste-t-il de la prédiction, l'ordonnée à l'origine n'étant pas pénalisée ?": False}'''),
    ]),
    Paper("9.Q8", "Pénalité sur les poids, dropout, batchnorm : même objectif ?", [
        ("a", 'the letters, in alphabetical order, e.g. "BD"', '"ABCE"',
         r'''mistakes={"l'early stopping est aussi une régularisation : il empêche d'apprendre le détail": "ABE",
          "l'augmentation de données vise aussi à mieux généraliser": "ABC",
          "un learning rate plus grand sert d'abord à aller plus vite": "ABCDE",
          "le dropout est une régularisation propre aux réseaux": "ACE",
          "une pénalité L2 sur les poids est le premier exemple de régularisation de la fiche §9.5": "BCE"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"relis l'encadré ⚠️ de la fiche §9.5 sur l'intention des auteurs de la batchnorm": False}'''),
        ("c", 'the letter of your choice', '"C"',
         r'''mistakes={"le dropout agit sur les neurones, pas sur les exemples": "A",
          "le dropout est temporaire : à chaque pas, d'autres neurones sont éteints": "B",
          "aucun neurone n'est supprimé pour de bon": "D"}'''),
        ("d", "True or False", "False",
         r'''mistakes={"compare les deux panneaux de la figure l1_l2.png : laquelle des deux pénalités met des poids exactement à zéro ?": True}'''),
    ]),
    Paper("9.Q9", "Biais et variance : des propriétés d'une famille de courbes", [
        ("a", "True or False", "False",
         r'''mistakes={"il faut une moyenne et une dispersion : sur combien de courbes les calculer ?": True}'''),
        ("b", 'the letter of your choice', '"A"',
         r'''mistakes={"comparer aux points bruités, c'est mesurer l'erreur d'entraînement": "B",
          "le test sert à l'évaluation, pas à définir le biais": "C",
          "la courbe moyenne est la référence de la variance": "D"}'''),
        ("c", 'the letter of your choice', '"D"',
         r'''mistakes={"c'est la définition du biais": "A",
          "c'est le troisième terme de la décomposition": "B",
          "l'erreur d'entraînement ne mesure pas la dispersion entre modèles": "C"}'''),
        ("d", 'the letter of your choice', '"B"',
         r'''mistakes={"même un modèle parfait se trompe sur des mesures bruitées": "A",
          "le biais est nul par hypothèse": "C",
          "l'erreur dépend de la taille du bruit, pas d'une constante": "D"}'''),
    ]),
    Paper("9.Q10", "Courbes raides ou souples : qui a quel biais, quelle variance ?", [
        ("a", 'one letter, "A" or "B"', '"A"',
         r'''mistakes={"un polynôme de degré 15 peut suivre une courbe ondulée, en moyenne sur les jeux": "B"}'''),
        ("b", 'one letter, "A" or "B"', '"B"',
         r'''mistakes={"des droites ajustées à des jeux de 30 points se ressemblent beaucoup": "A"}'''),
        ("c", "True or False", "True",
         r'''mistakes={"où les points qui retiennent le polynôme sont-ils les plus rares ?": False}'''),
        ("d", "True or False", "True",
         r'''mistakes={"avec plus de points par jeu, chaque polynôme dépend-il autant des points tirés ?": False}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("9.R2", "Ch. 6 : ce que mesure une cross-entropy utilisée comme loss", [
        ("a", "a number (3 decimals)", "CE_R2",
         r'''decimals=3, mistakes={"utilise le logarithme népérien (ln), pas log10": float(-np.mean(np.log10(P_R2))),
          "la cross-entropy est l'opposé du logarithme : elle est positive": -CE_R2,
          "c'est la valeur en bits : la question demande des nats": CE_R2 / math.log(2)}'''),
        ("b", "a number (3 decimals)", "CE_R2 / math.log(2)",
         r'''decimals=3, mistakes={"pour passer des nats aux bits, on divise par ln 2": CE_R2 * math.log(2),
          "c'est la valeur en nats": CE_R2}'''),
        ("c", "a whole number (1, 2, 3 or 4)", "3",
         r'''mistakes={"la plus grande loss vient de la plus petite probabilité donnée à la bonne classe": 1,
          "−ln 0,6 ≈ 0,51 : compare ce terme aux trois autres termes de la somme": 2,
          "−ln 0,8 ≈ 0,22 : compare ce terme aux trois autres termes de la somme": 4}'''),
        ("d", "True or False", "True",
         r'''mistakes={"une probabilité de 0,9 pour la bonne classe donne-t-elle une loss nulle ?": False}'''),
    ]),
    Paper("9.R3", "Ch. 2 : biais et variance d'un estimateur, et le bootstrap", [
        ("a", "a number", "float(S_R3.var())",
         r'''decimals=3, mistakes={"c'est la division par n − 1 : la question demande la division par n": float(S_R3.var(ddof=1)),
          "c'est la somme des carrés des écarts : divise-la": 26.0}'''),
        ("b", "a number (3 decimals)", "float(S_R3.var(ddof=1))",
         r'''decimals=3, mistakes={"c'est la division par n : la question demande n − 1": float(S_R3.var())}'''),
        ("c", "a negative number", "-0.25",
         r'''decimals=2, mistakes={"c'est le rapport (n − 1)/n : le biais est la différence avec 1": 0.75,
          "en moyenne, l'estimateur sous-estime la variance : quel est donc le signe du biais ?": 0.25,
          "le biais est E[estimateur] − σ², exprimé en fraction de σ²": -0.75}'''),
        ("d", "True or False", "False",
         r'''mistakes={"rééchantillonner un échantillon biaisé ne crée pas les personnes qui y manquent": True}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("9.1", "MSE et R² à la main sur cinq points", [
        ("a", "a number", "float(np.mean(R_91 ** 2))",
         r'''decimals=3, mistakes={"c'est la somme des carrés : la MSE en est la moyenne": float(np.sum(R_91 ** 2)),
          "on divise par n = 5, pas par n − 1": float(np.sum(R_91 ** 2) / 4)}'''),
        ("b", "a number (3 decimals)", "math.sqrt(np.mean(R_91 ** 2))",
         r'''decimals=3, mistakes={"c'est la MSE : la RMSE est sa racine": float(np.mean(R_91 ** 2))}'''),
        ("c", "a number", "float(np.mean(np.abs(R_91)))",
         r'''decimals=3, mistakes={"les résidus de signes opposés se compensent : prends leurs valeurs absolues": float(np.mean(R_91))}'''),
        ("d", "a number (3 decimals)", "1 - float(np.sum(R_91 ** 2)) / SST_91",
         r'''decimals=3, mistakes={"c'est le rapport SS_res / SS_tot : R² vaut 1 moins ce rapport": float(np.sum(R_91 ** 2)) / SST_91,
          "SS_tot se calcule autour de la moyenne des cibles": 1 - float(np.sum(R_91 ** 2)) / float(np.sum((Y_91 - YHAT_91.mean()) ** 2))}'''),
        ("e", "a number (3 decimals)", "1 - float(np.sum((Y_91 - 8) ** 2)) / SST_91",
         r'''decimals=3, mistakes={"un R² peut être négatif : ne l'arrondis pas à 0": 0.0,
          "ce modèle fait moins bien que la moyenne : quel est le signe de son R² ?": -(1 - float(np.sum((Y_91 - 8) ** 2)) / SST_91),
          "c'est le rapport SS_res / SS_tot : R² vaut 1 moins ce rapport": float(np.sum((Y_91 - 8) ** 2)) / SST_91}'''),
        ("f", "a number", "float(np.mean(R2_91 ** 2))",
         r'''decimals=2, mistakes={"c'est la somme des carrés : divise-la par n": float(np.sum(R2_91 ** 2))}'''),
        ("g", "a number", "float(np.mean(np.abs(R2_91)))",
         r'''decimals=2, mistakes={"c'est la somme des valeurs absolues : divise-la par n": float(np.sum(np.abs(R2_91)))}'''),
    ]),
    Paper("9.2", "Moindres carrés : la meilleure droite par dérivées partielles", [
        ("a", "a number", "A_92",
         r'''decimals=3, mistakes={"c'est la pente de la droite forcée de passer par l'origine : centre les données": float(X_92 @ Y_92 / (X_92 @ X_92))}'''),
        ("b", "a number", "B_92",
         r'''decimals=3, mistakes={"l'ordonnée à l'origine vaut ȳ − a·x̄": float(Y_92.mean() - A_92)}'''),
        ("c", "a number", "round(float(np.sum(RES_92)))",
         r'''fractional="la démonstration 3 donne cette somme sans calcul : vérifie-la avec la pente et l'ordonnée de a) et b), en gardant leurs valeurs exactes"'''),
        ("d", "a number", "float(np.sum(RES_92 ** 2))",
         r'''decimals=3, mistakes={"c'est la MSE : la question demande la somme des carrés": float(np.mean(RES_92 ** 2))}'''),
        ("e", "a number", "A_92 * 6 + B_92",
         r'''decimals=3, mistakes={"la prédiction est a·x + b": A_92 * 6}'''),
        ("f", "a whole number", "5 * 30 - 10 ** 2",
         r'''mistakes={"le déterminant de [[p, q], [q, r]] vaut p·r − q²": 5 * 30}'''),
    ]),
    Paper("9.3", "Ridge en dimension 1 : w* = Σxy / (Σx² + λ)", [
        ("a", "a number", "ridge_1d(0)", "decimals=3"),
        ("b", "a number", "ridge_1d(2.5)",
         r'''decimals=3, mistakes={"λ s'ajoute à Σx², au dénominateur, pas au poids": ridge_1d(0) + 2.5}'''),
        ("c", "a number (4 decimals)", "ridge_1d(5)",
         r'''decimals=4, mistakes={"λ s'ajoute à Σx², au dénominateur": SXY_93 / (SXX_93 * 5)}'''),
        ("d", "a number", "SXX_93",
         r'''decimals=3, mistakes={"c'est la valeur de Σxy": SXY_93,
          "c'est la valeur du dénominateur Σx² + λ : la question demande λ": 2 * SXX_93}'''),
        ("e", "a number (3 decimals)", "float(np.sum((Y_93 - ridge_1d(10) * X_93) ** 2))",
         r'''decimals=3, mistakes={"la question demande la somme des carrés des résidus seule, sans ajouter la pénalité": float(np.sum((Y_93 - ridge_1d(10) * X_93) ** 2)) + 10 * ridge_1d(10) ** 2,
          "c'est la somme des carrés sans pénalité (λ = 0)": float(np.sum((Y_93 - ridge_1d(0) * X_93) ** 2))}'''),
        ("f", "True or False", "True",
         r'''mistakes={"la somme des carrés est minimale au poids des moindres carrés : que se passe-t-il quand w* s'en éloigne ?": False}'''),
    ]),
    Paper("9.4", "Biais² et variance à partir d'un tableau de prédictions", [
        ("a", "a list of three numbers", "AVG_94.tolist()",
         r'''decimals=3, mistakes={"ce sont les valeurs de la courbe idéale f ; le modèle moyen se calcule à partir des quatre modèles": F_94.tolist()}'''),
        ("b", "a number (3 decimals)", "BIAS2_94",
         r'''decimals=3, mistakes={"élève chaque écart au carré avant de faire la moyenne": float(np.mean(AVG_94 - F_94) ** 2),
          "c'est l'erreur quadratique moyenne des modèles ; le biais² se calcule avec le modèle moyen": float(np.mean((P_94 - F_94) ** 2)),
          "le biais² est une moyenne de carrés : il ne peut pas être négatif": float(np.mean(AVG_94 - F_94))}'''),
        ("c", "a number (3 decimals)", "VAR_94",
         r'''decimals=3, mistakes={"la variance divise par le nombre de modèles (4), comme bias_variance_decomposition": float(np.mean(P_94.var(axis=0, ddof=1)))}'''),
        ("d", "a number (3 decimals)", "float(np.mean((P_94 - F_94) ** 2))",
         r'''decimals=3, mistakes={"c'est le biais² seul : la question porte sur les douze erreurs des modèles": BIAS2_94,
          "divise la somme par le nombre de couples (modèle, point)": float(np.sum((P_94 - F_94) ** 2) / 4)}'''),
        ("e", "a number (3 decimals)", "float(np.mean((P_94 - F_94) ** 2)) + 0.09",
         r'''decimals=3, mistakes={"sur des mesures bruitées, il faut ajouter le bruit": float(np.mean((P_94 - F_94) ** 2)),
          "le bruit s'ajoute par sa variance σ², pas par son écart-type": float(np.mean((P_94 - F_94) ** 2)) + 0.3}'''),
        ("f", "True or False", "True",
         r'''mistakes={"l'erreur du modèle moyen face à f, c'est l'un des termes déjà calculés : compare-le à d)": False}'''),
    ]),
    Paper("9.5", "Early stopping avec patience sur une courbe de loss", [
        ("a", "a whole number (an epoch)", "early_stopping(VAL_95, 2)[0]",
         r'''mistakes={"l'epoch de la meilleure loss ne compte pas dans l'attente": 8,
          "une loss de 0,52 est une amélioration sur 0,53 : l'attente repart de zéro": 7}'''),
        ("b", "a whole number (an epoch)", "early_stopping(VAL_95, 2)[1]",
         r'''mistakes={"on recharge les poids de la meilleure epoch, pas ceux de l'arrêt": 9,
          "une loss de 0,52 bat la meilleure loss précédente": 5}'''),
        ("c", "a whole number (an epoch)", "early_stopping(VAL_95, 1)[0]",
         r'''mistakes={"avec une patience de 1, une seule epoch sans amélioration suffit à arrêter": 9,
          "on s'arrête à la fin d'une epoch sans amélioration, pas à la fin de la meilleure": 5}'''),
        ("d", "a whole number (an epoch)", "early_stopping(VAL_95, 1)[1]",
         r'''mistakes={"on recharge les poids de la meilleure epoch, pas ceux de l'arrêt": 6,
          "cette epoch n'a pas été atteinte : l'entraînement s'est arrêté avant": 7}'''),
        ("e", "a whole number (an epoch)", "early_stopping(VAL_95, 4)[0]",
         r'''mistakes={"l'epoch de la meilleure loss ne compte pas dans l'attente": 10,
          "une loss de 0,52 est une amélioration sur 0,53 : l'attente repart de zéro": 9}'''),
        ("f", "a whole number (an epoch)", "early_stopping(VAL_95, 2, 0.02)[0]",
         r'''mistakes={"avec min_delta, une baisse ne compte que si elle dépasse 0,02 : refais le suivi": 9,
          "une baisse d'exactement 0,02 n'est pas une amélioration avec min_delta = 0,02 (comparaison stricte)": 7}'''),
        ("g", "a whole number (an epoch)", "early_stopping(VAL_95, 2, 0.02)[1]",
         r'''mistakes={"une baisse d'exactement 0,02 n'est pas une amélioration avec min_delta = 0,02 (comparaison stricte)": 5,
          "avec min_delta, une baisse ne compte que si elle dépasse 0,02 : refais le suivi": 7,
          "on recharge les poids de la dernière amélioration, pas ceux de l'epoch d'arrêt": 6}'''),
    ]),
    Paper("9.6", "Lasso en dimension 1 : le seuillage doux et les zéros exacts", [
        ("a", "a number", "soft_threshold(2.5, 1)",
         r'''decimals=3, mistakes={"le seuillage rapproche z de 0 : pour un z positif, on retire γ": 3.5}'''),
        ("b", "a number", "soft_threshold(-0.4, 0.5)",
         r'''decimals=3, mistakes={"si |z| ≤ γ, le seuillage doux donne exactement 0": -0.4 + 0.5,
          "c'est z sans seuillage : compare d'abord |z| au seuil γ": -0.4,
          "le seuillage rapproche z de 0, il ne l'en éloigne jamais": -0.4 - 0.5}'''),
        ("c", "a number", "soft_threshold(-3, 0.5)",
         r'''decimals=3, mistakes={"le seuillage rapproche z de 0 : pour un z négatif, on ajoute γ": -3.5,
          "le seuillage doux garde le signe de z": 2.5}'''),
        ("d", "a number", "RHO_96",
         r'''decimals=3, mistakes={"ρ divise Σxy par n": SXY_93,
          "c'est le poids des moindres carrés, Σxy / Σx²": SXY_93 / SXX_93}'''),
        ("e", "a number", "soft_threshold(RHO_96, 0.6) / Z_96",
         r'''decimals=3, mistakes={"le seuillage doux se divise ensuite par z = Σx² / n": soft_threshold(RHO_96, 0.6),
          "le seuil porte sur ρ, avant la division par z": RHO_96 / Z_96 - 0.6}'''),
        ("f", "a number", "soft_threshold(RHO_96, 1.3) / Z_96",
         r'''decimals=3, mistakes={"le seuillage doux se divise ensuite par z = Σx² / n": soft_threshold(RHO_96, 1.3)}'''),
        ("g", "a number", "abs(RHO_96)",
         r'''decimals=3, mistakes={"c'est le poids des moindres carrés : le seuil porte sur ρ": SXY_93 / SXX_93}'''),
        ("h", "True or False", "True",
         r'''mistakes={"relis la démonstration 2 de l'exercice 9.3 : le numérateur de w* change-t-il avec λ ?": False}'''),
    ]),
    Paper("9.7", "Mise à jour bayésienne d'une droite sur une grille 3 × 3", [
        ("a", "a number", "PRIOR_97[(0, 0)]",
         r'''decimals=3, mistakes={"c'est un poids : divise-le par la somme des neuf poids": 4}'''),
        ("b", "a number (3 decimals)", "POST1_97[(0, 1)]",
         r'''decimals=3, mistakes={"multiplie le prior par la vraisemblance, puis normalise par la somme des neuf produits": PRIOR_97[(0, 1)] * likelihood_97((0, 1), (1, 1)),
          "un écart de 3 a une vraisemblance nulle, pas 0,1 : refais la somme des neuf produits": 0.25}'''),
        ("c", "a whole number", "sum(1 for line in LINES_97 if POST1_97[line] == 0)",
         r'''mistakes={"une probabilité a posteriori est nulle quand la vraisemblance est nulle : regarde les écarts de 3 ou plus": 0}'''),
        ("d", "a number (3 decimals)", "POST2_97[(0, 0)]",
         r'''decimals=3, mistakes={"multiplie par la vraisemblance du second point, puis normalise par la somme des neuf nouveaux produits": POST1_97[(0, 0)] * likelihood_97((0, 0), (-1, 0))}'''),
        ("e", "a number (3 decimals)", "POST2_97[(1, 1)]",
         r'''decimals=3, mistakes={"c'est la probabilité après le premier point seulement": POST1_97[(1, 1)]}'''),
        ("f", "True or False", "True",
         r'''mistakes={"le posterior final est proportionnel au produit prior × L1 × L2 : l'ordre des facteurs compte-t-il ?": False}'''),
        ("g", "a list [a, b]", "list(max(LINES_97, key=POST2_97.get))",
         r'''decimals=1, mistakes={"c'est la droite exacte de h), mais elle n'est pas dans la grille : la réponse est l'une des neuf droites": [0.5, 0.5],
          "c'est l'une des droites les plus probables après P1 seulement : tiens compte aussi de P2": [0, 1],
          "après P1 seulement, cette droite était à égalité en tête : multiplie aussi par la vraisemblance de P2": [1, 0],
          "elle passe exactement par P2, mais pas par P1, et son prior est faible : compare les produits prior × L1 × L2": [1, 1]}'''),
    ]),
    Paper("9.9", "Diagnostiquer quatre paires de courbes d'entraînement et de validation", [
        ("a", "a panel number (1 to 4)", "4",
         r'''mistakes={"dans ce panneau, la loss de validation reste sous celle d'entraînement : ce n'est pas le symptôme cherché": 2,
          "dans ce panneau, la loss d'entraînement est très basse : le modèle réussit ses propres exemples": 1,
          "dans ce panneau, les deux losses descendent bas et restent proches : le modèle apprend sans rester bloqué": 3}'''),
        ("b", "a panel number (1 to 4)", "1",
         r'''mistakes={"dans ce panneau, les deux losses sont hautes et collées": 4,
          "dans ce panneau, les deux losses descendent bas et restent proches": 3,
          "dans ce panneau, la loss de validation reste sous celle d'entraînement : ce n'est pas le symptôme cherché": 2}'''),
        ("c", "one of 10, 30, 60 and 100", "30",
         r'''choices=[10, 30, 60, 100],
          mistakes={"c'est la fin de l'entraînement : compare la loss de validation à cet endroit avec celle des autres choix": 100,
          "lis la loss de validation à cette epoch et compare-la avec celle des autres choix": 60,
          "suis la courbe de validation (orange) sur toute sa longueur, et cherche son point le plus bas": 10}'''),
        ("d", 'the letters, in alphabetical order, e.g. "BD"', '"AC"',
         r'''mistakes={"un modèle qui sous-apprend a une validation au-dessus de l'entraînement, pas au-dessous": "ABC",
          "un learning rate trop petit ralentit les deux courbes sans changer leur ordre": "ACD",
          "une validation plus facile que l'entraînement (cas plus nets, labels plus sûrs) explique aussi ce panneau": "A",
          "une technique active seulement pendant l'entraînement explique aussi ce panneau": "C"}'''),
        ("e", "True or False", "False",
         r'''mistakes={"le panneau 4 montre de l'underfitting : plus de données le rendrait-il plus souple ?": True}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 9.1, 9.4, 9.5 et 9.7, ∂ 9.2, 9.3 et "
               "9.6, 📈 9.9)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "plusieurs choix : les lettres collées (`\"BD\"`) ; plusieurs nombres : une liste (`[4, 7]`). "
               "Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore "
               "remplies affichent ⏳. Les questions « dans ta copie », le quiz Q11, le rappel R1, la réflexion et "
               "l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 9.12 to 9.31): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "9.12 à 9.17", "sous- et surapprentissage, mesures d'erreur, features polynomiales, moindres carrés et "
                         "Ridge dans mylearn.linear"),
    ("B", "9.18 à 9.23", "courbes de validation, chemins de régularisation, early stopping, courbes "
                         "d'apprentissage, Ridge contre Lasso, Lasso par descente de coordonnées"),
    ("C", "9.24 à 9.27", "biais et variance mesurés par simulation, droites bayésiennes sur une grille"),
    ("D", "9.28 à 9.31", "bugs de régularisation, refactorisation testée, double descente, défi California"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 9.Q1 à 9.Q10, 9.R2, 9.R3, 9.1 à 9.7, 9.9 | vérifier tes réponses courtes | 🧠 🔁 ✏️ ∂ 📈 | ★ à ★★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |" for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# 9 · Surapprentissage et sous-apprentissage — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch09_overfitting/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 9`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 9 · Surapprentissage et sous-apprentissage — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Diagnostiquer l'underfitting et l'overfitting sur des courbes d'entraînement, de validation et "
               "d'apprentissage.\n"
               "- Écrire et vérifier contre scikit-learn les mesures d'erreur d'une régression, les features "
               "polynomiales, les moindres carrés, Ridge et Lasso.\n"
               "- Appliquer l'early stopping avec patience, et choisir la régularisation par validation croisée.\n"
               "- Mesurer le biais² et la variance d'une famille de modèles, et ajuster une droite à la manière "
               "bayésienne.\n\n"
               "**Rappel express.** $\\mathrm{MSE} = \\frac{1}{n}\\sum_i (y_i - \\hat{y}_i)^2$, "
               "$\\mathrm{MAE} = \\frac{1}{n}\\sum_i |y_i - \\hat{y}_i|$, $R^2 = 1 - SS_{\\text{res}}/SS_{\\text{tot}}$ ; "
               "moindres carrés : $a = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum (x_i - \\bar{x})^2}$, "
               "$b = \\bar{y} - a\\bar{x}$ ; Ridge : $(\\mathbf{X}_c^\\top\\mathbf{X}_c + \\alpha\\mathbf{I})\\mathbf{w} = "
               "\\mathbf{X}_c^\\top\\mathbf{y}_c$ ; Lasso : $w_j = S(\\rho_j, \\alpha)/z_j$ avec "
               "$S(z, \\gamma) = \\operatorname{signe}(z)\\max(|z| - \\gamma, 0)$ ; erreur attendue = biais² + "
               "variance + $\\sigma^2$. En Python : `np.linalg.lstsq`, `np.linalg.solve`, "
               "`sklearn.preprocessing.PolynomialFeatures`, `sklearn.linear_model.LinearRegression`, `Ridge`, "
               "`Lasso`, `sklearn.model_selection.learning_curve`, `validation_curve`.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 9.12 à 9.31 (parties A à D, ta librairie `mylearn.linear`) seront ajoutés "
             "à la prochaine session de génération ; relance alors `python tools/start_chapter.py 9` pour obtenir "
             "le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dire, à partir de deux erreurs et d'une référence, si un modèle sous-apprend ou "
               "sur-apprend, et quoi faire dans chaque cas ?\n"
               "2. Sais-tu dériver les moindres carrés et Ridge en dimension 1, et expliquer pourquoi le Lasso "
               "met des poids exactement à zéro ?\n"
               "3. Sais-tu calculer le biais² et la variance d'une famille de modèles, et dire pourquoi le "
               "compromis biais-variance n'est pas une loi ?\n\n"
               "**Pour aller plus loin** : le guide « Linear Models » de scikit-learn et le chapitre 6 d'*An "
               "Introduction to Statistical Learning*, cités dans la fiche. La suite : le ch. 10 (neurones), où "
               "un modèle linéaire suivi d'une activation devient un neurone." + later)]


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
