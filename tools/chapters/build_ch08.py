#!/usr/bin/env python
"""Build the two notebooks of chapter 8 from a single source (used by Claude).

    python tools/chapters/build_ch08.py
    python tools/run_all_notebooks.py chapitres/ch08_train_test/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch08_train_test/03_notebook.ipynb

Part 0 checks the short answers of the quizzes (all but Q8), of the recalls R1 and R2, of the
✏️ paper exercises 8.1 to 8.5, of ∂ 8.6 and of the 📈 reading 8.8. Parts A to D hold the code
exercises 8.11 to 8.27: splitting (A), choosing a hyperparameter on a validation set (B),
cross-validation in mylearn (C), leaks, honest comparisons and the challenge (D).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "8"
FOLDER = "chapitres/ch08_train_test"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np

# 8.1: the 344 penguins of the raw file
N_81, COUNTS_81 = 344, {"Adelie": 152, "Chinstrap": 68, "Gentoo": 124}
TEST_81 = math.ceil(0.2 * N_81)                                    # 60/20/20: the test first...
VAL_81 = math.ceil(0.25 * (N_81 - TEST_81))                        # ...then the validation in the rest


def largest_remainder(counts, n_part, n_total):
    """Number of samples of each class in a stratified part of n_part samples (largest remainder rule)."""
    exact = np.array(counts) * n_part / n_total
    base = np.floor(exact).astype(int)
    base[np.argsort(-(exact - base), kind="stable")[: n_part - base.sum()]] += 1
    return base.tolist()


# 8.4: standard error of an accuracy
P_84, N_84 = 0.92, 250
SE_84 = math.sqrt(P_84 * (1 - P_84) / N_84)

# 8.5: the scores of two models on the same 5 folds
A_85 = np.array([0.82, 0.88, 0.79, 0.85, 0.86])
B_85 = np.array([0.85, 0.84, 0.86, 0.85, 0.83])

# 8.6: the best of K independent validation scores
Q_86 = 0.0668'''

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("8.Q1", "La boucle d'entraînement : prédire, comparer, corriger", [
        ("a", 'the letter of your choice, e.g. "E"', '"B"',
         r'''mistakes={"c'est le comportement d'un modèle qui minimise une loss (question d) ; la boucle simplifiée du livre ne corrige que les erreurs": "A",
          "l'exemple reste dans le jeu d'entraînement : il resservira à l'epoch suivante": "C",
          "une epoch ne s'arrête qu'après le passage de tous les exemples": "D"}'''),
        ("b", 'the letter of your choice', '"C"',
         r'''mistakes={"le label seul ne dit pas dans quel sens corriger : il faut aussi savoir ce que le modèle a répondu": "A",
          "le jeu de test ne sert jamais à corriger le modèle (§8.3)": "B",
          "la correction porte sur l'exemple qu'on vient de traiter": "D"}'''),
        ("c", "True or False", "True",
         r'''mistakes={"pendant le test, l'erreur sert seulement à compter : rien ne remonte vers les paramètres": False}'''),
    ]),
    Paper("8.Q2", "Epoch, ordre des exemples et fréquence des mises à jour", [
        ("a", "a whole number", "1200 * 5",
         r'''mistakes={"une mise à jour par exemple, et chaque epoch présente tous les exemples : combien d'exemples en tout ?": 1200,
          "c'est le nombre de mises à jour en 5 epochs avec des mini-batches : ici, une mise à jour par exemple": 95}'''),
        ("b", "a whole number", "math.ceil(1200 / 64)",
         r'''fractional="un nombre de mises à jour est entier : le dernier mini-batch, plus petit, compte quand même pour une mise à jour",
          mistakes={"le dernier mini-batch, plus petit, compte aussi pour une mise à jour": 18}'''),
        ("c", "True or False", "True",
         r'''mistakes={"une epoch est un passage complet sur le jeu d'entraînement : chaque exemple y est vu une fois": False}'''),
        ("d", 'the letter of your choice', '"A"',
         r'''mistakes={"mélanger ne crée aucun exemple nouveau": "B",
          "le jeu de test n'intervient pas pendant l'entraînement": "C",
          "l'algorithme de mise à jour fonctionne quel que soit l'ordre ; c'est le modèle qui risque de s'adapter à cet ordre": "D"}'''),
    ]),
    Paper("8.Q3", "99 % sur l'entraînement : que peut-on vraiment conclure ?", [
        ("a", 'the letter of your choice', '"C"',
         r'''mistakes={"le score d'entraînement ne prédit pas le score sur des données nouvelles (§8.2.1)": "A",
          "« forcément » est trop fort : un raccourci est possible, pas certain": "B",
          "viser 100 % sur l'entraînement n'apprend rien sur la généralisation": "D"}'''),
        ("b", "True or False", "False",
         r'''mistakes={"relis la fin du §8.2.1 : il faut essayer le modèle pour le savoir": True}'''),
        ("c", 'the letter of your choice', '"A"',
         r'''mistakes={"les paramètres seuls ne disent pas comment le modèle se comportera sur des données nouvelles": "B",
          "le nombre de paramètres ne mesure pas la performance": "C",
          "un autre entraînement donnerait un autre modèle, pas une mesure du premier": "D"}'''),
    ]),
    Paper("8.Q4", "Le renard et la neige : repérer un raccourci appris", [
        ("a", 'the letter of your choice', '"B"',
         r'''mistakes={"le pelage distingue vraiment les deux animaux : ce n'est pas un raccourci": "A",
          "la forme des oreilles est une vraie différence entre renards et chats": "C",
          "la taille sur la photo dépend du cadrage, mais quel détail accompagne TOUJOURS une classe dans ces données ?": "D"}'''),
        ("b", "True or False", "False",
         r'''mistakes={"si le test vient du même lot, le décor y accompagne toujours la même classe : le raccourci y marche aussi": True}'''),
        ("c", 'the letter of your choice', '"C"',
         r'''mistakes={"un renard dans la neige est le cas où le raccourci et la vérité sont d'accord": "A",
          "un chat dans un salon est aussi un cas où le raccourci donne la bonne réponse": "B",
          "un renard en forêt ressemble à toutes les photos d'entraînement": "D"}'''),
    ]),
    Paper("8.Q5", "La règle d'or du jeu de test", [
        ("a", "True or False", "False",
         r'''mistakes={"choisir en regardant le test, c'est déjà s'en servir pour construire le modèle": True}'''),
        ("b", "True or False", "True",
         r'''mistakes={"relis le §8.3 : combien de fois le livre veut-il qu'on utilise le test ?": False}'''),
        ("c", "True or False", "False",
         r'''mistakes={"ces deux nombres transforment les données d'entraînement : ils font partie du modèle, au sens large": True}'''),
        ("d", "True or False", "True",
         r'''mistakes={"le test doit estimer la performance en déploiement : il doit donc y ressembler": False}'''),
    ]),
    Paper("8.Q6", "Fuite de données : ses formes courantes", [
        ("a", 'the letters of the leaky practices, in alphabetical order, e.g. "AC"', '"BDE"',
         r'''mistakes={"retirer les doublons avant de découper empêche au contraire une fuite": "ABDE",
          "ajuster la standardisation sur l'entraînement seulement, puis l'appliquer au test, est la bonne pratique": "BCDE",
          "une des pratiques restantes utilise une information qu'on n'aura pas au moment de prédire": "BD",
          "des radios du même patient des deux côtés du découpage, est-ce une fuite ?": "BE",
          "comment les 20 features ont-elles été choisies ?": "DE"}'''),
    ]),
    Paper("8.Q7", "Pourquoi un jeu de validation en plus du test ?", [
        ("a", 'the letter of your choice', '"D"',
         r'''mistakes={"le jeu de validation PREND des données à l'entraînement : il n'en ajoute pas": "A",
          "la validation ne remplace pas le test : les deux coexistent": "B",
          "relis le §8.4 : à quoi sert la boucle de la figure 8.9 ?": "C"}'''),
        ("b", "a whole number", "int(0.6 * 500)",
         r'''mistakes={"60 % + 20 % : c'est le réentraînement final, une fois la recherche terminée ; pendant la recherche, la validation ne fait que noter": 400,
          "c'est la taille de la validation (ou du test)": 100}'''),
        ("c", "True or False", "True",
         r'''mistakes={"chaque réglage apprend ses paramètres sur l'entraînement ; la validation ne sert qu'à le noter": False}'''),
    ]),
    Paper("8.Q9", "Validation croisée : ce qu'on moyenne, et pourquoi", [
        ("a", 'the letter of your choice', '"A"',
         r'''mistakes={"on moyenne des mesures de performance, pas des prédictions": "B",
          "les modèles des différents tours ne sont pas combinés entre eux": "C",
          "un score d'entraînement ne dit rien de la généralisation (§8.2.1)": "D"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"si le modèle gardait ce qu'il a appris au tour précédent, il aurait déjà vu le fold de validation": False}'''),
        ("c", 'the letter of your choice', '"B"',
         r'''mistakes={"le test est mis de côté AVANT la validation croisée (§8.5)": "A",
          "le test reste nécessaire, pour l'évaluation finale": "C",
          "à quoi servent les folds de validation temporaires ?": "D"}'''),
    ]),
    Paper("8.Q10", "k-fold : combien d'entraînements, quelle taille de fold ?", [
        ("a", "a whole number", "5",
         r'''mistakes={"un entraînement par fold de validation : combien de folds ?": 1}'''),
        ("b", "a whole number", "1003 // 5 + 1",
         r'''fractional="un fold contient un nombre entier d'exemples : écris 1003 = 5 × q + r",
          mistakes={"c'est la taille des derniers folds : les premiers en reçoivent un de plus": 200}'''),
        ("c", "a whole number", "1003 % 5",
         r'''mistakes={"c'est le nombre de folds de la petite taille": 2}'''),
        ("d", "a whole number", "1003 - 1003 // 5",
         r'''mistakes={"le fold 5 fait-il partie des grands folds ou des petits ?": 802,
          "il faut retirer le fold de validation": 1003}'''),
        ("e", "a whole number", "1003",
         r'''mistakes={"un exemple par fold : combien de folds ?": 1}'''),
    ]),
    Paper("8.Q11", "Les deux usages des résultats de test", [
        ("a", 'the two letters, in alphabetical order, e.g. "BD"', '"AE"',
         r'''mistakes={"l'un des deux usages a lieu PENDANT l'entraînement (§8.6)": "AC",
          "l'un des deux usages a lieu AVANT le déploiement (§8.6)": "CE"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"le nouveau réglage a été choisi après avoir vu le score de test : le test a servi à choisir": False}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("8.R1", "Ch. 7 : pourquoi l'inertie seule ne permet pas de choisir k", [
        ("a", "a number", "0",
         r'''mistakes={"chaque point est son propre centre : quelle est sa distance à ce centre ?": 200}'''),
        ("b", "True or False", "True",
         r'''mistakes={"avec un cluster de plus, on peut toujours garder les anciens centres et en ajouter un : l'inertie ne peut pas augmenter": False}'''),
        ("c", 'the letter of your choice', '"C"',
         r'''mistakes={"c'est justement le critère qui baisse mécaniquement": "A",
          "le nombre d'itérations ne juge pas la qualité des clusters": "B",
          "c'est un autre nom de l'inertie": "D"}'''),
    ]),
    Paper("8.R2", "Ch. 5 : un pas de descente de gradient à la main", [
        ("a", "a number (2 decimals at most)", "(0.5 * 2 - 3) ** 2",
         r'''decimals=2, mistakes={"c'est l'erreur w0·x − y : la loss est son carré": -2,
          "la loss est le carré de l'erreur, pas sa valeur absolue": 2}'''),
        ("b", "a number (2 decimals at most)", "2 * 2 * (0.5 * 2 - 3)",
         r'''decimals=2, mistakes={"regarde le signe de w0·x − y": 8,
          "n'oublie pas le facteur x de la dérivée de w·x": -4}'''),
        ("c", "a number (2 decimals at most)", "0.5 - 0.05 * (2 * 2 * (0.5 * 2 - 3))",
         r'''decimals=2, mistakes={"on retranche η × L'(w0), et L'(w0) est négatif : w doit augmenter": 0.1}'''),
        ("d", "a number (2 decimals at most)", "(0.9 * 2 - 3) ** 2",
         r'''decimals=2, mistakes={"c'est l'erreur w1·x − y : la loss est son carré": -1.2}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("8.1", "Découper 344 manchots : hold-out, validation et folds", [
        ("a", "a whole number", "math.ceil(0.25 * N_81)",
         r'''mistakes={"c'est la taille de l'entraînement : la question demande le test": 258}'''),
        ("b", "a whole number", "N_81 - math.ceil(0.25 * N_81)",
         r'''mistakes={"c'est la taille du test : la question demande l'entraînement": 86}'''),
        ("c", "a whole number", "TEST_81",
         r'''fractional="une taille est un nombre entier : applique l'arrondi de la règle",
          mistakes={"0,2 × 344 = 68,8 : la règle arrondit la taille du test vers le haut": 68}'''),
        ("d", "a whole number", "VAL_81",
         r'''fractional="une taille est un nombre entier : applique l'arrondi de la règle",
          mistakes={"la validation se prend dans les manchots qui restent, avec t = 0,25": 55,
                    "0,25 × 275 = 68,75 : la règle arrondit vers le haut": 68}'''),
        ("e", "a whole number", "N_81 - TEST_81 - VAL_81",
         r'''mistakes={"la validation compte ⌈0,25 × 275⌉ manchots": 207,
          "il faut encore retirer la validation": 275}'''),
        ("f", "a whole number", "(N_81 - TEST_81) // 5",
         r'''fractional="un fold contient un nombre entier de manchots",
          mistakes={"la validation croisée s'applique aux manchots qui restent une fois le test mis de côté": 69}'''),
        ("g", "a whole number", "N_81 % 10",
         r'''mistakes={"ce sont les folds de 34 : la question demande ceux de 35": 6}'''),
        ("h", "a whole number", "N_81 - (N_81 // 10 + 1)",
         r'''mistakes={"au premier tour, le fold 1 sert de validation : quelle est sa taille ?": 310,
          "il faut retirer le fold de validation": 344}'''),
        ("i", "a list of three whole numbers [Adelie, Chinstrap, Gentoo]",
         "largest_remainder([COUNTS_81[s] for s in ('Adelie', 'Chinstrap', 'Gentoo')], TEST_81, N_81)",
         r'''mistakes={"le total doit faire 69 : il manque des manchots, à donner aux plus grandes parties décimales": [30, 13, 24],
          "le total doit faire 69 : un manchot de trop": [31, 14, 25],
          "l'ordre demandé est Adélie, Chinstrap, Gentoo": [30, 25, 14]}'''),
    ]),
    Paper("8.2", "Compter les entraînements d'une recherche d'hyperparamètres", [
        ("a", "a whole number", "3 * 4 * 2",
         r'''mistakes={"chaque valeur de l'un se combine avec chaque valeur des autres : on multiplie, on n'additionne pas": 9}'''),
        ("b", "a whole number", "24",
         r'''mistakes={"un seul entraînement par réglage avec un jeu de validation fixe": 120}'''),
        ("c", "a whole number", "24 * 5",
         r'''mistakes={"chaque réglage est entraîné une fois par fold": 24,
          "il y a 5 tours, pas 4 : chaque fold sert une fois de validation": 96}'''),
        ("d", "a whole number", "24 * 5 + 1",
         r'''mistakes={"un seul réentraînement final, celui du réglage retenu": 144,
          "n'oublie pas le réentraînement final": 120}'''),
        ("e", "a number of hours (1 decimal)", "121 * 4 / 60",
         r'''decimals=1, mistakes={"121 entraînements de 4 minutes, puis la conversion en heures": 2.0,
          "n'oublie pas le réentraînement final de d) : 121 entraînements, pas 120": 8.0}'''),
        ("f", "a whole number", "5 * (24 * 5 + 1)",
         r'''mistakes={"dans chaque tour extérieur, le réglage retenu est aussi réentraîné": 600,
          "il y a 5 tours extérieurs": 121,
          "compte seulement le protocole décrit : la validation croisée imbriquée n'ajoute pas de réentraînement final sur toutes les données": 606}'''),
        ("g", "a whole number", "(200 - 1) // 5",
         r'''fractional="un nombre de réglages est entier : garde ceux qui tiennent dans le budget",
          mistakes={"n'oublie pas le réentraînement final, qui prend un entraînement du budget": 40}'''),
    ]),
    Paper("8.3", "Fuite ou pas ? Six protocoles à auditer", [
        ("a", "True or False", "True",
         r'''mistakes={"la médiane a été calculée sur toutes les annonces, test compris": False}'''),
        ("b", "True or False", "False",
         r'''mistakes={"la standardisation est ajustée sur les seules images d'entraînement, et le test est un fichier à part": True}'''),
        ("c", "True or False", "True",
         r'''mistakes={"le réglage a été choisi en regardant le test, puis son score publié": False}'''),
        ("d", "True or False", "True",
         r'''mistakes={"pour prévoir un mois, le modèle aura-t-il, en production, les mois qui le suivent ?": False}'''),
        ("e", "True or False", "False",
         r'''mistakes={"tous les enregistrements d'un locuteur sont dans le même fold : c'est le bon découpage ici": True}'''),
        ("f", "True or False", "True",
         r'''mistakes={"cette information sera-t-elle connue au moment où l'on doit prédire ?": False}'''),
    ]),
    Paper("8.4", "Quelle confiance accorder à une accuracy de test ? Erreur type et taille du test", [
        ("a", "a number (4 decimals)", "SE_84",
         r'''decimals=4, mistakes={"c'est la variance : l'erreur type est sa racine": P_84 * (1 - P_84) / N_84,
          "la racine porte sur tout le quotient p(1 − p) / n": math.sqrt(P_84 * (1 - P_84)) / N_84}'''),
        ("b", "a number (3 decimals)", "1.96 * SE_84",
         r'''decimals=3, mistakes={"c'est l'erreur type : la demi-largeur à 95 % la multiplie par 1,96": SE_84}'''),
        ("c", "a number (3 decimals)", "P_84 - 1.96 * SE_84",
         r'''decimals=3, mistakes={"c'est la borne haute": P_84 + 1.96 * SE_84,
          "retire la demi-largeur, pas l'erreur type": P_84 - SE_84}'''),
        ("d", "a whole number", "math.ceil((1.96 / 0.01) ** 2 * P_84 * (1 - P_84))",
         r'''fractional="un nombre d'exemples est entier : arrondis vers le haut, pour que la demi-largeur ne dépasse pas 0,01",
          mistakes={"arrondis vers le haut : avec un exemple de moins, la demi-largeur dépasserait 0,01": 2827,
                    "n'oublie pas le facteur 1,96 de la demi-largeur": 736}'''),
        ("e", "a whole number", "9",
         r'''mistakes={"n est sous une racine : diviser la demi-largeur par 3 demande plus que 3 fois plus d'exemples": 3}'''),
        ("f", "True or False", "False",
         r'''mistakes={"compare l'écart entre A et B à la demi-largeur trouvée en b)": True}'''),
    ]),
    Paper("8.5", "Moyenne et écart-type de scores de validation croisée", [
        ("a", "a number (3 decimals)", "A_85.mean()", "decimals=3"),
        ("b", "a number (3 decimals)", "A_85.std()",
         r'''decimals=3, mistakes={"la formule de la fiche divise par k = 5, pas par k − 1": A_85.std(ddof=1),
          "c'est la variance : l'écart-type est sa racine": A_85.var()}'''),
        ("c", "a number (3 decimals)", "B_85.mean()", "decimals=3"),
        ("d", "a number (3 decimals)", "B_85.std()",
         r'''decimals=3, mistakes={"la formule de la fiche divise par k = 5, pas par k − 1": B_85.std(ddof=1)}'''),
        ("e", "a whole number", "int(np.sum(B_85 > A_85))",
         r'''mistakes={"« strictement mieux » : une égalité ne compte pas": 3,
          "c'est le nombre de folds où A fait mieux que B... vérifie le sens de la comparaison": 1}'''),
        ("f", "a number (3 decimals)", "(B_85 - A_85).mean()",
         r'''decimals=3, mistakes={"vérifie le sens : B − A, pas A − B": (A_85 - B_85).mean()}'''),
        ("g", "a number (3 decimals)", "(B_85 - A_85).std()",
         r'''decimals=3, mistakes={"la formule de la fiche divise par k = 5, pas par k − 1": (B_85 - A_85).std(ddof=1)}'''),
        ("h", "True or False", "False",
         r'''mistakes={"compare la moyenne des différences à leur dispersion, et compte les folds gagnés par chacun": True}'''),
    ]),
    Paper("8.6", "Le biais d'optimisme du meilleur de K modèles", [
        ("a", "a number (2 decimals)", "math.sqrt(0.8 * 0.2 / 100)",
         r'''decimals=2, mistakes={"c'est la variance : l'erreur type est sa racine": 0.0016}'''),
        ("b", "a number (3 decimals)", "1 - (1 - Q_86) ** 10",
         r'''decimals=3, mistakes={"c'est la probabilité qu'AUCUN réglage n'atteigne 0,86 : prends le complément": (1 - Q_86) ** 10,
          "additionner les probabilités compte plusieurs fois les cas où plusieurs réglages dépassent 0,86": 10 * Q_86}'''),
        ("c", "a number (3 decimals)", "1 - (1 - Q_86) ** 50",
         r'''decimals=3, mistakes={"c'est la probabilité qu'AUCUN réglage n'atteigne 0,86 : prends le complément": (1 - Q_86) ** 50}'''),
        ("d", "a whole number", "math.ceil(math.log(0.1) / math.log(1 - Q_86))",
         r'''fractional="K est un nombre entier de réglages : arrondis vers le haut, pour atteindre AU MOINS 0,9",
          mistakes={"avec ce K, la probabilité reste juste sous 0,9 : arrondis vers le haut": 33}'''),
        ("e", "True or False", "True",
         r'''mistakes={"le réglage retenu a la même accuracy réelle que les autres : seul son score de validation a eu de la chance": False}'''),
    ]),
    Paper("8.8", "Comparer des modèles à partir de boîtes à moustaches de scores", [
        ("a", 'one letter, "A", "B" or "C"', '"C"',
         r'''mistakes={"compare les traits noirs, au milieu des boîtes": "B"}'''),
        ("b", 'one letter, "A", "B" or "C"', '"C"',
         r'''mistakes={"compare la hauteur des boîtes et l'écart entre les moustaches": "B"}'''),
        ("c", "a whole number", "1",
         r'''mistakes={"un point isolé est dessiné au-delà de la moustache, à part": 0,
          "ne compte que les points de la boîte de A": 2}'''),
        ("d", "a whole number", "5",
         r'''mistakes={"ce sont les folds où B fait mieux que A : la question demande l'inverse": 25,
          "compte les points sous la ligne horizontale du panneau (b)": 0}'''),
        ("e", "True or False", "True",
         r'''mistakes={"regarde le panneau (b) : combien de points sont au-dessus de zéro ?": False}'''),
        ("f", "True or False", "False",
         r'''mistakes={"regarde le bas de la boîte de C, sa moustache et ses points isolés : ses plus mauvais folds sont les pires des trois modèles": True}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 8.1 à 8.5, ∂ 8.6 et 📈 8.8)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.375` (en "
               "Python, le séparateur décimal est un **point** ; `0,375` sans guillemets serait un couple de deux "
               "nombres) ; un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets "
               "(`\"E\"`) ; plusieurs choix : les lettres collées (`\"AC\"`) ; plusieurs nombres : une liste "
               "(`[2, 5]`). Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas "
               "encore remplies affichent ⏳. Les questions « dans ta copie », le quiz Q8, le rappel R3, la "
               "réflexion et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Part A: splitting (8.11 to 8.15)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools for the notebook exercises (parts A to D)
import copy
import itertools
import math
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.model_selection import cross_val_score as sklearn_cross_val_score
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.neighbors import KNeighborsRegressor, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


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


def fitted(model, X, y=None):
    """The model after model.fit(X) or model.fit(X, y): no chained call, so that a fit that forgets
    `return self` (the tests report it) does not stop the check cells."""
    if y is None:
        model.fit(X)
    else:
        model.fit(X, y)
    return model


def split_pairs(ex_id, name, splits, n_splits):
    """True when `splits` is a list of n_splits pairs (train_idx, val_idx); otherwise a ❌ message and False."""
    if not returned(ex_id, name, splits):
        return False
    if not isinstance(splits, (list, tuple)):
        print(f"❌ Ex {ex_id} : {name} doit renvoyer une liste de paires (train_idx, val_idx), pas un objet de type "
              f"{type(splits).__name__} (un générateur ne se relit pas : renvoie une liste).")
        return False
    if len(splits) != n_splits or not all(isinstance(pair, (tuple, list)) and len(pair) == 2 for pair in splits):
        print(f"❌ Ex {ex_id} : {name} doit renvoyer une liste de {n_splits} paires (train_idx, val_idx).")
        return False
    return True


def as_letters(value):
    """Letters typed as "B, D", "bd" or ["D", "B"] -> "BD" (sorted, upper case); other values are returned unchanged."""
    if isinstance(value, (list, tuple)):
        value = "".join(str(item) for item in value)
    if isinstance(value, str):
        letters = sorted({char.upper() for char in value if char.isalpha()})
        if letters and all(len(letter) == 1 and "A" <= letter <= "Z" for letter in letters):
            return "".join(letters)
    return value


def print_answer(ex_id, value, **_):
    """Solutions notebook: show the value that the exercise notebook checks with wb.check."""
    print(f"{ex_id}:", np.round(value, 4).tolist() if isinstance(value, (np.ndarray, list)) else value)


TEST_FILE = "tests/test_ch08_model_selection.py"


def run_mylearn_tests(keyword, impl="learner"):
    """Run the tests of mylearn.model_selection selected by `keyword` (on YOUR code by default)."""
    command = [sys.executable, "-m", "pytest", TEST_FILE, "-k", keyword, "-q", "-p", "no:cacheprovider",
               "--color=no", "-rf", "--tb=no"]
    if impl != "learner":
        command.append(f"--impl={impl}")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": "1000"})   # long lines: the reason of each failure
    lines = result.stdout.strip().splitlines()
    failed = [line for line in lines if line.startswith("FAILED ")]
    if any("'NoneType' object has no attribute" in line for line in failed):
        print("💡 un attribut est lu sur None : une fonction oublie-t-elle son return ?")
    for line in failed[:8]:                                  # the test, then the reason of its failure
        name, _, reason = line.removeprefix(f"FAILED {TEST_FILE}::").partition(" - ")
        print(f"❌ {name}\n   {reason[:800]}")
    if len(failed) > 8:
        print(f"   ... and {len(failed) - 8} other failed test(s)")
    print("pytest:", lines[-1] if lines else result.stderr.strip()[-300:])


class OneNN:
    """The nearest-neighbour classifier (1-NN, ch. 7): fit memorises the training samples, predict copies
    the label of the closest one (the first one on ties). No hyperparameter."""

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float).copy()
        self.y_ = np.asarray(y).copy()
        return self

    def predict(self, X):
        return self.y_[cdist(np.asarray(X, dtype=float), self.X_, "sqeuclidean").argmin(axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
penguins = wb.datasets.load_penguins(dropna=True)        # the 333 complete penguins of ch. 1, sorted by species
X_peng = penguins[FEATURES].to_numpy(dtype=float)
species = penguins["species"].to_numpy()
PERM_CH1 = np.random.default_rng(42).permutation(len(penguins))   # the split of ch. 1 and 7: 233 + 100
TEST_CH1, TRAIN_CH1 = PERM_CH1[:100], PERM_CH1[100:]
print(f"{len(penguins)} penguins:", {name: int(count) for name, count in zip(*np.unique(species, return_counts=True))},
      "· species of the first, middle and last rows:", species[[0, 166, -1]].tolist())'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="8")   # reload your saved file\n'


def indented(code_text: str, spaces: int) -> str:
    """The code with every non-empty line indented by `spaces` spaces (to put a block inside an `if`)."""
    return "\n".join((" " * spaces + line) if line.strip() else "" for line in code_text.splitlines())


def solved(check: str, ex_id: str) -> str:
    """The check cell turned into the solution cell: no wb.attempt, the values are shown instead of checked
    (print_answer), and the tests run on the reference."""
    return (check.replace(f'with wb.attempt("{ex_id}"):\n', "if True:\n")
            .replace("wb.check(", "print_answer(")
            .replace('run_mylearn_tests("', 'run_mylearn_tests(impl="ref", keyword="'))


MYLEARN_HOWTO = ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre "
                 "`mon_travail/mylearn/model_selection.py` (créé par `python tools/start_chapter.py 8`), lis la "
                 "docstring de chaque fonction, remplace les `raise NotImplementedError(...)` par ton code et "
                 "**enregistre**. NumPy, `math`, `copy` et `warnings` sont permis ; scikit-learn non : ses "
                 "fonctions (`train_test_split`, `KFold`, `StratifiedKFold`, `cross_val_score`, `clone`) sont les "
                 "**oracles** des tests. La cellule de vérification recharge ta librairie, montre quelques "
                 "résultats, puis lance les tests de ta fonction ; `python -m pytest "
                 "tests/test_ch08_model_selection.py -q` les lance tous dans un terminal.")

MYLEARN_SHORT = ("> **mylearn** : dans `mon_travail/mylearn/model_selection.py`, mêmes règles qu'en 8.13 (NumPy "
                 "permis, scikit-learn non). Enregistre, puis relance la cellule de vérification.")

TODO_11 = r'''n_test_11a = ...       # a) number of test penguins (test_size=0.2, random_state=42)
chinstrap_11b = ...    # b) number of Chinstrap in that test part
chinstrap_11c = ...    # c) the same with stratify=species
n_species_11d = ...    # d) number of different species in the test part with shuffle=False
range_11e = ...        # e) [smallest, largest] number of Chinstrap in the test part, random_state = 0 to 99'''

SOLUTION_11 = r'''X_tr_11, X_te_11, y_tr_11, y_te_11 = sklearn_train_test_split(X_peng, species, test_size=0.2, random_state=42)
n_test_11a = len(X_te_11)
chinstrap_11b = int(np.sum(y_te_11 == "Chinstrap"))
y_te_strat_11 = sklearn_train_test_split(X_peng, species, test_size=0.2, random_state=42, stratify=species)[3]
chinstrap_11c = int(np.sum(y_te_strat_11 == "Chinstrap"))
y_te_ordered_11 = sklearn_train_test_split(X_peng, species, test_size=0.2, shuffle=False)[3]
n_species_11d = len(np.unique(y_te_ordered_11))
counts_11 = [int(np.sum(sklearn_train_test_split(X_peng, species, test_size=0.2, random_state=seed)[3] == "Chinstrap"))
             for seed in range(100)]
range_11e = [min(counts_11), max(counts_11)]
print("a)", n_test_11a, "· b)", chinstrap_11b, "· c)", chinstrap_11c, "· d)", n_species_11d, "· e)", range_11e)
print("species of the test part with shuffle=False:", np.unique(y_te_ordered_11).tolist())
plt.figure(figsize=(6, 3))
plt.hist(counts_11, bins=np.arange(min(counts_11), max(counts_11) + 2) - 0.5, color="tab:blue")
plt.axvline(68 * 67 / 333, color="black", ls="--", label="68 × 67 / 333 (stratified share)")
plt.xlabel("Chinstrap in the test part (100 random_state values)")
plt.ylabel("count")
plt.legend()
plt.show()'''

CHECK_11 = r'''wb.check("8.11a", n_test_11a)
wb.check("8.11b", chinstrap_11b)
wb.check("8.11c", chinstrap_11c)
wb.check("8.11d", n_species_11d)
wb.check("8.11e", range_11e)'''

DATA_12 = r'''Z_12 = (X_peng - X_peng[TRAIN_CH1].mean(axis=0)) / X_peng[TRAIN_CH1].std(axis=0)   # standardised with the training rows
shuffled_12 = np.random.default_rng(812).permutation(species)   # the species, shuffled at random among the penguins
print("the first five penguins, true and shuffled species:", species[:5].tolist(), shuffled_12[:5].tolist())'''

EXPERIMENT_12 = r'''accuracies_12 = {}
for name_12, labels_12 in [("true species", species), ("shuffled species", shuffled_12)]:
    model_12 = OneNN().fit(Z_12[TRAIN_CH1], labels_12[TRAIN_CH1])
    accuracies_12[(name_12, "train")] = model_12.score(Z_12[TRAIN_CH1], labels_12[TRAIN_CH1])
    accuracies_12[(name_12, "test")] = model_12.score(Z_12[TEST_CH1], labels_12[TEST_CH1])
for (name_12, part_12), value_12 in accuracies_12.items():
    print(f"{name_12:17s} {part_12:5s}: accuracy {value_12:.2f}")
shares_12 = np.unique(shuffled_12[TRAIN_CH1], return_counts=True)[1] / len(TRAIN_CH1)
print(f"chance level of the shuffled labels (sum of the squared class shares): {np.sum(shares_12 ** 2):.2f}")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
names_12 = [f"{name}\n{part}" for name, part in accuracies_12]
ax.bar(names_12, list(accuracies_12.values()), color=["tab:blue", "tab:orange", "tab:blue", "tab:orange"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("accuracy")
ax.set_title("1-NN: training (blue) and test (orange) accuracy")
plt.show()'''

DATA_15 = r'''# 20 samples and 5 folds, as in figure 8.13 of the book; then shuffled; then stratified on 20 labels
LABELS_15 = np.array([0] * 12 + [1] * 8)                  # 12 samples of class 0, then 8 of class 1'''

TODO_15 = r'''def fold_matrix_15(splits, n):
    """Matrix of shape (len(splits), n): 1 where sample j is in the validation fold of split i, 0 elsewhere."""
    raise NotImplementedError


def draw_folds_15(ax, splits, n, title):
    """Draw fold_matrix_15(splits, n) on ax: one row per split, validation and training in two colours."""
    raise NotImplementedError'''

SOLUTION_15 = r'''def fold_matrix_15(splits, n):
    """Matrix of shape (len(splits), n): 1 where sample j is in the validation fold of split i, 0 elsewhere."""
    matrix = np.zeros((len(splits), n), dtype=int)
    for i, (_, val_idx) in enumerate(splits):
        matrix[i, val_idx] = 1
    return matrix


def draw_folds_15(ax, splits, n, title):
    """Draw fold_matrix_15(splits, n) on ax: one row per split, validation and training in two colours."""
    from matplotlib.colors import ListedColormap
    ax.imshow(fold_matrix_15(splits, n), cmap=ListedColormap(["tab:blue", "tab:orange"]), aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(splits)), [f"round {i + 1}" for i in range(len(splits))])
    ax.set_xticks(range(0, n, 5))
    ax.set_xlabel("sample")
    ax.set_title(title, fontsize=10)


'''

CHECK_15 = r'''with wb.attempt("8.15"):
    splits_15 = mylearn.model_selection.kfold_indices(20, n_splits=5)
    good_15 = split_pairs("8.15", "kfold_indices (8.14)", splits_15, 5)
    matrix_15 = fold_matrix_15(splits_15, 20) if good_15 else None
    if good_15 and returned("8.15", "fold_matrix_15", matrix_15):
        matrix_15 = np.asarray(matrix_15)
        expected_15 = np.kron(np.eye(5, dtype=int), np.ones((1, 4), dtype=int))     # fold i = samples 4i to 4i + 3
        verdict("8.15", matrix_15.shape == (5, 20) and np.array_equal(matrix_15, expected_15),
                "la matrice de la rotation : chaque échantillon passe une fois en validation, les folds se suivent.",
                f"attendu une matrice (5, 20) de 0 et de 1, avec des 1 aux échantillons du fold de validation de "
                f"chaque tour ; reçu une forme {matrix_15.shape}.")
        try:
            stratified_15 = mylearn.model_selection.stratified_kfold_indices(LABELS_15, 4)
        except NotImplementedError:
            stratified_15 = None
            print("⏳ la troisième figure, stratifiée, s'affichera quand tu auras écrit stratified_kfold_indices (8.21).")
        if stratified_15 is not None and not split_pairs("8.15", "stratified_kfold_indices (8.21)", stratified_15, 4):
            stratified_15 = None
        fig, axes = plt.subplots(2 if stratified_15 is None else 3, 1, figsize=(8, 4.8 if stratified_15 is None else 7))
        try:
            draw_folds_15(axes[0], splits_15, 20, "5-fold, no shuffle (figure 8.13 of the book)")
        except NotImplementedError:
            plt.close(fig)                                  # no empty figure before draw_folds_15 is written
            raise
        draw_folds_15(axes[1], mylearn.model_selection.kfold_indices(20, 5, shuffle=True, rng=np.random.default_rng(815)),
                      20, "5-fold, shuffled once")
        if stratified_15 is not None:
            draw_folds_15(axes[2], stratified_15, 20, "stratified 4-fold on 12 + 8 labels (exercise 8.21)")
        fig.tight_layout()
        plt.show()'''

PART_A = Part("A", "Découper : hold-out, stratification et k-fold",
              "Les manchots du ch. 1 (333 complets) servent de fil rouge : `X_peng` contient leurs quatre mesures, "
              "`species` leur espèce. **Attention** : le fichier est trié par espèce (Adélie, puis Gentoo, puis "
              "Chinstrap), comme beaucoup de fichiers réels. `TRAIN_CH1` et `TEST_CH1` sont les indices du "
              "découpage du ch. 1 (233 manchots d'entraînement, 100 de test, graine 42). La cellule suivante "
              "définit aussi les outils des parties A à D, dont `OneNN`, le classifieur du plus proche voisin "
              "(ch. 7). Les valeurs que tu calcules dans une cellule à compléter ne s'arrondissent pas : la "
              "vérification s'en charge.",
              given=PART_A_GIVEN, exercises=[
    Ex("8.11", "📦", 1, 10, "train_test_split de scikit-learn : tailles, stratify, random_state",
       "découper des données avec scikit-learn, et voir ce que changent la graine, la stratification et l'absence "
       "de mélange.",
       "Ex 8.1 · fiche §8.3", thread="Penguins", tracks="R, M, C",
       body=r"""La fonction `train_test_split` de scikit-learn est importée sous le nom `sklearn_train_test_split` (ta propre version, en 8.13, s'appellera `mylearn.model_selection.train_test_split`). Elle renvoie `[X_train, X_test, y_train, y_test]`.

a) `n_test_11a` : le nombre de manchots de test avec `test_size=0.2` et `random_state=42`.
b) `chinstrap_11b` : le nombre de Chinstrap dans ce jeu de test.
c) `chinstrap_11c` : la même chose avec `stratify=species` (toujours `test_size=0.2`, `random_state=42`).
d) `n_species_11d` : le nombre d'espèces **différentes** dans le jeu de test obtenu avec `shuffle=False` (et `test_size=0.2`).
e) `range_11e` : sans stratification, refais le découpage de a) pour chaque `random_state` de 0 à 99 ; donne la liste `[plus petit, plus grand]` des nombres de Chinstrap dans le test.

Dans tes notes : combien de Chinstrap la stratification place-t-elle dans le test, et pourquoi ce nombre ? Que montre e) sur un découpage non stratifié d'un petit dataset ? Pourquoi d) est-il un piège fréquent ?""",
       todo=TODO_11, check=CHECK_11, solution=SOLUTION_11,
       record=r'''wb.record("8.11a", n_test_11a, mistakes={"0,2 × 333 = 66,6 : scikit-learn arrondit la taille du test vers le haut": 66,
                                       "c'est la taille de l'entraînement": 266})
wb.record("8.11b", chinstrap_11b, mistakes={"c'est le nombre obtenu avec stratify : ici, sans stratify": chinstrap_11c})
wb.record("8.11c", chinstrap_11c, mistakes={"c'est le nombre obtenu sans stratify : ajoute stratify=species": chinstrap_11b})
wb.record("8.11d", n_species_11d, mistakes={"regarde les espèces des 67 dernières lignes : le fichier est trié par espèce": 3})
wb.record("8.11e", range_11e)''',
       note="La stratification place 14 Chinstrap dans le test : leur part exacte vaut $68 \\times 67 / 333 \\approx "
            "13{,}7$, arrondie par la règle du plus fort reste (scikit-learn répartit d'abord l'entraînement, ce "
            "qui revient ici au même). Sans stratification, la graine 42 en donne 18, et les graines 0 à 99 "
            "donnent de 6 à 25 Chinstrap : un test de 67 manchots peut contenir deux fois trop ou deux fois trop "
            "peu d'une classe rare. Avec `shuffle=False`, les 67 dernières lignes sont toutes des Chinstrap : "
            "le modèle s'entraînerait sur une seule Chinstrap et serait testé sur elles seules."),

    Ex("8.12", "🔮", 1, 15, "Le modèle qui apprend par cœur : accuracy d'entraînement et de test",
       "prévoir l'accuracy d'un modèle qui retient tous ses exemples, sur l'entraînement et sur le test, avec de "
       "vrais labels et avec des labels tirés au hasard.",
       "ch. 1 (1.14, le mémoriseur) · ch. 7 (plus proche voisin) · fiche §8.2.1", thread="Penguins",
       tracks="C", hypothesis=True,
       body=r"""Au ch. 1 (1.14), le mémoriseur ne savait répondre qu'aux manchots qu'il avait déjà vus. `OneNN` apprend aussi « par cœur » (son `fit` range tous les exemples), mais il répond à tout : il copie le label du manchot d'entraînement **le plus proche** (ch. 7). On l'entraîne sur les 233 manchots du découpage du ch. 1, avec leurs quatre mesures standardisées (`Z_12`, moyenne et écart-type des 233 manchots d'entraînement), puis on mesure son accuracy sur ces mêmes 233 manchots et sur les 100 du test. Deux expériences :
- avec les **vraies espèces** ;
- avec des espèces **mélangées au hasard** entre les manchots (`shuffled_12`) : le label n'a plus aucun lien avec les mesures.

Prédis, **avant** d'exécuter quoi que ce soit, l'intervalle de chacune des quatre accuracies : `"A"` exactement 1 ; `"B"` entre 0,9 et 1, sans atteindre 1 ; `"C"` entre 0,5 et 0,9 ; `"D"` moins de 0,5.
a) entraînement, vraies espèces ;
b) test, vraies espèces ;
c) entraînement, espèces mélangées ;
d) test, espèces mélangées.

Puis exécute l'expérience. Dans tes notes : pourquoi c) vaut-il ce qu'il vaut ? Que dit l'expérience du score d'entraînement d'un modèle très flexible ? À quoi correspond la valeur de d) (le niveau du hasard affiché par l'expérience) ?""",
       given=DATA_12,
       todo=r'''prediction_12a = ...   # "A", "B", "C" or "D": training accuracy, true species
prediction_12b = ...   # test accuracy, true species
prediction_12c = ...   # training accuracy, shuffled species
prediction_12d = ...   # test accuracy, shuffled species''',
       check=r'''for letter_12, answer_12 in zip("abcd", [prediction_12a, prediction_12b, prediction_12c, prediction_12d]):
    wb.check(f"8.12{letter_12}", answer_12)''',
       solution=r'''prediction_12a, prediction_12b, prediction_12c, prediction_12d = "A", "B", "A", "D"   # the answers, for the record''',
       record=r'''wb.record("8.12a", prediction_12a, mistakes={"chaque manchot d'entraînement est dans la mémoire du modèle : quel est son plus proche voisin ?": "B"})
wb.record("8.12b", prediction_12b, mistakes={"sur des manchots nouveaux, le plus proche voisin peut se tromper": "A",
                                             "les espèces se séparent bien avec les quatre mesures (ch. 1 et 7)": "C"})
wb.record("8.12c", prediction_12c, mistakes={"le label d'un manchot d'entraînement, même tiré au hasard, est dans la mémoire du modèle": "D",
                                             "quel est le plus proche voisin d'un manchot d'entraînement ?": "B"})
wb.record("8.12d", prediction_12d, mistakes={"le label d'un manchot de test n'a plus aucun lien avec ses mesures : que vaut copier celui d'un voisin ?": "C"})''',
       after=[("code", guarded(EXPERIMENT_12, ["prediction_12a", "prediction_12b", "prediction_12c", "prediction_12d"],
                               "⏳ Ex 8.12 : écris d'abord tes quatre prédictions, puis relance cette cellule."))],
       note="Sur l'entraînement, le plus proche voisin d'un manchot est lui-même (distance nulle, et aucun manchot "
            "n'a exactement les mêmes quatre mesures qu'un autre) : l'accuracy vaut **exactement 1**, avec les "
            "vraies espèces comme avec des espèces tirées au hasard. Sur le test, les vraies espèces donnent 0,99, "
            "les espèces mélangées 0,36, soit le niveau du hasard (environ $\\sum_k p_k^2 \\approx 0{,}37$ : la "
            "probabilité que deux manchots tirés au hasard aient le même label). Le score d'entraînement d'un modèle qui peut "
            "tout retenir ne distingue donc pas un modèle qui a appris d'un modèle qui n'a rien appris : seul le "
            "test les sépare. C. Zhang et ses collègues ont montré en 2017 que de grands réseaux de neurones "
            "retiennent eux aussi des labels tirés au hasard."),

    Ex("8.13", "🔨", 2, 30, "train_test_split from scratch",
       "écrire un découpage hold-out complet : tailles, mélange, stratification, alignement de plusieurs tableaux.",
       "Ex 8.11 · 0A (`rng.permutation`, indexation par un tableau d'indices) · fiche §8.3", thread="Penguins",
       tracks="R, M, C", mylearn="model_selection.py",
       body=MYLEARN_HOWTO + r"""

Écris `train_test_split(*arrays, test_size=0.25, shuffle=True, stratify=None, rng=None)` (lis sa docstring) :
- convertis chaque tableau avec `np.asarray` ; lève une `ValueError` s'il n'y a aucun tableau ou si leurs nombres de lignes diffèrent ;
- la taille du test : un `float` de $]0, 1[$ donne $n_{\text{test}} = \lceil t \cdot n \rceil$ (`math.ceil(test_size * n)`), un `int` donne directement $n_{\text{test}}$ ; lève une `ValueError` si `test_size` est hors de ces bornes ou si l'une des deux parties serait vide ;
- choisis **une seule fois** les indices du test et ceux de l'entraînement, puis indexe **tous** les tableaux avec ces mêmes indices : c'est ce qui garde `X` et `y` alignés. Le résultat : `[a1_train, a1_test, a2_train, a2_test, ...]` ;
- `shuffle=False` : le test est formé des **dernières** lignes, dans leur ordre ; `shuffle=True` : tire les indices au hasard avec `rng` (`np.random.default_rng()` si `rng` vaut `None`), par exemple avec `rng.permutation(n)` ;
- `stratify=y` : chaque classe reçoit dans le test la partie entière de sa part exacte $n_c \cdot n_{\text{test}} / n$, et les exemples qui manquent vont aux classes dont la partie décimale est la plus grande (règle du plus fort reste, fiche §8.3) ; tire ensuite au hasard, avec `rng`, les lignes de chaque classe. Lève une `ValueError` avec `shuffle=False`, ou si une classe n'a qu'un seul membre.

Ne modifie jamais les tableaux reçus (mélange des indices, pas des données). La vérification essaie l'exemple de la docstring, puis un découpage stratifié des manchots, et lance les tests.

Dans tes notes : pourquoi faut-il tirer les indices **une seule fois** pour tous les tableaux ? Pourquoi les tests comparent-ils la **taille** de ton test à celle de scikit-learn, mais pas les manchots choisis ?""",
       check=RELOAD + r'''with wb.attempt("8.13"):
    X_doc_13, y_doc_13 = np.arange(10).reshape(5, 2), np.array([0, 1, 0, 1, 0])
    parts_13 = mylearn.model_selection.train_test_split(X_doc_13, y_doc_13, test_size=0.4, shuffle=False)
    if returned("8.13", "train_test_split", parts_13) and len(parts_13) == 4:
        print("docstring example: X_test =", np.asarray(parts_13[1]).tolist(), "· y_train =", np.asarray(parts_13[2]).tolist())
    parts_13 = mylearn.model_selection.train_test_split(X_peng, species, test_size=0.2, stratify=species,
                                                        rng=np.random.default_rng(813))
    if returned("8.13", "train_test_split", parts_13):
        if len(parts_13) != 4:
            print(f"❌ Ex 8.13 : attendu 4 tableaux [X_train, X_test, y_train, y_test], reçu {len(parts_13)}.")
        else:
            X_tr_13, X_te_13, y_tr_13, y_te_13 = (np.asarray(part) for part in parts_13)
            counts_13 = {name: int(np.sum(y_te_13 == name)) for name in ("Adelie", "Chinstrap", "Gentoo")}
            print("stratified test part of the penguins:", len(X_te_13), "penguins,", counts_13)
            shares_13 = {name: np.sum(species == name) * 67 / 333 for name in counts_13}
            verdict("8.13", len(X_te_13) == 67 and all(math.floor(shares_13[n]) <= counts_13[n] <= math.ceil(shares_13[n])
                                                         for n in counts_13),
                    "67 manchots de test, et chaque espèce garde sa proportion.",
                    "attendu 67 manchots de test (⌈0,2 × 333⌉), chaque espèce recevant la partie entière de sa part "
                    "exacte ou un de plus.")
            rows_13 = {tuple(row): name for row, name in zip(X_peng, species)}
            verdict("8.13", all(rows_13.get(tuple(row)) == name for row, name in zip(X_te_13, y_te_13)),
                    "X_test et y_test restent alignés : chaque ligne garde l'espèce de son manchot.",
                    "une ligne de X_test n'a pas l'espèce de son manchot : les lignes de X et de y doivent être prises "
                    "avec les mêmes indices.")
    run_mylearn_tests("test_train_test_split_")''',
       solution=r'''parts_13 = mylearn.model_selection.train_test_split(X_peng, species, test_size=0.2, stratify=species,
                                                    rng=np.random.default_rng(813))
X_tr_13, X_te_13, y_tr_13, y_te_13 = parts_13
print("stratified test part of the penguins:", len(X_te_13), "penguins,",
      {name: int(np.sum(y_te_13 == name)) for name in ("Adelie", "Chinstrap", "Gentoo")})
run_mylearn_tests(impl="ref", keyword="test_train_test_split_")''',
       note="Dans la référence (`solutions/mylearn_ref/model_selection.py`, à lire **après** avoir réussi les "
            "tests), une fonction auxiliaire calcule $n_{\\text{test}}$ et ses contrôles ; avec `stratify`, les "
            "quotas par classe viennent du plus fort reste, puis `rng.permutation` tire les lignes de chaque classe ; "
            "enfin, une seule paire d'indices sert à indexer tous les tableaux. Les tests ne comparent pas les "
            "manchots choisis à ceux de scikit-learn : deux générateurs différents tirent d'autres lignes, et "
            "les deux découpages sont justes. Ils vérifient les **propriétés** (tailles, parties disjointes, "
            "alignement, même graine = même découpage) et les effectifs par classe."),

    Ex("8.14", "🔨", 2, 25, "Les indices de la k-fold : kfold_indices",
       "écrire les indices (entraînement, validation) des k tours d'une validation croisée.",
       "Ex 8.1 · Ex 8.13 · fiche §8.5.1", tracks="R, M, C", mylearn="model_selection.py",
       body=MYLEARN_SHORT + r"""

Écris `kfold_indices(n_samples, n_splits=5, shuffle=False, rng=None)` (lis sa docstring) :
- lève une `ValueError` si `n_splits < 2` ou si `n_splits > n_samples` ;
- les indices `0, 1, …, n_samples - 1` (mélangés **une fois** avec `rng.permutation` si `shuffle=True`) sont coupés en `n_splits` folds **consécutifs** ; les `n_samples % n_splits` premiers folds ont un exemple de plus (fiche §8.5.1) ;
- renvoie une **liste** de `n_splits` paires `(train_idx, val_idx)` : le fold `i` est la validation du tour `i`, tous les autres indices son entraînement. Sans mélange, les indices de chaque partie sont en ordre croissant, comme ceux de `KFold` de scikit-learn, l'oracle des tests.

La vérification affiche l'exemple de la docstring, compare tes folds à ceux de `KFold` sur les 333 manchots, puis lance les tests.

Dans tes notes : que se passe-t-il si l'on applique `kfold_indices(333, 5)` **sans mélange** aux manchots, triés par espèce ? Regarde les espèces de chaque fold de validation.""",
       check=RELOAD + r'''with wb.attempt("8.14"):
    splits_14 = mylearn.model_selection.kfold_indices(5, n_splits=2)
    if split_pairs("8.14", "kfold_indices", splits_14, 2):
        for train_14, val_14 in splits_14:
            print("docstring example:", np.asarray(train_14).tolist(), np.asarray(val_14).tolist())
    splits_14 = mylearn.model_selection.kfold_indices(len(species), n_splits=5)
    if split_pairs("8.14", "kfold_indices", splits_14, 5):
        oracle_14 = list(KFold(n_splits=5).split(X_peng))
        same_14 = len(splits_14) == 5 and all(np.array_equal(np.asarray(a), b) and np.array_equal(np.asarray(c), d)
                                              for (a, c), (b, d) in zip(splits_14, oracle_14))
        verdict("8.14", same_14, "les mêmes indices que KFold(5) de scikit-learn sur les 333 manchots.",
                "tes indices diffèrent de ceux de KFold(5) : relis la règle des tailles et l'ordre des indices.")
        for i_14, (_, val_14) in enumerate(splits_14):
            names_14, counts_14 = np.unique(species[np.asarray(val_14)], return_counts=True)
            print(f"fold {i_14 + 1}: {len(val_14)} penguins,", dict(zip(names_14.tolist(), counts_14.tolist())))
    run_mylearn_tests("test_kfold_indices_")''',
       solution=r'''for i_14, (train_14, val_14) in enumerate(mylearn.model_selection.kfold_indices(len(species), n_splits=5)):
    names_14, counts_14 = np.unique(species[val_14], return_counts=True)
    print(f"fold {i_14 + 1}: {len(val_14)} penguins,", dict(zip(names_14.tolist(), counts_14.tolist())))
run_mylearn_tests(impl="ref", keyword="test_kfold_indices_")''',
       note="333 = 5 × 66 + 3 : trois folds de 67 et deux de 66. Sans mélange, sur les manchots triés par espèce, "
            "les deux premiers folds ne contiennent que des Adélie, le troisième 12 Adélie et 55 Gentoo, le "
            "quatrième 64 Gentoo et 2 Chinstrap, le dernier 66 Chinstrap : au dernier tour, le modèle "
            "s'entraîne avec seulement deux Chinstrap et doit reconnaître toutes les autres. Il faut mélanger "
            "(`shuffle=True`) ou stratifier (8.21)."),

    Ex("8.15", "🎨", 2, 20, "Reproduire la figure 8.13 : la rotation des folds",
       "dessiner la rotation des folds d'une validation croisée à partir des indices, et la comparer avec un "
       "mélange et une stratification.",
       "Ex 8.14 · livre §8.5.1 (figure 8.13)", tracks="C",
       body=r"""La figure 8.13 du livre montre les 5 tours d'une validation croisée à 5 folds : à chaque tour, un fold sert de validation et les autres d'entraînement. Reproduis-la à partir de **tes** indices.

Écris :
- `fold_matrix_15(splits, n)` : un tableau d'entiers de forme `(len(splits), n)`, qui vaut 1 en ligne `i`, colonne `j` quand l'échantillon `j` est dans le fold de validation du tour `i`, et 0 sinon ;
- `draw_folds_15(ax, splits, n, title)` : le dessin de cette matrice sur `ax`, un tour par ligne, deux couleurs (par exemple `ax.imshow(..., cmap=ListedColormap(["tab:blue", "tab:orange"]), aspect="auto")`, avec `from matplotlib.colors import ListedColormap`), des graduations lisibles et le titre.

La vérification contrôle ta matrice pour 20 échantillons et 5 folds, puis dessine trois versions : la figure du livre, la même après un mélange, et une validation croisée **stratifiée** à 4 folds sur 12 échantillons d'une classe et 8 d'une autre (elle utilise ta fonction de 8.21 si tu l'as déjà écrite, et affiche ⏳ sinon : reviens alors ici après 8.21).

Dans tes notes : combien de 1 dans chaque colonne de la matrice, et pourquoi ? Dans la version stratifiée, combien d'échantillons de chaque classe chaque fold reçoit-il ?""",
       given=DATA_15, todo=TODO_15, check=RELOAD + CHECK_15,
       solution=SOLUTION_15 + solved(CHECK_15, "8.15"),
       note="Chaque colonne contient exactement un 1 : chaque échantillon sert une fois, et une seule, de "
            "validation. Sans mélange, les folds sont des blocs consécutifs, et la matrice est une suite de blocs "
            "le long de la diagonale ; après un mélange, les 1 d'un tour sont dispersés. Dans la version stratifiée, "
            "chaque fold de validation reçoit 3 échantillons de la classe 0 et 2 de la classe 1 (12 et 8 répartis "
            "sur 4 folds)."),
])

# ---------------------------------------------------------------------------
# Part B: choosing a hyperparameter on a validation set (8.16 to 8.20)
# ---------------------------------------------------------------------------
PART_B_GIVEN = r'''california = wb.datasets.load_california()       # 20 640 districts of California (1990 census)
keep_cal = (california["MedInc"] >= 1) & (california["MedInc"] <= 8) & (california["MedHouseVal"] < 5)
income_all = california.loc[keep_cal, "MedInc"].to_numpy()       # median income of a district (tens of thousands of $)
value_all = california.loc[keep_cal, "MedHouseVal"].to_numpy()   # median house value (hundreds of thousands of $)
rows_cal = np.random.default_rng(937).choice(len(income_all), size=400, replace=False)   # 400 districts, in random order
X_cal = income_all[rows_cal].reshape(-1, 1)        # one feature, but a 2-D X of shape (400, 1), as in scikit-learn
y_cal = value_all[rows_cal]
print(f"{len(income_all)} districts kept, {len(X_cal)} drawn at random: income from {X_cal.min():.2f} to "
      f"{X_cal.max():.2f}, house value from {y_cal.min():.2f} to {y_cal.max():.2f}")'''

TODO_16 = r'''class PolyFit:
    """Polynomial regression of y on the single feature X[:, 0], fitted by least squares (np.polyfit)."""

    def __init__(self, degree=1):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError'''

SOLUTION_16 = r'''class PolyFit:
    """Polynomial regression of y on the single feature X[:, 0], fitted by least squares (np.polyfit)."""

    def __init__(self, degree=1):
        self.degree = degree                        # only store the hyperparameter (convention used by clone)

    def fit(self, X, y):
        x = np.asarray(X, dtype=float)[:, 0]
        self.coef_ = np.polyfit(x, np.asarray(y, dtype=float), self.degree)
        return self

    def predict(self, X):
        return np.polyval(self.coef_, np.asarray(X, dtype=float)[:, 0])

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        residual = np.sum((y - self.predict(X)) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        return float(1 - residual / total)


'''

CHECK_16 = r'''with wb.attempt("8.16"):
    toy_16 = fitted(PolyFit(degree=2), np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([1.0, 3.0, 7.0, 13.0]))
    pred_16 = toy_16.predict(np.array([[4.0]]))
    if returned("8.16", "predict", pred_16):
        pred_16 = np.asarray(pred_16, dtype=float).ravel()
        if pred_16.shape != (1,):
            print(f"❌ Ex 8.16 : predict doit renvoyer une valeur par ligne de X (ici une seule), reçu une forme {pred_16.shape}.")
        else:
            wb.check("8.16a", float(pred_16[0]), computed=True)
    params_16 = vars(PolyFit(degree=3))
    verdict("8.16", params_16 == {"degree": 3},
            "__init__ ne fait que ranger degree : clone (8.22) saura copier PolyFit.",
            f"vars(PolyFit(degree=3)) vaut {params_16} : __init__ doit seulement écrire self.degree = degree.")
    one_16 = PolyFit(degree=1)
    verdict("8.16", one_16.fit(X_cal, y_cal) is one_16, "fit renvoie le modèle lui-même.",
            "fit doit renvoyer le modèle lui-même (return self).")
    model_16 = fitted(PolyFit(degree=2), X_cal[:300], y_cal[:300])
    many_16 = model_16.predict(X_cal[300:305])
    if returned("8.16", "predict", many_16) and np.shape(many_16) != (5,):
        print(f"❌ Ex 8.16 : predict doit renvoyer une prédiction par ligne de X : forme (5,) attendue pour 5 "
              f"districts, reçu {np.shape(many_16)}.")
    r2_16 = model_16.score(X_cal[300:], y_cal[300:])
    if returned("8.16", "score", r2_16) and not isinstance(r2_16, (int, float)):
        print(f"❌ Ex 8.16 : score doit renvoyer un nombre (le R²), reçu un objet de type {type(r2_16).__name__}.")
    elif r2_16 is not None:
        wb.check("8.16b", r2_16, computed=True)
        oracle_16 = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_cal[:300], y_cal[:300])
        verdict("8.16", abs(float(r2_16) - oracle_16.score(X_cal[300:], y_cal[300:])) < 1e-8,
                "le même R² que PolynomialFeatures + LinearRegression de scikit-learn.",
                "ton R² diffère de celui de scikit-learn : relis la formule de l'encadré 🧮 de la fiche (§8.4).")
        if np.shape(many_16) == (5,):                     # draw only with a predict of the right shape
            grid_16 = np.linspace(1, 8, 200).reshape(-1, 1)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(X_cal[:300, 0], y_cal[:300], s=8, color="lightgray", label="300 training districts")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")              # np.polyfit warns that degree 8 is poorly conditioned
                for degree_16, colour_16 in [(1, "tab:blue"), (3, "tab:orange"), (8, "tab:red")]:
                    curve_16 = fitted(PolyFit(degree=degree_16), X_cal[:300], y_cal[:300]).predict(grid_16)
                    ax.plot(grid_16[:, 0], curve_16, color=colour_16, label=f"degree {degree_16}")
            ax.set(xlabel="median income (tens of thousands of $)", ylabel="median house value (100 000 $)",
                   ylim=(0, 5.2), title="PolyFit on the California districts")
            ax.legend(fontsize=8)
            plt.show()'''

DATA_17 = r'''def repeat_17(n_repeats=300):
    """For each repetition: 360 districts drawn at random, PolyFit of degree 1 to 12 trained on 40 of them, R² on 20
    validation districts and on 300 test districts. Returns, for each repetition, the validation and test R² of the
    degree chosen on the validation set, then those of degree 1 (fixed in advance)."""
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # high degrees on 40 points: np.polyfit warns, as expected
        for repetition in range(n_repeats):
            drawn = np.random.default_rng(1700 + repetition).choice(len(income_all), size=360, replace=False)
            x, v = income_all[drawn].reshape(-1, 1), value_all[drawn]
            val_scores, test_scores = [], []
            for degree in range(1, 13):
                model = fitted(PolyFit(degree=degree), x[:40], v[:40])
                val_scores.append(model.score(x[40:60], v[40:60]))
                test_scores.append(model.score(x[60:], v[60:]))
            best = int(np.argmax(val_scores))
            rows.append((val_scores[best], test_scores[best], val_scores[0], test_scores[0]))
    return np.array(rows, dtype=float)'''

EXPERIMENT_17 = r'''with wb.attempt("8.17"):
    trial_17 = fitted(PolyFit(degree=2), X_cal[:40], y_cal[:40]).score(X_cal[40:60], y_cal[40:60])
    if not isinstance(trial_17, (int, float)):        # np.float64 is a float
        print(f"❌ Ex 8.17 : l'expérience utilise ta classe PolyFit, dont score renvoie {type(trial_17).__name__} "
              "au lieu d'un nombre : termine d'abord 8.16.")
    else:
        rows_17 = repeat_17()
        share_chosen_17 = np.mean(rows_17[:, 0] > rows_17[:, 1])
        share_fixed_17 = np.mean(rows_17[:, 2] > rows_17[:, 3])
        print(f"validation R² above test R²: chosen degree {share_chosen_17:.0%} of the repetitions, degree 1 {share_fixed_17:.0%}")
        print(f"medians: chosen degree, validation {np.median(rows_17[:, 0]):.3f} and test {np.median(rows_17[:, 1]):.3f}; "
              f"degree 1, validation {np.median(rows_17[:, 2]):.3f} and test {np.median(rows_17[:, 3]):.3f}")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        bins_17 = np.linspace(-0.6, 0.6, 49)
        ax.hist(np.clip(rows_17[:, 0] - rows_17[:, 1], -0.6, 0.6), bins=bins_17, alpha=0.6,
                label="degree chosen on the validation set")
        ax.hist(np.clip(rows_17[:, 2] - rows_17[:, 3], -0.6, 0.6), bins=bins_17, alpha=0.6, label="degree 1, fixed in advance")
        ax.axvline(0, color="black", lw=1)
        ax.set(xlabel="validation R² − test R² (clipped to ±0.6)", ylabel="repetitions", title="300 repetitions")
        ax.legend(fontsize=8)
        plt.show()'''

TODO_18 = r'''def select_degree_18(X, y, degrees):
    """Choose the degree of PolyFit on a validation set, then refit it and test it once.

    Returns (val_scores, best_degree, test_score): the validation R² of each degree (a list, in the order of
    `degrees`), the degree with the highest one (the first one on ties), and the test R² of PolyFit(best_degree)
    refitted on training + validation."""
    raise NotImplementedError'''

SOLUTION_18 = r'''def select_degree_18(X, y, degrees):
    """Choose the degree of PolyFit on a validation set, then refit it and test it once.

    Returns (val_scores, best_degree, test_score): the validation R² of each degree (a list, in the order of
    `degrees`), the degree with the highest one (the first one on ties), and the test R² of PolyFit(best_degree)
    refitted on training + validation."""
    split = mylearn.model_selection.train_test_split
    X_rest, X_test, y_rest, y_test = split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = split(X_rest, y_rest, test_size=0.25, shuffle=False)
    val_scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # high degrees: np.polyfit warns, as expected
        for degree in degrees:
            val_scores.append(PolyFit(degree=degree).fit(X_train, y_train).score(X_val, y_val))
        best_degree = degrees[int(np.argmax(val_scores))]
        final = PolyFit(degree=best_degree).fit(X_rest, y_rest)          # training + validation
    return val_scores, best_degree, final.score(X_test, y_test)          # the test, once


'''

CHECK_18 = r'''with wb.attempt("8.18"):
    degrees_18 = list(range(1, 9))
    result_18 = select_degree_18(X_cal, y_cal, degrees_18)
    if returned("8.18", "select_degree_18", result_18):
        if not (isinstance(result_18, (tuple, list)) and len(result_18) == 3):
            print("❌ Ex 8.18 : select_degree_18 doit renvoyer un triplet (val_scores, best_degree, test_score).")
        else:
            val_18, best_18, test_18 = result_18
            wb.check("8.18a", best_18)
            wb.check("8.18b", test_18, computed=True)
            if np.shape(val_18) == (len(degrees_18),):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    train_18 = [fitted(PolyFit(degree=d), X_cal[:240], y_cal[:240]).score(X_cal[:240], y_cal[:240])
                                for d in degrees_18]
                fig, ax = plt.subplots(figsize=(6.5, 3.6))
                ax.plot(degrees_18, train_18, "o-", label="training R² (240 districts)")
                ax.plot(degrees_18, val_18, "s-", label="validation R² (80 districts)")
                ax.axvline(best_18, color="gray", ls="--", label=f"chosen degree: {best_18}")
                ax.set(xlabel="degree", ylabel="R²", ylim=(-0.1, 0.5), title="Choosing the degree on the validation set")
                ax.legend(fontsize=8)
                plt.show()
            else:
                print(f"❌ Ex 8.18 : val_scores doit contenir un R² par degré ({len(degrees_18)} valeurs).")'''

DATA_19 = r'''sunspots = wb.datasets.load_sunspots()                       # monthly sunspot numbers since 1749 (ch. 1 and 5)
smooth_19 = sunspots["sunspots"].rolling(12).mean()            # mean of the last 12 months: the past only
known_19 = smooth_19.notna().to_numpy()
series_19, years_all_19 = smooth_19.to_numpy()[known_19], sunspots["year"].to_numpy()[known_19]
LAGS_19, HORIZON_19 = 24, 12
n_19 = len(series_19) - LAGS_19 - HORIZON_19 + 1
X_19 = np.column_stack([series_19[i:i + n_19] for i in range(LAGS_19)])   # the last 24 known values
y_19 = series_19[LAGS_19 + HORIZON_19 - 1:]                               # the value 12 months later
years_19 = years_all_19[LAGS_19 - 1:LAGS_19 - 1 + n_19]                   # the year of the last known value
print(f"{len(X_19)} samples, {X_19.shape[1]} features, years {years_19[0]} to {years_19[-1]}")'''

TODO_19 = r'''cv_shuffle_19 = ...   # a) a k-fold with 5 folds, shuffled once (random_state=0)
groups_19 = ...       # b) the decade of each sample (1750, 1760...), computed from years_19
cv_groups_19 = ...    # c) 5 folds that keep each decade whole
cv_time_19 = ...      # d) 5 chronological splits: the training part always before the validation part'''

SOLUTION_19 = r'''cv_shuffle_19 = KFold(n_splits=5, shuffle=True, random_state=0)
groups_19 = years_19 // 10 * 10
cv_groups_19 = GroupKFold(n_splits=5)
cv_time_19 = TimeSeriesSplit(n_splits=5)

'''

CHECK_19 = r'''if not filled(cv_shuffle_19, groups_19, cv_groups_19, cv_time_19):
    print("⏳ Ex 8.19 : pas encore fait.")
else:
    try:
        table_19 = {}
        for model_name_19, model_19 in [("1-NN", KNeighborsRegressor(n_neighbors=1)), ("linear", LinearRegression())]:
            for cv_name_19, cv_19, groups_used_19 in [("shuffled k-fold", cv_shuffle_19, None),
                                                      ("decades (groups)", cv_groups_19, groups_19),
                                                      ("chronological", cv_time_19, None)]:
                table_19[(model_name_19, cv_name_19)] = sklearn_cross_val_score(model_19, X_19, y_19, cv=cv_19,
                                                                                groups=groups_used_19).mean()
    except Exception as error_19:   # noqa: BLE001 - a wrong splitter or groups of the wrong length
        print(f"❌ Ex 8.19 : la validation croisée a échoué ({type(error_19).__name__}: {error_19}). "
              "Vérifie tes découpeurs et la longueur de groups_19 (un groupe par échantillon).")
    else:
        print(pd.Series(table_19).unstack().round(3))
        time_ok_19 = all(np.max(train) < np.min(val) for train, val in cv_time_19.split(X_19))
        verdict("8.19", time_ok_19, "chronologique : à chaque tour, l'entraînement est avant la validation.",
                "cv_time_19 met des échantillons postérieurs à la validation dans l'entraînement.")
        decades_19 = np.asarray(groups_19)
        groups_ok_19 = len(decades_19) == len(X_19) and all(
            not set(decades_19[train]) & set(decades_19[val]) for train, val in cv_groups_19.split(X_19, y_19, decades_19))
        verdict("8.19", groups_ok_19, "par groupes : aucune décennie n'est des deux côtés.",
                "une décennie se retrouve des deux côtés : groups_19 doit donner la décennie de chaque échantillon, "
                "et cv_groups_19 garder chaque groupe entier.")
        gap_nn_19 = table_19[("1-NN", "shuffled k-fold")] - table_19[("1-NN", "decades (groups)")]
        verdict("8.19", gap_nn_19 > 0.1,
                f"le 1-NN perd {fr(gap_nn_19, 2)} de R² quand les décennies restent entières : le mélange le flattait.",
                "attendu un R² plus haut pour le 1-NN avec la k-fold mélangée qu'avec les décennies : vérifie tes découpeurs.")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(years_19 + 0.5, y_19, color="gray", lw=0.8)
        for k_19, (_, val_19) in enumerate(cv_time_19.split(X_19)):
            ax.axvspan(years_19[val_19[0]], years_19[val_19[-1]] + 1, alpha=0.15, color=f"C{k_19}")
        ax.set(xlabel="year", ylabel="sunspots, 12-month mean", title="The 5 validation blocks of the chronological split")
        plt.show()'''

DATA_20 = r'''def split_ok(*arrays, test_size=0.25, shuffle=True, rng=None):
    """A correct hold-out split (without stratify): test_size float -> ceil(test_size * n) test rows."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    n_test = math.ceil(test_size * n) if isinstance(test_size, float) else int(test_size)
    if rng is None:
        rng = np.random.default_rng()
    if shuffle:
        order = rng.permutation(n)
        test_idx, train_idx = order[:n_test], order[n_test:]
    else:
        test_idx, train_idx = np.arange(n - n_test, n), np.arange(n - n_test)
    result = []
    for a in arrays:
        result += [a[train_idx], a[test_idx]]
    return result


def split_bug_1(*arrays, test_size=0.25, shuffle=True, rng=None):
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    n_test = int(test_size * n) if isinstance(test_size, float) else int(test_size)
    if rng is None:
        rng = np.random.default_rng()
    if shuffle:
        order = rng.permutation(n)
        test_idx, train_idx = order[:n_test], order[n_test:]
    else:
        test_idx, train_idx = np.arange(n - n_test, n), np.arange(n - n_test)
    result = []
    for a in arrays:
        result += [a[train_idx], a[test_idx]]
    return result


def split_bug_2(*arrays, test_size=0.25, shuffle=True, rng=None):
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    n_test = math.ceil(test_size * n) if isinstance(test_size, float) else int(test_size)
    if rng is None:
        rng = np.random.default_rng()
    result = []
    for a in arrays:
        if shuffle:
            order = rng.permutation(n)
            test_idx, train_idx = order[:n_test], order[n_test:]
        else:
            test_idx, train_idx = np.arange(n - n_test, n), np.arange(n - n_test)
        result += [a[train_idx], a[test_idx]]
    return result


def split_bug_3(*arrays, test_size=0.25, shuffle=True, rng=None):
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    n_test = math.ceil(test_size * n) if isinstance(test_size, float) else int(test_size)
    if rng is None:
        rng = np.random.default_rng()
    if shuffle:
        order = rng.permutation(n)
        test_idx, train_idx = order[:n_test], order[n_test - 1:]
    else:
        test_idx, train_idx = np.arange(n - n_test, n), np.arange(n - n_test)
    result = []
    for a in arrays:
        result += [a[train_idx], a[test_idx]]
    return result


def split_bug_4(*arrays, test_size=0.25, shuffle=True, rng=None):
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    n_test = math.ceil(test_size * n) if isinstance(test_size, float) else int(test_size)
    rng = np.random.default_rng()
    if shuffle:
        order = rng.permutation(n)
        test_idx, train_idx = order[:n_test], order[n_test:]
    else:
        test_idx, train_idx = np.arange(n - n_test, n), np.arange(n - n_test)
    result = []
    for a in arrays:
        result += [a[train_idx], a[test_idx]]
    return result


def split_bug_5(*arrays, test_size=0.25, shuffle=True, rng=None):
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    n_test = math.ceil(test_size * n) if isinstance(test_size, float) else int(test_size)
    if rng is None:
        rng = np.random.default_rng()
    if shuffle:
        order = rng.permutation(n)
        test_idx, train_idx = order[:n_test], order[n_test:]
    else:
        test_idx, train_idx = np.arange(n_test), np.arange(n_test, n)
    result = []
    for a in arrays:
        result += [a[train_idx], a[test_idx]]
    return result


BUGS_20 = [split_bug_1, split_bug_2, split_bug_3, split_bug_4, split_bug_5]'''

TODO_20 = r'''# Write your tests here: each one calls train_test_split(...) and checks one property with assert.
# Then list them in TESTS_20, e.g. TESTS_20 = [test_one_property, test_another_property].

TESTS_20 = ...'''

SOLUTION_20 = r'''def test_sizes():
    """ceil(test_size * n) test rows, also when test_size * n is not a whole number."""
    for n, test_size, n_test in [(10, 0.25, 3), (8, 0.25, 2), (10, 3, 3)]:
        train, test = train_test_split(np.arange(n), test_size=test_size, rng=np.random.default_rng(0))
        assert len(test) == n_test and len(train) == n - n_test


def test_parts_are_disjoint_and_cover_everything():
    train, test = train_test_split(np.arange(50), test_size=0.3, rng=np.random.default_rng(1))
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(50))


def test_rows_stay_aligned():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20) * 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, rng=np.random.default_rng(2))
    assert np.array_equal(X_train[:, 0] // 2 * 10, y_train) and np.array_equal(X_test[:, 0] // 2 * 10, y_test)


def test_same_seed_same_split():
    first = train_test_split(np.arange(30), rng=np.random.default_rng(3))
    second = train_test_split(np.arange(30), rng=np.random.default_rng(3))
    assert np.array_equal(first[1], second[1])


def test_last_rows_without_shuffle():
    train, test = train_test_split(np.arange(10), test_size=0.3, shuffle=False)
    assert test.tolist() == [7, 8, 9] and train.tolist() == list(range(7))


TESTS_20 = [test_sizes, test_parts_are_disjoint_and_cover_everything, test_rows_stay_aligned,
            test_same_seed_same_split, test_last_rows_without_shuffle]

'''

CHECK_20 = r'''with wb.attempt("8.20"):
    if not filled(TESTS_20) or not list(TESTS_20):
        raise NotImplementedError
    tests_20 = list(TESTS_20)
    names_20 = [getattr(test, "__name__", repr(test)) for test in tests_20]
    good_20 = None
    if not all(callable(test) for test in tests_20):
        print("❌ Ex 8.20 : TESTS_20 doit être une liste de fonctions de test, pas de leurs noms ni de leurs résultats.")
    elif not all(name.startswith("test_") for name in names_20):
        print(f"❌ Ex 8.20 : pytest ne lance que les fonctions dont le nom commence par test_ : renomme "
              f"{[name for name in names_20 if not name.startswith('test_')]}.")
    else:
        good_20 = wb.run_pytest(tests_20, subject=split_ok, name="train_test_split", quiet=True)
        if "NameError" in good_20.output:
            print("❌ Ex 8.20 : un de tes tests utilise un nom que le fichier de test ne connaît pas (il n'importe que "
                  "math, numpy sous le nom np, et pytest) : importe ce qu'il te faut dans le test lui-même.")
        verdict("8.20", good_20.ok, f"tes {good_20.passed} tests passent sur la version juste (split_ok).",
                f"un de tes tests échoue sur la version juste : ce test-là est faux ({good_20.summary}).")
    if good_20 is None:
        pass
    elif not good_20.ok:
        print(good_20.output[-1500:])
    else:
        for bug_20 in BUGS_20:
            result_20 = wb.run_pytest(tests_20, subject=bug_20, name="train_test_split", quiet=True)
            verdict("8.20", not result_20.ok, f"{bug_20.__name__} attrapée ({result_20.failed} test(s) en échec).",
                    f"{bug_20.__name__} passe tous tes tests : compare son code à split_ok, et ajoute le test "
                    "d'une propriété de la docstring qu'elle viole.")'''

PART_B = Part("B", "Choisir un hyperparamètre sur un jeu de validation",
              "Fil rouge : le prix des logements de Californie (recensement de 1990), qui remplace dans le "
              "workbook le dataset de Boston retiré de scikit-learn. On garde les districts dont le revenu médian "
              "est entre 1 et 8 (dizaines de milliers de dollars) et dont la valeur médiane n'est pas plafonnée "
              "à 5 (centaines de milliers de dollars), puis on en tire 400 au hasard : `X_cal` (le revenu, un "
              "tableau de forme (400, 1)) et `y_cal` (la valeur des logements). L'hyperparamètre à régler est le "
              "**degré** d'un polynôme ; le ch. 9 y reviendra en détail.",
              given=PART_B_GIVEN, exercises=[
    Ex("8.16", "🔨", 2, 25, "Un estimateur maison à la scikit-learn : PolyFit(degree)",
       "écrire un régresseur qui respecte les conventions de scikit-learn : hyperparamètres rangés par __init__, "
       "fit qui renvoie self, predict, et score qui renvoie le R².",
       "0A (classes) · 0B (polynômes) · fiche §8.4 (encadré 🧮 sur le $R^2$)", thread="California", tracks="R, C",
       body=r"""Écris, dans la cellule ci-dessous, la classe `PolyFit` : une régression polynomiale de `y` sur la seule feature `X[:, 0]`.
- `__init__(self, degree=1)` : range l'hyperparamètre, **et rien d'autre** (`self.degree = degree` : ni calcul, ni vérification). C'est la convention qui permettra à `clone` (8.22) de fabriquer un `PolyFit` neuf avec le même degré ;
- `fit(self, X, y)` : `x = np.asarray(X, dtype=float)[:, 0]`, puis `self.coef_ = np.polyfit(x, y, self.degree)` (les moindres carrés, du plus haut degré au plus bas) ; renvoie `self` ;
- `predict(self, X)` : `np.polyval(self.coef_, x)`, une prédiction par ligne de `X` ;
- `score(self, X, y)` : le $R^2$ des prédictions (encadré 🧮 de la fiche, §8.4), un `float`.

La vérification :
a) entraîne `PolyFit(degree=2)` sur quatre points d'un polynôme de degré 2 et vérifie la prédiction en $x = 4$ ;
b) entraîne `PolyFit(degree=2)` sur les 300 premiers districts de `X_cal` et vérifie son $R^2$ sur les 100 autres ;
puis elle contrôle les conventions (`vars(PolyFit(degree=3))`, `fit` qui renvoie le modèle), compare ton $R^2$ à celui de scikit-learn (`PolynomialFeatures` puis `LinearRegression`) et dessine trois polynômes.

Dans tes notes : que fait le polynôme de degré 8 aux bords des données ? Pourquoi `__init__` ne doit-il pas calculer quoi que ce soit ?""",
       todo=TODO_16, check=CHECK_16,
       solution=SOLUTION_16 + solved(CHECK_16, "8.16"),
       record=r'''wb.record("8.16a", float(toy_16.predict(np.array([[4.0]]))[0]), decimals=4)
wb.record("8.16b", r2_16, decimals=4, mistakes={"c'est le R² sur les 300 districts d'entraînement : la question demande les 100 autres": model_16.score(X_cal[:300], y_cal[:300])})''',
       note="Les quatre points vérifient $y = x^2 + x + 1$ : le polynôme appris passe par eux, et prédit 21 en "
            "$x = 4$. Sur les districts, le degré 2 obtient un $R^2$ de 0,418 sur les 100 districts qu'il n'a pas "
            "vus. Le degré 8 ondule aux bords, là où les districts sont rares : un polynôme de haut degré "
            "extrapole très mal. Si `__init__` calculait quelque chose (une normalisation, une vérification qui "
            "modifie le degré), `clone` reconstruirait un objet différent de l'original ; et un `__init__` qui "
            "entraîne déjà le modèle n'aurait plus rien à faire dans `fit`."),

    Ex("8.17", "🔮", 2, 15, "Validation ou test : lequel sera le plus optimiste ?",
       "prévoir, sur des centaines de répétitions, l'écart entre le score de validation du réglage retenu et son "
       "score de test.",
       "Ex 8.16 · ∂ 8.6 · fiche §8.4", thread="California", tracks="C", hypothesis=True,
       body=r"""L'expérience répète 300 fois la même procédure : tirer 360 districts au hasard ; entraîner `PolyFit` de degré 1 à 12 sur 40 d'entre eux ; noter chaque degré sur 20 districts de validation ; garder le degré qui a le meilleur $R^2$ de validation ; noter **ce** degré sur 300 districts de test. Elle note aussi, à chaque répétition, le degré 1, choisi **d'avance**, sur les mêmes districts. Elle utilise **ta** classe `PolyFit` (8.16).

Prédis, **avant** d'exécuter l'expérience (`True` ou `False`) :
a) `chosen_17` : le $R^2$ de validation du degré retenu dépassera son $R^2$ de test dans **plus de la moitié** des 300 répétitions ;
b) `fixed_17` : pour le degré 1, choisi d'avance, la part des répétitions où le $R^2$ de validation dépasse le $R^2$ de test sera comprise **entre 40 % et 60 %** ;
c) `helps_17` : le $R^2$ de test **médian** du degré retenu sera plus haut que celui du degré 1.

Puis exécute l'expérience. Dans tes notes : explique a) et b) avec ∂ 8.6. Pourquoi choisir le degré sur 20 districts de validation peut-il donner, en test, un moins bon modèle que de fixer le degré 1 d'avance ? Que faudrait-il changer dans la procédure ?""",
       given=DATA_17,
       todo=r'''chosen_17 = ...   # a) True or False
fixed_17 = ...    # b) True or False
helps_17 = ...    # c) True or False''',
       check=r'''for letter_17, answer_17 in zip("abc", [chosen_17, fixed_17, helps_17]):
    wb.check(f"8.17{letter_17}", answer_17)''',
       solution=r'''chosen_17, fixed_17, helps_17 = True, True, False   # the answers, for the record''',
       record=r'''wb.record("8.17a", chosen_17, mistakes={"le degré retenu a été choisi PARCE QUE son score de validation était le plus haut (∂ 8.6)": False})
wb.record("8.17b", fixed_17, mistakes={"un degré fixé d'avance n'a profité d'aucune sélection : sa validation et son test sont deux mesures honnêtes": False})
wb.record("8.17c", helps_17, mistakes={"sur 20 districts, le meilleur score de validation est surtout le plus chanceux : le degré retenu est-il vraiment meilleur ?": True})''',
       after=[("code", guarded(EXPERIMENT_17, ["chosen_17", "fixed_17", "helps_17"],
                               "⏳ Ex 8.17 : écris d'abord tes trois prédictions, puis relance cette cellule."))],
       note="Le $R^2$ de validation du degré retenu dépasse son $R^2$ de test dans 64 % des répétitions, contre "
            "49 % pour le degré 1 : choisir le maximum de douze scores bruités retient en partie le plus "
            "chanceux (∂ 8.6), et ce bonus disparaît sur le test. Pire, le $R^2$ de test médian du degré retenu "
            "(0,336) est **plus bas** que celui du degré 1 (0,368) : avec 20 districts de validation, la sélection "
            "choisit souvent un degré élevé qui a eu de la chance, et qui extrapole mal. Il faudrait plus de "
            "données de validation, une validation croisée, ou peu de candidats (un degré simple choisi d'avance). "
            "Sur **une** répétition, le test peut tout de même battre la validation : c'est une affaire de "
            "moyenne, pas une loi pour chaque découpage."),

    Ex("8.18", "🔨", 2, 30, "Boucle de sélection sur un jeu de validation : le degré du polynôme",
       "programmer la boucle de la figure 8.9 : choisir un hyperparamètre sur la validation, réentraîner, tester "
       "une seule fois.",
       "Ex 8.16 · Ex 8.13 · fiche §8.4", thread="California", tracks="R, C",
       body=r"""Écris `select_degree_18(X, y, degrees)`, la boucle de recherche d'hyperparamètres du livre (figure 8.9), avec **ta** fonction `mylearn.model_selection.train_test_split` (8.13) et **ta** classe `PolyFit` (8.16). Procédure imposée, pour que tout le monde trouve les mêmes nombres (les 400 districts sont déjà dans un ordre aléatoire, d'où `shuffle=False`) :
1. `X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)` : le test, mis de côté ;
2. `X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.25, shuffle=False)` : soit 60 / 20 / 20 ;
3. pour chaque degré de `degrees`, un `PolyFit` **neuf** entraîné sur l'entraînement et noté ($R^2$) sur la validation ;
4. `best_degree` : le degré du meilleur $R^2$ de validation (le premier en cas d'égalité, comme `np.argmax`) ;
5. un `PolyFit(best_degree)` réentraîné sur entraînement + validation (`X_rest`, `y_rest`), puis noté **une fois** sur le test.

Renvoie `(val_scores, best_degree, test_score)`. La vérification lance ta fonction pour les degrés 1 à 8 et vérifie :
a) le degré retenu ;
b) le $R^2$ de test ;
puis elle trace les $R^2$ d'entraînement et de validation de chaque degré.

Dans tes notes : comment évolue le $R^2$ d'entraînement quand le degré augmente ? Et celui de validation ? Pourquoi réentraîner sur entraînement + validation avant le test ? Compare le $R^2$ de test au $R^2$ de validation du degré retenu : qu'en penses-tu, après 8.17 ?""",
       todo=TODO_18, check=RELOAD + CHECK_18,
       solution=SOLUTION_18 + solved(CHECK_18, "8.18"),
       record=r'''trained_on_240_18 = fitted(PolyFit(degree=best_18), X_cal[:240], y_cal[:240]).score(X_cal[320:], y_cal[320:])
wb.record("8.18a", best_18, mistakes={"c'est le degré qui a le meilleur R² d'entraînement : le choix se fait sur la validation": 8})
wb.record("8.18b", test_18, decimals=4, mistakes={"tu notes le modèle entraîné sur 240 districts : réentraîne le degré retenu sur entraînement + validation (320 districts)": trained_on_240_18})''',
       note="Le $R^2$ d'entraînement monte avec le degré (de 0,405 à 0,415) : un polynôme de plus haut degré peut "
            "toujours faire au moins aussi bien sur ses propres données. Le $R^2$ de validation, lui, reste entre 0,34 et "
            "0,36 jusqu'au degré 4 (le meilleur, 0,356, au degré 3), puis baisse, jusqu'à −0,056 au degré 7 : le "
            "polynôme extrapole mal sur les districts de validation situés hors de la zone qu'il a vue. Le degré 3 est retenu, et son "
            "$R^2$ de test, après réentraînement sur 320 districts, vaut 0,421. Ici, le test fait **mieux** que la "
            "validation (0,356) : sur un seul découpage, c'est possible (8.17 montre que c'est l'inverse en "
            "moyenne). Note aussi que les degrés 1 à 4 ont des $R^2$ de validation très proches (de 0,338 à "
            "0,356) : le choix entre eux tient en partie au hasard, et un modèle simple est un choix raisonnable."),

    Ex("8.19", "📦", 2, 25, "Données dépendantes : GroupKFold et TimeSeriesSplit",
       "choisir le découpeur de scikit-learn adapté à une série temporelle, et mesurer ce que coûte un mélange "
       "qui ignore le temps.",
       "Ex 8.14 · ch. 1 et 5 (taches solaires) · fiche §8.5 (données dépendantes)", thread="taches solaires",
       tracks="C",
       body=r"""On prédit la moyenne des taches solaires sur les 12 mois à venir à partir des 24 dernières valeurs connues de la moyenne glissante sur 12 mois, une par mois (`X_19`, 24 features ; `y_19`, la cible ; `years_19`, l'année de la dernière valeur connue). Deux modèles de scikit-learn : le plus proche voisin (`KNeighborsRegressor(n_neighbors=1)`, qui recopie la cible de l'exemple d'entraînement le plus proche, comme `OneNN`) et une régression linéaire (`LinearRegression`).

Définis trois découpeurs de `sklearn.model_selection` :
a) `cv_shuffle_19` : une k-fold à 5 folds, mélangée une fois (`random_state=0`) ;
b) `groups_19` : la décennie de chaque échantillon (1750, 1760…), calculée à partir de `years_19` ;
c) `cv_groups_19` : 5 folds qui gardent chaque décennie entière (`GroupKFold`) ;
d) `cv_time_19` : 5 découpages chronologiques (`TimeSeriesSplit`).

La vérification calcule, avec `cross_val_score` de scikit-learn, le $R^2$ moyen de chaque modèle pour chaque découpeur (`groups=groups_19` pour le deuxième), vérifie les propriétés des découpages, puis colore les 5 blocs de validation chronologiques.

Dans tes notes : pourquoi le plus proche voisin est-il si bon avec la k-fold mélangée, et la régression linéaire presque pas ? Quel découpeur ressemble le plus à l'usage réel du modèle ? Les fenêtres de 24 mois d'un échantillon et de ses voisins se chevauchent : que reste-t-il de la fuite aux frontières des décennies, et à quoi sert le paramètre `gap` de `TimeSeriesSplit` ?""",
       given=DATA_19, todo=TODO_19, check=CHECK_19,
       solution=SOLUTION_19 + CHECK_19,
       note="Avec la k-fold mélangée, chaque mois de validation a ses voisins dans l'entraînement : leurs 24 "
            "valeurs passées sont presque les mêmes, et leurs cibles aussi. Le plus proche voisin recopie la cible "
            "du mois voisin et obtient un $R^2$ de 0,933 ; en gardant les décennies entières, il tombe à 0,782, et "
            "à 0,670 en découpage chronologique, le plus proche de l'usage réel (prédire le futur à partir du "
            "passé, avec moins d'historique dans les premiers tours). La régression linéaire, qui ne retient pas "
            "les exemples, bouge à peine (0,857, 0,851, 0,833) : la fuite profite surtout aux modèles qui "
            "mémorisent. Aux frontières des décennies, les premiers échantillons d'une décennie de validation ont "
            "encore leurs voisins de la décennie précédente dans l'entraînement ; `gap` retire des échantillons "
            "entre l'entraînement et la validation pour couper ce lien."),

    Ex("8.20", "🛠️", 2, 25, "Écrire tes propres tests pytest pour train_test_split",
       "écrire des tests unitaires qui vérifient les propriétés d'un découpage, et les éprouver sur des versions "
       "boguées.",
       "Ex 8.13 · 0A (pytest, assert) · fiche §8.3", tracks="C",
       body=r"""Une collègue a écrit cinq versions de `train_test_split` (sans `stratify`) : `split_ok` est juste, `split_bug_1` à `split_bug_5` contiennent chacune un bug. Écris des **tests pytest** qui passent sur la bonne version et en attrapent le plus possible, **sans lire le code des versions boguées** dans un premier temps : pars des propriétés que promet la docstring de `train_test_split` (8.13).

Chaque test est une fonction dont le nom commence par `test_`, qui appelle `train_test_split(...)` et vérifie **une** propriété avec `assert`. Tes tests sont recopiés dans un fichier à part, qui n'importe que `math`, `numpy` (sous le nom `np`) et `pytest` : importe dans le test lui-même ce dont tu as besoin d'autre. Idées de propriétés : la taille du test ($\lceil t \cdot n \rceil$, avec au moins un cas où $t \cdot n$ n'est pas entier) ; des parties disjointes dont la réunion redonne toutes les lignes ; des tableaux qui restent alignés ; la même graine qui donne le même découpage ; les dernières lignes en test avec `shuffle=False`. Range tes tests dans la liste `TESTS_20`.

La vérification lance **vraiment** pytest (`wb.run_pytest`) : d'abord sur `split_ok`, renommée `train_test_split` (tous tes tests doivent passer), puis sur chaque version boguée (au moins un test doit échouer). Quand une version passe tous tes tests, lis son code, trouve le bug, et ajoute le test qui l'attrape.

Dans tes notes : quel bug était le plus difficile à attraper, et pourquoi ? Un test qui passe sur la version juste et sur toutes les versions boguées sert-il à quelque chose ?""",
       given=DATA_20, todo=TODO_20, check=CHECK_20,
       solution=SOLUTION_20 + solved(CHECK_20, "8.20"),
       note="`split_bug_1` arrondit la taille du test vers le bas (attrapée par $n = 10$, $t = 0{,}25$) ; "
            "`split_bug_2` mélange chaque tableau avec sa propre permutation (attrapée par l'alignement) ; "
            "`split_bug_3` met une ligne des deux côtés (attrapée par « disjoint et complet ») ; `split_bug_4` "
            "ignore le générateur reçu (attrapée par « même graine, même découpage ») ; `split_bug_5` met les "
            "**premières** lignes en test quand `shuffle=False`. Un test qui ne casse sur aucune version boguée "
            "peut rester utile (il protège contre un bug qu'on n'a pas encore imaginé), mais il ne prouve rien "
            "ici : c'est l'idée du *mutation testing*."),
])

# ---------------------------------------------------------------------------
# Part C: cross-validation in mylearn (8.21 to 8.23)
# ---------------------------------------------------------------------------
CHECK_21 = r'''with wb.attempt("8.21"):
    splits_21 = mylearn.model_selection.stratified_kfold_indices([0, 0, 0, 0, 1, 1], n_splits=2)
    if split_pairs("8.21", "stratified_kfold_indices", splits_21, 2):
        print("docstring example:", [(np.asarray(a).tolist(), np.asarray(b).tolist()) for a, b in splits_21])
    splits_21 = mylearn.model_selection.stratified_kfold_indices(species, n_splits=5)
    if split_pairs("8.21", "stratified_kfold_indices", splits_21, 5):
        names_21 = ["Adelie", "Chinstrap", "Gentoo"]
        table_21 = np.array([[int(np.sum(species[np.asarray(val)] == name)) for name in names_21]
                             for _, val in splits_21])
        print(pd.DataFrame(table_21, columns=names_21, index=[f"fold {i + 1}" for i in range(len(table_21))]))
        wb.check("8.21", table_21, computed=True)
        oracle_21 = [sorted(int(np.sum(species[val] == name)) for _, val in StratifiedKFold(5).split(X_peng, species))
                     for name in names_21]
        verdict("8.21", [sorted(column) for column in table_21.T.tolist()] == oracle_21,
                "pour chaque espèce, les mêmes effectifs par fold que StratifiedKFold de scikit-learn (à l'ordre des folds près).",
                "tes effectifs par espèce diffèrent de ceux de StratifiedKFold : relis la règle de la docstring.")
        order_21 = np.argsort(species, kind="stable")              # the rule of the docstring, with NumPy
        verdict("8.21", all(np.array_equal(np.sort(np.asarray(val)), np.sort(order_21[i::5]))
                            for i, (_, val) in enumerate(splits_21)),
                "chaque fold contient exactement les manchots prévus par la docstring (tri stable, puis une carte par fold).",
                "les manchots de tes folds ne sont pas ceux de la règle de la docstring : trie les indices par label avec "
                "un tri stable (np.argsort(..., kind=\"stable\")), puis donne au fold i les positions i, i + k, i + 2k…")
    run_mylearn_tests("test_stratified_kfold_indices_")'''

CHECK_22 = r'''with wb.attempt("8.22"):
    model_22 = NearestCentroid(shrink_threshold=0.5)
    copy_22 = mylearn.model_selection.clone(model_22)
    if returned("8.22", "clone", copy_22):
        verdict("8.22", copy_22 is not model_22 and type(copy_22) is NearestCentroid
                and copy_22.get_params() == model_22.get_params(),
                "clone fabrique un nouveau NearestCentroid avec les mêmes hyperparamètres.",
                "clone doit renvoyer un NOUVEL objet de la même classe, avec les mêmes hyperparamètres.")
        copy_trained_22 = mylearn.model_selection.clone(NearestCentroid().fit(X_peng, species))
        verdict("8.22", copy_trained_22 is not None and not hasattr(copy_trained_22, "centroids_"),
                "le clone d'un modèle entraîné n'a rien appris (pas de centroids_).",
                "le clone d'un modèle entraîné doit être un nouveau modèle, sans aucun attribut appris (ceux qui finissent par _).")
    A_22 = X_peng[:, :2]                                   # bill length and depth, raw
    scores_22 = mylearn.model_selection.cross_val_score(NearestCentroid(), A_22, species, cv=5)
    if returned("8.22", "cross_val_score", scores_22):
        print("a) cv=5, folds in the order of the file (sorted by species):", np.round(np.asarray(scores_22, dtype=float), 3))
        wb.check("8.22a", scores_22, computed=True)
        try:
            folds_22 = mylearn.model_selection.stratified_kfold_indices(species, n_splits=5)
        except NotImplementedError:
            folds_22 = None
            print("⏳ Ex 8.22 : b) a besoin de stratified_kfold_indices (8.21) : reviens ici quand tu l'auras écrite.")
        order_22 = np.argsort(species, kind="stable")      # the rule of the docstring of 8.21, with NumPy
        if folds_22 is None or not split_pairs("8.22", "stratified_kfold_indices (8.21)", folds_22, 5):
            pass
        elif not all(np.array_equal(np.sort(np.asarray(val)), np.sort(order_22[i::5])) for i, (_, val) in enumerate(folds_22)):
            print("❌ Ex 8.22 : b) utilise tes folds de 8.21, qui ne suivent pas encore la règle de la docstring de "
                  "stratified_kfold_indices : corrige d'abord 8.21 (ses tests disent ce qui ne va pas).")
        else:
            stratified_22 = mylearn.model_selection.cross_val_score(NearestCentroid(), A_22, species, cv=folds_22)
            print("b) stratified folds:", np.round(np.asarray(stratified_22, dtype=float), 3),
                  "· mean", round(float(np.mean(stratified_22)), 4))
            wb.check("8.22b", float(np.mean(stratified_22)), computed=True)
            oracle_22 = sklearn_cross_val_score(NearestCentroid(), A_22, species,
                                                cv=[(np.asarray(a), np.asarray(b)) for a, b in folds_22])
            verdict("8.22", np.allclose(np.asarray(stratified_22, dtype=float), oracle_22),
                    "les mêmes scores que cross_val_score de scikit-learn, sur les mêmes folds.",
                    "tes scores diffèrent de ceux de scikit-learn sur les mêmes folds : relis la docstring.")
        print("scikit-learn's cross_val_score with cv=5 (StratifiedKFold for a classifier):",
              np.round(sklearn_cross_val_score(NearestCentroid(), A_22, species, cv=5), 3))
    run_mylearn_tests("test_clone_ or test_cross_val_score_")'''

TODO_23 = r'''def holdout_scores_23(X, y, n_repeats):
    """Test accuracy of NearestCentroid on n_repeats random hold-out splits (test_size=0.25, seeds 0, 1, ...)."""
    raise NotImplementedError


def cv_means_23(X, y, n_repeats):
    """Mean accuracy of a shuffled 5-fold cross-validation of NearestCentroid, repeated n_repeats times (seeds 0, 1, ...)."""
    raise NotImplementedError'''

SOLUTION_23 = r'''def holdout_scores_23(X, y, n_repeats):
    """Test accuracy of NearestCentroid on n_repeats random hold-out splits (test_size=0.25, seeds 0, 1, ...)."""
    scores = []
    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = mylearn.model_selection.train_test_split(
            X, y, test_size=0.25, rng=np.random.default_rng(seed))
        scores.append(NearestCentroid().fit(X_train, y_train).score(X_test, y_test))
    return np.array(scores)


def cv_means_23(X, y, n_repeats):
    """Mean accuracy of a shuffled 5-fold cross-validation of NearestCentroid, repeated n_repeats times (seeds 0, 1, ...)."""
    means = []
    for seed in range(n_repeats):
        folds = mylearn.model_selection.kfold_indices(len(X), n_splits=5, shuffle=True, rng=np.random.default_rng(seed))
        means.append(np.mean(mylearn.model_selection.cross_val_score(NearestCentroid(), X, y, cv=folds)))
    return np.array(means)


'''

CHECK_23 = r'''with wb.attempt("8.23"):
    A_23 = X_peng[:, :2]                                   # bill length and depth, raw (accuracy about 0.88)
    with threadpool_limits(limits=1):                      # 1 200 small fits: one thread is faster
        holdout_23 = holdout_scores_23(A_23, species, 200)
        cv_23 = cv_means_23(A_23, species, 200)
    if returned("8.23", "holdout_scores_23", holdout_23) and returned("8.23", "cv_means_23", cv_23):
        holdout_23, cv_23 = np.asarray(holdout_23, dtype=float), np.asarray(cv_23, dtype=float)
        if holdout_23.shape != (200,) or cv_23.shape != (200,):
            print(f"❌ Ex 8.23 : attendu 200 valeurs par fonction, reçu les formes {holdout_23.shape} et {cv_23.shape}.")
        else:
            for name_23, values_23 in [("hold-out (25 % test)", holdout_23), ("mean of a 5-fold", cv_23)]:
                print(f"{name_23:21s}: mean {values_23.mean():.3f}, std {values_23.std():.3f}, "
                      f"from {values_23.min():.3f} to {values_23.max():.3f}")
            verdict("8.23", abs(holdout_23.mean() - cv_23.mean()) < 0.01,
                    "les deux méthodes visent le même score : leurs moyennes sont proches.",
                    "les moyennes des deux méthodes devraient être proches : relis les deux procédures imposées.")
            verdict("8.23", holdout_23.std() > 3 * cv_23.std(),
                    f"la moyenne d'une 5-fold varie {fr(holdout_23.std() / cv_23.std(), 1)} fois moins d'un tirage à l'autre "
                    "qu'un hold-out.",
                    "attendu une dispersion bien plus faible pour la moyenne des 5-fold que pour le hold-out.")
            fig, ax = plt.subplots(figsize=(7, 3.6))
            bins_23 = np.linspace(0.75, 1.0, 51)
            ax.hist(holdout_23, bins=bins_23, alpha=0.6, label="200 random hold-outs")
            ax.hist(cv_23, bins=bins_23, alpha=0.6, label="200 shuffled 5-fold (mean)")
            ax.set(xlabel="estimated accuracy", ylabel="count", title="The same model, two ways to evaluate it")
            ax.legend(fontsize=8)
            plt.show()'''

PART_C = Part("C", "La validation croisée dans mylearn",
              "Tu complètes `model_selection.py` : la k-fold stratifiée, `clone` et `cross_val_score`, puis tu "
              "mesures ce que la validation croisée apporte par rapport à un simple hold-out. Le modèle est le "
              "centroïde le plus proche de scikit-learn (`NearestCentroid`, le même que ton modèle du ch. 7), "
              "sur la longueur et l'épaisseur du bec des manchots.",
              exercises=[
    Ex("8.21", "🔨", 3, 40, "k-fold stratifiée : stratified_kfold_indices",
       "écrire une k-fold dont chaque fold garde les proportions des classes.",
       "Ex 8.14 · fiche §8.5.1 (stratifier les folds)", thread="Penguins", tracks="R, M, C",
       mylearn="model_selection.py",
       body=MYLEARN_SHORT + r"""

Écris `stratified_kfold_indices(y, n_splits=5, shuffle=False, rng=None)` (lis sa docstring et le mini-exemple de la fiche) :
- lève une `ValueError` si `n_splits < 2` ou si `n_splits` dépasse le nombre d'échantillons ; émets un `UserWarning` (`warnings.warn(..., UserWarning)`) si une classe a moins de `n_splits` membres ;
- trie les indices **par label**, avec un tri **stable** (`np.argsort(y, kind="stable")` : à l'intérieur d'une classe, l'ordre d'origine est gardé) ; avec `shuffle=True`, mélange d'abord les indices **de chaque classe** avec `rng`, puis mets les classes bout à bout dans l'ordre trié des labels ;
- distribue cet ordre comme des cartes : le fold `i` reçoit les positions `i`, `i + k`, `i + 2k`… (`order[i::k]`) ;
- renvoie une liste de `n_splits` paires `(train_idx, val_idx)`, chaque partie triée en ordre croissant.

La vérification affiche l'exemple de la docstring, puis vérifie le tableau des effectifs de chaque espèce dans chacun des 5 folds des 333 manchots (une ligne par fold, colonnes Adélie, Chinstrap, Gentoo), le compare à `StratifiedKFold` de scikit-learn, et lance les tests.

Dans tes notes : compare ce tableau à celui de la k-fold sans mélange de 8.14. Pourquoi les effectifs d'une espèce diffèrent-ils au plus d'une unité d'un fold à l'autre ?""",
       check=RELOAD + CHECK_21,
       solution=solved(CHECK_21, "8.21"),
       record=r'''wb.record("8.21", table_21, mistakes={"tes classes sont rangées dans leur ordre d'apparition dans le fichier : la docstring trie les indices par valeur de label (tri stable)": [[30, 14, 23], [29, 14, 24], [29, 14, 24], [29, 13, 24], [29, 13, 24]]})''',
       note="Le tableau vaut, fold par fold : (30, 13, 24), (29, 14, 24), (29, 14, 24), (29, 14, 23), (29, 13, 24) "
            "manchots Adélie, Chinstrap et Gentoo. Chaque espèce occupe un bloc de positions consécutives de "
            "l'ordre trié, et la distribution « une carte par fold » donne à chaque fold soit $\\lfloor n_c / k "
            "\\rfloor$, soit $\\lfloor n_c / k \\rfloor + 1$ membres de la classe $c$ ; les folds ont 67, 67, 67, "
            "66 et 66 manchots. Sans mélange ni stratification (8.14), deux folds ne contenaient que des Adélie et "
            "le dernier que des Chinstrap. scikit-learn range les classes dans leur ordre d'apparition "
            "(Adélie, Gentoo, Chinstrap) et répartit les exemples d'une classe par blocs : pour chaque espèce, il "
            "obtient les mêmes effectifs par fold, mais pas dans les mêmes folds (son tableau commence par "
            "(30, 14, 23)), et les manchots de chaque fold diffèrent."),

    Ex("8.22", "🔨", 3, 45, "clone et cross_val_score",
       "écrire le clonage d'un estimateur et la validation croisée de mylearn, et les comparer à scikit-learn.",
       "Ex 8.14 · Ex 8.21 · ch. 7 (estimateurs à la scikit-learn) · fiche §8.5.1 (cloner, `cross_val_score`)",
       thread="Penguins", tracks="R, M, C", mylearn="model_selection.py",
       body=MYLEARN_SHORT + r"""

Écris (lis les docstrings) :
- `clone(estimator)` : relis les attributs de l'estimateur (`vars(estimator)`) dont le nom ne commence ni ne finit par `_` (ses hyperparamètres), copie chaque valeur avec `copy.deepcopy`, et renvoie `type(estimator)(**params)`. Ne rattrape pas l'erreur d'un `__init__` qui refuse ces arguments : elle doit remonter (`TypeError`) ;
- `cross_val_score(estimator, X, y, cv=5, scoring=None)` : convertis `X` et `y` avec `np.asarray` (un `DataFrame` indexé par `X[idx]` sélectionnerait des colonnes) ; lève une `ValueError` si leurs longueurs diffèrent, ou si `scoring` vaut `None` et que l'estimateur n'a pas de méthode `score`. Un `cv` entier `k` donne `kfold_indices(len(X), k)`, **sans mélange** (une liste de paires est utilisée telle quelle). Pour chaque paire `(train_idx, val_idx)` : un **clone** de l'estimateur, entraîné sur les lignes d'entraînement, noté sur celles de validation, avec `scoring(fitted_clone, X_val, y_val)` ou `fitted_clone.score(X_val, y_val)`. N'enchaîne pas `.fit(...).score(...)` : appelle `fit`, puis `score`. Renvoie un tableau NumPy de flottants, un score par paire. L'estimateur reçu n'est **jamais** entraîné.

La vérification essaie `clone` sur des modèles de scikit-learn, puis calcule, avec `NearestCentroid` sur la longueur et l'épaisseur du bec :
a) les 5 scores de `cv=5`, les manchots dans l'ordre du fichier (triés par espèce) ;
b) la moyenne des 5 scores avec les folds stratifiés de ta fonction 8.21 ;
et compare ces scores à `cross_val_score` de scikit-learn, avant de lancer les tests.

Dans tes notes : pourquoi les scores de a) sont-ils si irréguliers ? Avec `cv=5`, `cross_val_score` de scikit-learn ne donne pas les mêmes scores que le tien sur les mêmes données : pourquoi (fiche, encadré 🕰️ sur les découpeurs) ? Pourquoi faut-il cloner l'estimateur à chaque tour, au lieu de réutiliser le même objet ?""",
       check=RELOAD + CHECK_22,
       solution=solved(CHECK_22, "8.22"),
       record=r'''wb.record("8.22a", scores_22, decimals=5)
wb.record("8.22b", float(np.mean(stratified_22)), decimals=4, mistakes={"c'est la moyenne des scores de a) : la question demande les folds stratifiés de 8.21": float(np.mean(scores_22))})''',
       note="Dans l'ordre du fichier, les deux premiers folds ne contiennent que des Adélie, et le dernier que des "
            "Chinstrap : les scores (0,970, 0,955, 0,955, 0,636, 0,788) dépendent surtout de l'espèce du fold, et "
            "au dernier tour le modèle n'a appris les Chinstrap que sur deux individus. Avec des folds "
            "stratifiés, les scores sont plus réguliers, autour de 0,877 en moyenne. Avec `cv=5`, scikit-learn "
            "utilise `StratifiedKFold` pour un classifieur (sans mélanger), d'où d'autres folds et d'autres "
            "scores. Réutiliser le même objet ferait dépendre chaque tour des précédents dès que `fit` ne repart "
            "pas de zéro (un réseau qui reprend ses poids, un `warm_start`), et l'objet passé à la fonction "
            "ressortirait modifié : `clone` garantit un modèle neuf à chaque tour."),

    Ex("8.23", "🔬", 3, 40, "Variabilité de l'évaluation : hold-out répétés contre k-fold",
       "mesurer combien une estimation de l'accuracy dépend du découpage, avec un hold-out et avec une "
       "validation croisée.",
       "Ex 8.22 · Ex 8.13 · Ex 8.14 · ch. 2 (dispersion) · fiche §8.5.1", thread="Penguins", tracks="M, C",
       body=r"""Combien une accuracy estimée dépend-elle du hasard du découpage ? Écris, avec **tes** fonctions de `mylearn.model_selection` et `NearestCentroid` de scikit-learn :
- `holdout_scores_23(X, y, n_repeats)` : pour chaque graine `seed` de 0 à `n_repeats - 1`, `train_test_split(X, y, test_size=0.25, rng=np.random.default_rng(seed))`, puis l'accuracy de `NearestCentroid` entraîné sur l'entraînement et noté sur le test ; renvoie les `n_repeats` accuracies ;
- `cv_means_23(X, y, n_repeats)` : pour chaque graine, `kfold_indices(len(X), 5, shuffle=True, rng=np.random.default_rng(seed))`, puis la **moyenne** des 5 scores de `cross_val_score` ; renvoie les `n_repeats` moyennes.

La vérification lance les deux fonctions 200 fois chacune sur la longueur et l'épaisseur du bec (brutes), compare les moyennes et les dispersions, et superpose les deux histogrammes.

Dans tes notes : entre quelles valeurs l'accuracy d'un hold-out peut-elle tomber, selon le tirage ? Pourquoi la moyenne d'une 5-fold varie-t-elle beaucoup moins ? Combien d'entraînements chaque méthode coûte-t-elle ? Cette faible dispersion veut-elle dire que l'accuracy du modèle sur de **nouveaux** manchots est connue à ce point près (fiche, « la dispersion des folds n'est pas une erreur type ») ?""",
       todo=TODO_23, check=RELOAD + CHECK_23,
       solution=SOLUTION_23 + solved(CHECK_23, "8.23"),
       note="Sur 200 tirages, l'accuracy d'un hold-out va d'environ 0,79 à 0,96 (écart-type 0,035) : un seul "
            "découpage peut faire paraître le modèle médiocre ou excellent. La moyenne d'une 5-fold reste entre "
            "0,86 et 0,89 (écart-type 0,005, environ 7,5 fois moins), parce qu'elle note **chaque** manchot une "
            "fois et moyenne cinq modèles : le hasard du découpage s'y compense. Elle coûte 5 entraînements au "
            "lieu d'un. Mais les 200 répétitions utilisent toujours les **mêmes** 333 manchots : la faible "
            "dispersion mesure l'effet du découpage, pas l'incertitude face à de nouveaux manchots, qui reste de "
            "l'ordre de l'erreur type d'une accuracy sur 333 exemples (environ 0,018)."),
])

# ---------------------------------------------------------------------------
# Part D: leaks, honest comparisons and the challenge (8.24 to 8.27)
# ---------------------------------------------------------------------------
DATA_24 = r'''island = penguins["island"].to_numpy()                       # Biscoe, Dream or Torgersen


def spread_score_24(x, labels):
    """How far apart the class means of the feature x are, in standard deviations of x."""
    means = [x[labels == label].mean() for label in np.unique(labels)]
    return (max(means) - min(means)) / x.std()


def select_2(X, labels):
    """The indices (sorted) of the two columns of X with the largest spread_score_24."""
    scores = np.array([spread_score_24(X[:, j], labels) for j in range(X.shape[1])])
    return np.sort(np.argsort(-scores, kind="stable")[:2])


class IslandPipeline:
    """select_2, then standardisation, then the model ("1-NN" or "centroid"): every step is learnt in fit,
    on the rows that fit receives."""

    def __init__(self, model="1-NN"):
        self.model = model

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.columns_ = select_2(X, y)
        A = X[:, self.columns_]
        self.mean_, self.std_ = A.mean(axis=0), A.std(axis=0)
        model = OneNN() if self.model == "1-NN" else NearestCentroid()
        self.model_ = model.fit((A - self.mean_) / self.std_, y)
        return self

    def predict(self, X):
        A = np.asarray(X, dtype=float)[:, self.columns_]
        return self.model_.predict((A - self.mean_) / self.std_)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


def colleague_24():
    """The colleague's notebook: guess the island of a penguin from its measurements."""
    # Step A: the four measurements of the 333 penguins, and their island
    X, y = X_peng.copy(), island.copy()
    # Step B: "data augmentation": every penguin twice more, with a tiny measurement noise
    rng = np.random.default_rng(24)
    X = np.vstack([X] + [X + rng.normal(0.0, 0.01, X.shape) * X.std(axis=0) for _ in range(2)])
    y = np.concatenate([y, y, y])
    # Step C: standardisation, with the mean and the standard deviation of all the rows
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Step D: keep the two measurements whose island means differ the most (select_2 on all the rows)
    X = X[:, select_2(X, y)]
    # Step E: split, 70 % for training and 30 % for the test
    X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.3, random_state=0)
    # Step F: try two models, keep the one with the best test accuracy, and report that accuracy
    scores = {"1-NN": OneNN().fit(X_train, y_train).score(X_test, y_test),
              "centroid": NearestCentroid().fit(X_train, y_train).score(X_test, y_test)}
    best = max(scores, key=scores.get)
    return best, scores[best]


best_24, reported_24 = colleague_24()
print(f"the colleague reports: model {best_24}, test accuracy {reported_24:.3f}")
print("islands of the 333 penguins:", {name: int(count) for name, count in zip(*np.unique(island, return_counts=True))})'''

TODO_24 = r'''leaky_steps_24 = ...   # a) the letters of the steps that leak, e.g. "XY"


def honest_24():
    """The honest version of the colleague's study: returns (model_name, test_accuracy)."""
    raise NotImplementedError'''

SOLUTION_24 = r'''leaky_steps_24 = "BCDF"


def honest_24():
    """The honest version of the colleague's study: returns (model_name, test_accuracy)."""
    X_train, y_train = X_peng[TRAIN_CH1], island[TRAIN_CH1]          # step 2: the split of ch. 1, first
    folds = mylearn.model_selection.stratified_kfold_indices(y_train, n_splits=5)
    cv_means = {name: float(np.mean(mylearn.model_selection.cross_val_score(IslandPipeline(name), X_train, y_train,
                                                                            cv=folds)))
                for name in ("1-NN", "centroid")}                     # step 3: the choice, on the training part only
    print("cross-validation on the 233 training penguins:", {name: round(value, 3) for name, value in cv_means.items()})
    best = max(cv_means, key=cv_means.get)                            # 1-NN first on ties
    final = IslandPipeline(best).fit(X_train, y_train)                # step 4: refit, then the test, once
    return best, final.score(X_peng[TEST_CH1], island[TEST_CH1])


'''

CHECK_24 = r'''wb.check("8.24a", as_letters(leaky_steps_24))
with wb.attempt("8.24"):
    result_24 = honest_24()
    if returned("8.24", "honest_24", result_24):
        if not (isinstance(result_24, (tuple, list)) and len(result_24) == 2):
            print("❌ Ex 8.24 : honest_24 doit renvoyer un couple (model_name, test_accuracy).")
        else:
            print("honest study:", result_24[0], "· test accuracy", round(float(result_24[1]), 3))
            wb.check("8.24b", result_24[0])
            wb.check("8.24c", result_24[1], computed=True)'''

DATA_25 = r'''rng_25 = np.random.default_rng(825)
X_25 = rng_25.normal(size=(60, 1000))                  # 60 samples, 1000 features of pure noise
y_25 = rng_25.permutation(np.repeat([0, 1], 30))       # labels drawn at random: there is nothing to learn
print(X_25.shape, "· labels:", np.bincount(y_25).tolist())'''

TODO_25 = r'''def top_k_25(X, y, k):
    """Indices of the k columns of X whose correlation with y is the largest in absolute value."""
    raise NotImplementedError


def selected_before_25(X, y, k):
    """Select the k features on ALL the rows, then the mean of cross_val_score(NearestCentroid(), ..., cv=5)."""
    raise NotImplementedError


def selected_inside_25(X, y, k):
    """For each split of kfold_indices(len(X), 5): select the k features on the training part only, fit
    NearestCentroid on them, score it on the validation part. Returns the mean of the 5 scores."""
    raise NotImplementedError'''

SOLUTION_25 = r'''def top_k_25(X, y, k):
    """Indices of the k columns of X whose correlation with y is the largest in absolute value."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    r = (Xc * yc[:, None]).sum(axis=0) / np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    return np.argsort(-np.abs(r))[:k]


def selected_before_25(X, y, k):
    """Select the k features on ALL the rows, then the mean of cross_val_score(NearestCentroid(), ..., cv=5)."""
    columns = top_k_25(X, y, k)                                   # the leak: the validation rows take part
    return float(np.mean(mylearn.model_selection.cross_val_score(NearestCentroid(), X[:, columns], y, cv=5)))


def selected_inside_25(X, y, k):
    """For each split of kfold_indices(len(X), 5): select the k features on the training part only, fit
    NearestCentroid on them, score it on the validation part. Returns the mean of the 5 scores."""
    scores = []
    for train_idx, val_idx in mylearn.model_selection.kfold_indices(len(X), 5):
        columns = top_k_25(X[train_idx], y[train_idx], k)
        model = NearestCentroid().fit(X[train_idx][:, columns], y[train_idx])
        scores.append(model.score(X[val_idx][:, columns], y[val_idx]))
    return float(np.mean(scores))


'''

CHECK_25 = r'''with wb.attempt("8.25"):
    before_25 = selected_before_25(X_25, y_25, 20)
    inside_25 = selected_inside_25(X_25, y_25, 20)
    if not (returned("8.25", "selected_before_25", before_25) and returned("8.25", "selected_inside_25", inside_25)):
        pass
    elif not all(isinstance(value, (int, float)) for value in (before_25, inside_25)):
        print(f"❌ Ex 8.25 : les deux fonctions doivent renvoyer une accuracy moyenne (un nombre), reçu "
              f"{type(before_25).__name__} et {type(inside_25).__name__}.")
    else:
        print(f"k = 20: selection before the cross-validation {float(before_25):.3f}, inside {float(inside_25):.3f}")
        wb.check("8.25a", before_25, computed=True)
        wb.check("8.25b", inside_25, computed=True)
        ks_25 = [5, 10, 20, 50, 100]
        curves_25 = [(selected_before_25(X_25, y_25, k), selected_inside_25(X_25, y_25, k)) for k in ks_25]
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(ks_25, [b for b, _ in curves_25], "o-", label="selection on all the rows, then cross-validation")
        ax.plot(ks_25, [i for _, i in curves_25], "s-", label="selection inside each training part")
        ax.axhline(0.5, color="gray", ls="--", label="chance level")
        ax.set(xscale="log", xlabel="number k of selected features", ylabel="cross-validated accuracy", ylim=(0, 1.05),
               title="60 samples, 1000 noise features, random labels")
        ax.legend(fontsize=8)
        plt.show()
        verdict("8.25", all(0.25 <= i <= 0.75 for _, i in curves_25),
                "sélectionnées dans chaque tour, les features ne font pas mieux que le hasard, comme il se doit sur du bruit.",
                "avec la sélection dans chaque tour, l'accuracy devrait rester proche de 0,5 : vérifie que top_k_25 "
                "ne voit que la partie d'entraînement.")'''

DATA_26 = r'''Z_26 = (X_peng - X_peng[TRAIN_CH1].mean(axis=0)) / X_peng[TRAIN_CH1].std(axis=0)   # standardised with the training rows


def correct_26(columns, standardised=True):
    """1 where NearestCentroid, trained on the 233 training penguins of ch. 1, is right on a test penguin, else 0."""
    A = (Z_26 if standardised else X_peng)[:, columns]
    model = NearestCentroid().fit(A[TRAIN_CH1], species[TRAIN_CH1])
    return (model.predict(A[TEST_CH1]) == species[TEST_CH1]).astype(int)


correct_A_26 = correct_26([0, 1])                       # A: bill length and depth, standardised
correct_B_26 = correct_26([0, 1, 2, 3])                 # B: the four measurements, standardised
correct_raw_26 = correct_26([0, 1, 2, 3], standardised=False)   # the four raw measurements (comparison 2)
print("accuracies on the 100 test penguins: A", correct_A_26.mean(), "· B", correct_B_26.mean(),
      "· four raw measurements", correct_raw_26.mean())'''

TODO_26 = r'''discordant_26 = ...   # a) [number of test penguins where only A is right, number where only B is right]


def permutation_pvalue_26(correct_a, correct_b, n_perm=10_000, seed=826):
    """Two-sided p-value of the paired permutation test of the fiche (sign flips of d = correct_b - correct_a)."""
    raise NotImplementedError


def paired_bootstrap_26(correct_a, correct_b, n_boot=10_000, seed=8260):
    """95 % percentile interval [low, high] of the accuracy difference B - A, test penguins resampled with replacement."""
    raise NotImplementedError


significant_26 = ...   # e) True or False: does B beat A at the 5 % level (comparison 1)?'''

SOLUTION_26 = r'''discordant_26 = [int(np.sum((correct_A_26 == 1) & (correct_B_26 == 0))),
                 int(np.sum((correct_A_26 == 0) & (correct_B_26 == 1)))]


def permutation_pvalue_26(correct_a, correct_b, n_perm=10_000, seed=826):
    """Two-sided p-value of the paired permutation test of the fiche (sign flips of d = correct_b - correct_a)."""
    d = np.asarray(correct_b, dtype=int) - np.asarray(correct_a, dtype=int)
    rng = np.random.default_rng(seed)
    flips = rng.random((n_perm, len(d))) < 0.5                 # one draw: True = swap the answers of A and B
    sums = np.where(flips, -d, d).sum(axis=1)                  # integers: no rounding problem in the comparison
    return (1 + int(np.sum(np.abs(sums) >= abs(int(d.sum()))))) / (n_perm + 1)


def paired_bootstrap_26(correct_a, correct_b, n_boot=10_000, seed=8260):
    """95 % percentile interval [low, high] of the accuracy difference B - A, test penguins resampled with replacement."""
    d = np.asarray(correct_b, dtype=float) - np.asarray(correct_a, dtype=float)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(d), size=(n_boot, len(d)))     # one draw: one resample per row
    return np.percentile(d[rows].mean(axis=1), [2.5, 97.5])


significant_26 = False

'''

CHECK_26 = r'''wb.check("8.26a", discordant_26)
with wb.attempt("8.26"):
    p_value_26 = permutation_pvalue_26(correct_A_26, correct_B_26)
    interval_26 = paired_bootstrap_26(correct_A_26, correct_B_26)
    if not (returned("8.26", "permutation_pvalue_26", p_value_26) and returned("8.26", "paired_bootstrap_26", interval_26)):
        pass
    elif not isinstance(p_value_26, (int, float)) or np.shape(interval_26) != (2,):
        print(f"❌ Ex 8.26 : permutation_pvalue_26 doit renvoyer un nombre et paired_bootstrap_26 deux bornes [low, high] ; "
              f"reçu {type(p_value_26).__name__} et un objet de forme {np.shape(interval_26)}.")
    else:
        print(f"comparison 1 (A, then B): p-value {float(p_value_26):.4f}, 95 % interval of B − A:",
              np.round(np.asarray(interval_26, dtype=float), 3).tolist())
        wb.check("8.26b", p_value_26, computed=True)
        wb.check("8.26c", interval_26, computed=True)
        only_a_26 = int(np.sum((correct_A_26 == 1) & (correct_B_26 == 0)))
        only_b_26 = int(np.sum((correct_A_26 == 0) & (correct_B_26 == 1)))
        exact_26 = scipy.stats.binomtest(min(only_a_26, only_b_26), only_a_26 + only_b_26, 0.5).pvalue
        verdict("8.26", abs(float(p_value_26) - exact_26) < 0.02,
                f"proche de la p-valeur exacte du test de McNemar ({fr(exact_26, 4)}).",
                f"la p-valeur exacte du test de McNemar vaut {fr(exact_26, 4)} : la tienne devrait en être proche.")
        p_value_raw_26 = permutation_pvalue_26(correct_raw_26, correct_B_26)
        print(f"comparison 2 (raw measurements, then B): p-value {float(p_value_raw_26):.4f}")
        wb.check("8.26d", p_value_raw_26, computed=True)
wb.check("8.26e", significant_26)'''

DATA_27 = r'''sex = penguins["sex"].to_numpy()                     # "male" or "female"


class PairModel:
    """Two of the four measurements, standardised or not (statistics of the rows given to fit), then
    NearestCentroid ("centroid") or OneNN ("1-NN")."""

    def __init__(self, columns=(0, 1), standardize=True, model="centroid"):
        self.columns = columns
        self.standardize = standardize
        self.model = model

    def fit(self, X, y):
        A = np.asarray(X, dtype=float)[:, list(self.columns)]
        self.mean_ = A.mean(axis=0) if self.standardize else np.zeros(A.shape[1])
        self.std_ = A.std(axis=0) if self.standardize else np.ones(A.shape[1])
        model = NearestCentroid() if self.model == "centroid" else OneNN()
        self.model_ = model.fit((A - self.mean_) / self.std_, y)
        return self

    def predict(self, X):
        A = np.asarray(X, dtype=float)[:, list(self.columns)]
        return self.model_.predict((A - self.mean_) / self.std_)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


TEST_27 = {"first": None}                             # the test of ch. 1 is revealed once per session


def _check_choice_27(choice):
    """(columns, standardize, model) cleaned up, or (None, the reason why the choice is not valid)."""
    if not (isinstance(choice, (tuple, list)) and len(choice) == 3):
        return None, f"choose_27 doit renvoyer un triplet (columns, standardize, model), pas {choice!r}"
    columns, standardize, model = choice
    try:
        columns = tuple(int(c) for c in columns)
    except (TypeError, ValueError):
        return None, f"columns doit être une paire d'indices de colonnes, pas {columns!r}"
    if len(columns) != 2 or len(set(columns)) != 2 or not all(0 <= c < 4 for c in columns):
        return None, f"columns doit contenir deux indices différents parmi 0, 1, 2 et 3, pas {columns}"
    if model not in ("centroid", "1-NN"):
        return None, f'model doit valoir "centroid" ou "1-NN", pas {model!r}'
    return (columns, bool(standardize), model), ""


def grade_27(choose, reveal):
    """(result, reason): choose_27 on the 233 training penguins of ch. 1; with reveal=True, the test accuracy of the
    choice (shown once per session), then the mean test accuracy of choose_27 rerun on 20 other splits."""
    choice, reason = _check_choice_27(choose(X_peng[TRAIN_CH1].copy(), sex[TRAIN_CH1].copy()))
    if choice is None:
        return None, reason
    print("your choice on the 233 training penguins:", choice)
    if not reveal:
        return None, "le test n'est pas encore révélé : mets READY_27 = True quand ton choix est arrêté"
    if TEST_27["first"] is None:
        main = PairModel(*choice).fit(X_peng[TRAIN_CH1], sex[TRAIN_CH1]).score(X_peng[TEST_CH1], sex[TEST_CH1])
        TEST_27["first"] = (choice, main)
    elif TEST_27["first"][0] != choice:
        print("⚠️ le test du ch. 1 a déjà été révélé dans cette session : la note reste celle de ton premier choix,",
              TEST_27["first"][0])
    first_choice, main = TEST_27["first"]
    others = []
    for seed in range(2700, 2720):                    # 20 other splits of the 333 penguins (233 + 100)
        order = np.random.default_rng(seed).permutation(len(sex))
        test, train = order[:100], order[100:]
        other, reason = _check_choice_27(choose(X_peng[train].copy(), sex[train].copy()))
        if other is None:
            return None, reason
        others.append(PairModel(*other).fit(X_peng[train], sex[train]).score(X_peng[test], sex[test]))
    return {"choice": first_choice, "test": main, "others": float(np.mean(others))}, ""'''

TODO_27 = r'''def choose_27(X_train, y_train):
    """Choose (columns, standardize, model) to predict the sex of a penguin, with the training penguins only."""
    candidates = [(columns, standardize, model) for columns in itertools.combinations(range(4), 2)
                  for standardize in (False, True) for model in ("centroid", "1-NN")]
    scores = [fitted(PairModel(*candidate), X_train, y_train).score(X_train, y_train)    # the starting point:
              for candidate in candidates]                                              # the TRAINING accuracy
    return candidates[int(np.argmax(scores))]


READY_27 = False   # set to True when your choice is final: the test is revealed once'''

SOLUTION_27 = r'''def choose_27(X_train, y_train):
    """Choose (columns, standardize, model) to predict the sex of a penguin, with the training penguins only."""
    candidates = [(columns, standardize, model) for columns in itertools.combinations(range(4), 2)
                  for standardize in (False, True) for model in ("centroid", "1-NN")]
    folds = mylearn.model_selection.stratified_kfold_indices(y_train, n_splits=5, shuffle=True,
                                                             rng=np.random.default_rng(27))
    scores = [np.mean(mylearn.model_selection.cross_val_score(PairModel(*candidate), X_train, y_train, cv=folds))
              for candidate in candidates]
    return candidates[int(np.argmax(scores))]


READY_27 = True

'''

CHECK_27 = r'''with wb.attempt("8.27"):
    with threadpool_limits(limits=1):
        result_27, reason_27 = grade_27(choose_27, READY_27)
    if result_27 is None:
        print(f"⏳ Ex 8.27 : {reason_27}." if "révélé" in reason_27 else f"❌ Ex 8.27 : {reason_27}.")
    else:
        print(f"test of ch. 1 (100 penguins): {result_27['test']:.2f} · mean of the 20 other splits: {result_27['others']:.3f}")
        verdict("8.27", result_27["test"] >= 0.90, f"accuracy de {fr(result_27['test'], 2)} sur le test : objectif atteint.",
                f"accuracy de {fr(result_27['test'], 2)} sur le test : il faut au moins 0,90.")
        verdict("8.27", result_27["others"] >= 0.86,
                f"ta méthode tient sur 20 autres découpages (moyenne {fr(result_27['others'], 3)}).",
                f"sur 20 autres découpages, ta méthode n'obtient que {fr(result_27['others'], 3)} en moyenne : il faut "
                "au moins 0,86.")'''

PART_D = Part("D", "Fuites, comparaisons honnêtes et défi",
              "Trois expériences sur les pièges de l'évaluation, puis un défi. Les manchots reviennent, avec le "
              "découpage du ch. 1 (`TRAIN_CH1`, `TEST_CH1`) et deux nouvelles cibles : l'**île** (8.24) et le "
              "**sexe** (8.27).",
              exercises=[
    Ex("8.24", "🐛", 3, 35, "Un notebook trop beau pour être vrai : quatre fuites à corriger",
       "repérer les fuites d'un protocole d'évaluation et le refaire honnêtement.",
       "Ex 8.22 · Ex 8.21 · Ex 8.3 · fiche §8.3 (fuites), §8.5 (« seulement si tout est refait dans la boucle »)",
       thread="Penguins", tracks="R, C",
       body=r"""Une collègue annonce qu'on peut deviner l'**île** d'un manchot (Biscoe, Dream ou Torgersen) à partir de ses quatre mesures, avec une accuracy de près de 0,9. Son code est la fonction `colleague_24`, en six étapes, A à F : lis-la attentivement.

a) `leaky_steps_24` : les lettres des étapes qui créent une fuite de données (par exemple `"XY"`).

Écris ensuite `honest_24()`, la même étude **sans fuite**. Pour que tout le monde trouve les mêmes nombres, suis ce protocole :
1. pas d'augmentation ; les quatre mesures des 333 manchots (`X_peng`) et leur île (`island`) ;
2. le découpage du ch. 1, **en premier** : 233 manchots d'entraînement (`TRAIN_CH1`), 100 de test (`TEST_CH1`) ;
3. le choix entre `IslandPipeline("1-NN")` et `IslandPipeline("centroid")` (fournie : elle refait la sélection des deux mesures et la standardisation dans son `fit`, sur les seules lignes qu'elle reçoit) se fait avec **ta** `cross_val_score` sur les 233 manchots d'entraînement, avec `cv=stratified_kfold_indices(island[TRAIN_CH1], 5)` : garde la meilleure moyenne (`"1-NN"` en cas d'égalité) ;
4. le pipeline retenu est réentraîné sur les 233 manchots, puis noté **une seule fois** sur les 100 du test.

`honest_24()` renvoie `(model_name, test_accuracy)`, le nom du modèle retenu et son accuracy de test. La vérification contrôle :
b) le modèle retenu ;
c) son accuracy de test.

Dans tes notes : pour chaque fuite, ce qu'elle laisse passer du test vers le modèle, et ce qu'elle a coûté ici. Laquelle pèse le plus, et pourquoi avec le plus proche voisin ? L'étape D change-t-elle les mesures choisies sur ces données ? Le test aurait-il préféré l'autre modèle ? Fallait-il pour autant le choisir ?""",
       given=DATA_24, todo=TODO_24, check=RELOAD + CHECK_24,
       solution=SOLUTION_24 + solved(CHECK_24, "8.24").replace('print_answer("8.24a", as_letters(leaky_steps_24))',
                                                              'print_answer("8.24a", leaky_steps_24)'),
       record=r'''test_centroid_24 = IslandPipeline("centroid").fit(X_peng[TRAIN_CH1], island[TRAIN_CH1]).score(X_peng[TEST_CH1], island[TEST_CH1])
wb.record("8.24a", leaky_steps_24, mistakes={"l'étape E découpe les données : la fuite vient de ce qui a été fait AVANT elle": "BCDEF",
                                             "il manque une étape : relis la dernière": "BCD",
                                             "il manque une étape qui calcule des statistiques sur toutes les lignes": "BDF",
                                             "il manque une étape qui choisit des colonnes en regardant toutes les lignes": "BCF",
                                             "les copies bruitées d'un manchot finissent des deux côtés du découpage : il manque une étape": "CDF"})
wb.record("8.24b", result_24[0], mistakes={"c'est le modèle que le test préfère : le choix se fait par validation croisée sur l'entraînement": "centroid"})
wb.record("8.24c", result_24[1], decimals=4, mistakes={"c'est le score annoncé par la collègue : refais l'étude sans fuite": reported_24,
                                                     "c'est le score de test de l'autre modèle : le choix se fait sur l'entraînement, pas sur le test": test_centroid_24})''',
       note="Les étapes **B**, **C**, **D** et **F** fuient. B fabrique des quasi-doublons avant le découpage : la "
            "plupart des manchots de test ont une copie bruitée dans l'entraînement, et le plus proche voisin la "
            "retrouve ; c'est la fuite qui pèse le plus (0,893 annoncé). C calcule la moyenne et l'écart-type sur "
            "toutes les lignes, test compris. D choisit les mesures en regardant toutes les lignes ; sur ces "
            "données, elle retient les mêmes deux mesures (épaisseur du bec, nageoire) qu'avec l'entraînement "
            "seul, mais rien ne le garantissait. F choisit le modèle sur le test et publie son score. L'étude "
            "honnête choisit le plus proche voisin par validation croisée (0,631 contre 0,575) et obtient 0,66 sur "
            "le test. Le centroïde aurait fait 0,71 sur ce test : le choisir après coup serait refaire la fuite F. "
            "Les deux scores restent modestes (prédire toujours Biscoe, l'île la plus fréquente, ferait 0,54 sur "
            "ce test) : les Gentoo ne vivent qu'à Biscoe et les Chinstrap qu'à Dream, mais les Adélie vivent sur "
            "les trois îles, et leurs mesures ne disent pas laquelle."),

    Ex("8.25", "🔬", 3, 40, "Sélectionner des features avant la validation croisée : 90 % sur du bruit",
       "mesurer l'optimisme d'une sélection de features faite avant la validation croisée, sur des données "
       "sans aucun signal.",
       "Ex 8.22 · ch. 2 (corrélation) · fiche §8.5 (« seulement si tout est refait dans la boucle »)",
       thread="synthétique", tracks="R, C",
       body=r"""`X_25` contient 60 échantillons de 1 000 features tirées d'une loi normale, et `y_25` des labels 0 ou 1 tirés au hasard : il n'y a **rien** à apprendre, et un modèle honnête doit faire environ 50 %. Écris :
- `top_k_25(X, y, k)` : les indices des `k` colonnes de `X` dont la corrélation de Pearson avec `y` (ch. 2) est la plus grande en valeur absolue (ici, $k$ compte les features retenues, comme le paramètre `k` de `SelectKBest` dans scikit-learn ; la validation croisée garde 5 folds) ;
- `selected_before_25(X, y, k)` : la sélection des `k` features sur **toutes** les lignes, puis la moyenne de **ta** `cross_val_score(NearestCentroid(), X[:, columns], y, cv=5)` ;
- `selected_inside_25(X, y, k)` : pour chaque paire de **ta** `kfold_indices(len(X), 5)`, la sélection sur la partie d'entraînement **seulement**, puis `NearestCentroid` entraîné sur ces colonnes et noté sur la partie de validation ; la moyenne des 5 scores.

La vérification contrôle, pour $k = 20$ :
a) le score de la sélection faite avant la validation croisée ;
b) celui de la sélection faite dans chaque tour ;
puis trace les deux scores pour $k$ = 5, 10, 20, 50 et 100.

Dans tes notes : d'où viennent les 90 % de a) ? Pourquoi le score de a) monte-t-il avec $k$ ? Que dirais-tu d'un article qui annonce 90 % de bonnes prédictions avec 60 patients et 1 000 gènes ? Comment `Pipeline` de scikit-learn (ch. 15) évite-t-il ce piège ?""",
       given=DATA_25, todo=TODO_25, check=RELOAD + CHECK_25,
       solution=SOLUTION_25 + solved(CHECK_25, "8.25"),
       record=r'''wb.record("8.25a", before_25, decimals=4, mistakes={"c'est le score de la sélection faite dans chaque tour (question b)": inside_25})
wb.record("8.25b", inside_25, decimals=4, mistakes={"c'est le score de la sélection faite sur toutes les lignes (question a)": before_25})''',
       note="Sélectionnées sur toutes les lignes, les 20 features les plus corrélées au label le sont **aussi** "
            "avec les labels des lignes de validation : parmi 1 000 features de bruit, certaines s'alignent par "
            "hasard sur les 60 labels, et la sélection les trouve. La validation croisée note alors un modèle "
            "qui a déjà vu, par la sélection, ce qu'il devait prédire : 0,900. Sélectionnées dans chaque tour, "
            "elles ne disent rien des lignes de validation : 0,433, le niveau du hasard (à la variabilité près, "
            "avec 12 lignes par fold). Plus $k$ est grand, plus la sélection capture de hasard (0,77 pour $k = 5$, "
            "0,98 pour $k = 50$). Un tel article doit dire **où** la sélection a eu lieu : c'est l'erreur relevée "
            "par Ambroise et McLachlan (2002). Un `Pipeline` contient la sélection : `cross_val_score` la refait "
            "dans chaque tour, comme `IslandPipeline` en 8.24 (15.24 refait cette expérience)."),

    Ex("8.26", "🔬", 3, 40, "Comparer deux modèles honnêtement : test par permutation, p-valeur et bootstrap apparié",
       "décider si un modèle en bat vraiment un autre sur le même jeu de test, avec un test par permutation et un "
       "intervalle bootstrap apparié.",
       "Ex 8.22 · ch. 2 (bootstrap, 2.22 à 2.24) · fiche §8.6 (encadré 🧮 sur le test par permutation)",
       thread="Penguins", tracks="R, M, C",
       body=r"""Deux centroïdes les plus proches, entraînés sur les 233 manchots du ch. 1 et notés sur les 100 du test : **A** utilise la longueur et l'épaisseur du bec, standardisées ; **B** les quatre mesures, standardisées. `correct_A_26` et `correct_B_26` valent 1 pour chaque manchot de test bien classé, 0 sinon. B fait 0,98, A 0,95 : B est-il vraiment meilleur ?

a) `discordant_26` : la liste `[nombre de manchots de test où seul A a raison, nombre où seul B a raison]`.

Écris ensuite, avec $d_i$ = `correct_b[i] - correct_a[i]` (fiche §8.6) :
- `permutation_pvalue_26(correct_a, correct_b, n_perm=10_000, seed=826)` : `rng = np.random.default_rng(seed)`, puis **un seul** tirage `flips = rng.random((n_perm, n)) < 0.5` ; chaque ligne de `flips` change le signe des $d_i$ où elle vaut `True` ; compte les lignes dont la somme, en valeur absolue, est au moins égale à $|\sum_i d_i|$ (travaille avec ces sommes **entières**, pas avec des moyennes, pour éviter les pièges d'arrondi), et renvoie $(\text{compte} + 1) / (n_{\text{perm}} + 1)$ ;
- `paired_bootstrap_26(correct_a, correct_b, n_boot=10_000, seed=8260)` : `rng = np.random.default_rng(seed)`, puis **un seul** tirage `rows = rng.integers(0, n, size=(n_boot, n))` ; chaque ligne de `rows` est un rééchantillon des manchots de test (avec remise), dont on calcule l'écart d'accuracy B − A ; renvoie les percentiles 2,5 et 97,5 de ces écarts (`np.percentile`).

La vérification contrôle, pour la comparaison 1 (A, puis B) :
b) la p-valeur ;
c) l'intervalle à 95 % de l'écart B − A ;
puis la compare à la p-valeur exacte du test de McNemar (`scipy.stats.binomtest` sur les désaccords), et contrôle, pour la comparaison 2 (les quatre mesures **brutes**, puis B) :
d) la p-valeur.
e) `significant_26` : d'après b) et c), B bat-il A au seuil de 5 % ? (`True` ou `False`)

Dans tes notes : pourquoi seuls les désaccords comptent-ils ? Que veut dire la p-valeur de b), et que ne veut-elle **pas** dire ? Combien de manchots de test faudrait-il, à peu près, pour départager A et B ? Pourquoi la p-valeur de d) ne peut-elle pas descendre sous $1 / 10\,001$ ?""",
       given=DATA_26, todo=TODO_26, check=RELOAD + CHECK_26,
       solution=SOLUTION_26 + solved(CHECK_26, "8.26"),
       record=r'''d_26 = correct_B_26 - correct_A_26
sums_26 = np.where(np.random.default_rng(826).random((10_000, 100)) < 0.5, -d_26, d_26).sum(axis=1)
one_sided_26 = (1 + int(np.sum(sums_26 >= d_26.sum()))) / 10_001
count_26 = round(p_value_26 * 10_001) - 1
wb.record("8.26a", discordant_26, mistakes={"l'ordre demandé est [seul A a raison, seul B a raison]": discordant_26[::-1]})
wb.record("8.26b", p_value_26, decimals=4, mistakes={"c'est une p-valeur unilatérale : le test de la fiche compte |somme| ≥ |écart observé|, dans les deux sens": one_sided_26,
                                                     "applique la formule de l'énoncé : (compte + 1) / (n_perm + 1), pas compte / n_perm": count_26 / 10_000})
wb.record("8.26c", interval_26, decimals=4, mistakes={"c'est l'intervalle de A − B : la question demande l'écart B − A": [-interval_26[1], -interval_26[0]]})
wb.record("8.26d", p_value_raw_26, decimals=4, mistakes={"une p-valeur par permutation ne vaut jamais 0 : la formule de l'énoncé ajoute 1 au compte et au nombre de permutations": 0.0})
wb.record("8.26e", significant_26, mistakes={"compare la p-valeur de b) à 0,05, et regarde si l'intervalle de c) contient 0": True})''',
       note="A et B ne sont en désaccord que sur 3 manchots, et B a raison les 3 fois : $\\sum_i d_i = 3$. Sous "
            "l'hypothèse nulle, ces trois désaccords tombent du côté de B ou de A à pile ou face, et $|D| = 3$ "
            "arrive dans 2 cas sur 8 : la p-valeur exacte vaut 0,25 (McNemar), la permutation donne 0,248. "
            "L'intervalle bootstrap va de 0 à 0,07 : il touche 0. Conclusion : rien ne permet de dire que B est "
            "meilleur que A avec 100 manchots de test (`significant_26 = False`). Ce n'est pas une preuve que A et "
            "B se valent : il faudrait plus de désaccords, donc un test bien plus grand (avec la même proportion "
            "de désaccords, au moins 6 désaccords tous favorables à B, soit environ deux fois plus de manchots, "
            "pour descendre sous 0,05). Contre les mesures brutes (0,67), B gagne les 31 désaccords : aucune des "
            "10 000 permutations n'atteint 31, et la p-valeur vaut son minimum, $1/10\\,001 \\approx 0{,}0001$."),

    Ex("8.27", "🏆", 3, 60, "Défi : la meilleure paire de features, choisie sans toucher au test",
       "choisir un modèle avec les seules données d'entraînement, et le valider une seule fois sur le test.",
       "Ex 8.22 · Ex 8.21 · Ex 8.12 · fiche §8.4 et §8.5", thread="Penguins", tracks="C",
       body=r"""Prédis le **sexe** d'un manchot à partir de **deux** de ses quatre mesures. Les candidats : une paire de colonnes de `X_peng` (6 paires), standardisée ou non, puis le centroïde le plus proche (`"centroid"`) ou le plus proche voisin (`"1-NN"`), soit 24 candidats, que `PairModel(columns, standardize, model)` (fourni) sait construire. Les règles :
- tu écris `choose_27(X_train, y_train)`, qui renvoie le triplet `(columns, standardize, model)` choisi à partir des **seuls** manchots d'entraînement qu'elle reçoit ;
- le test du ch. 1 (100 manchots) n'est révélé qu'**une fois** par session, quand tu mets `READY_27 = True` : si tu changes ensuite d'avis, ta note reste celle du premier choix (comme sur une plateforme de compétition, le score de test ne sert pas à choisir) ;
- `grade_27` relance aussi ta fonction sur 20 autres découpages des 333 manchots (233 + 100) : ta **méthode** doit tenir, pas seulement ton choix.

**Objectif : une accuracy d'au moins 0,90 sur le test du ch. 1, et d'au moins 0,86 en moyenne sur les 20 autres découpages.** Le point de départ fourni choisit le candidat qui a la meilleure accuracy **d'entraînement** : il n'y arrive pas (pourquoi ? relis 8.12).

Dans tes notes : ta méthode, le candidat choisi, et pourquoi le point de départ échoue. Pourquoi la paire choisie est-elle plausible, d'après ce que tu sais des manchots (les mâles sont plus lourds, à espèce égale) ?""",
       given=DATA_27, todo=TODO_27, check=RELOAD + CHECK_27,
       solution=SOLUTION_27 + solved(CHECK_27, "8.27"),
       note="Le point de départ échoue : un plus proche voisin recopie le label de chaque manchot d'entraînement, sauf "
            "quand deux manchots de sexes différents ont exactement les mêmes mesures (8.12) ; son accuracy d'entraînement "
            "vaut donc 1 ou presque (1 pour la longueur du bec associée à son épaisseur ou à la masse, 0,98 ou 0,99 pour "
            "les autres paires), et le point de départ retient le premier 1-NN de la liste (longueur et épaisseur du bec, "
            "brutes), qui fait 0,81 sur le test. Une validation croisée stratifiée (5 folds, mélangée) sur les 233 manchots d'entraînement "
            "retient l'épaisseur du bec et la masse, standardisées, avec le centroïde le plus proche (0,880 en "
            "validation croisée, contre 0,832 pour le candidat suivant) : 0,91 sur le test, et 0,876 en moyenne sur les 20 autres découpages. Les variantes "
            "honnêtes (folds non mélangés, 10 folds, k-fold répétée) font le même choix. La masse et l'épaisseur "
            "du bec séparent bien les sexes **à l'intérieur** de chaque espèce (les mâles sont plus lourds et "
            "ont un bec plus épais), et la standardisation les met à la même échelle : sans elle, la masse, en "
            "grammes, écraserait tout le reste."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 8.Q1 à 8.Q11, 8.R1, 8.R2, 8.1 à 8.6, 8.8 | vérifier tes réponses courtes | 🧠 🔁 ✏️ ∂ 📈 | ★ à ★★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")          # a "|" in a title must not split the table row
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if kind == "exercise":
        title = "# 8 · Entraînement et test — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch08_train_test/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 8`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge ouvre cette version du dépôt, "
               "qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique `00_setup/COLAB.md` §2.")
    else:
        title = "# 8 · Entraînement et test — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Découper des données en entraînement, validation et test, avec ou sans stratification.\n"
               "- Programmer `train_test_split`, la k-fold simple et stratifiée, `clone` et `cross_val_score`.\n"
               "- Choisir un hyperparamètre sur un jeu de validation, puis tester une seule fois.\n"
               "- Reconnaître et corriger les fuites de données ; traiter les données dépendantes.\n"
               "- Chiffrer l'incertitude d'un score et comparer deux modèles avec un test apparié.\n\n"
               "**Rappel express.** $n_{\\text{test}} = \\lceil t \\cdot n \\rceil$ ; folds de "
               "$\\lfloor n/k \\rfloor + 1$ exemples pour les $n \\bmod k$ premiers ; "
               "$\\mathrm{SE} = \\sqrt{\\hat{p}(1-\\hat{p})/n}$ ; $P(\\max_j S_j \\ge s) = 1 - (1 - P(S \\ge s))^K$ ; "
               "$R^2 = 1 - SS_{\\text{res}} / SS_{\\text{tot}}$ ; p-valeur d'une permutation "
               "$(C + 1)/(n_{\\text{perm}} + 1)$. En Python : `sklearn.model_selection` (`train_test_split`, "
               "`KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, `cross_val_score`), "
               "`scipy.stats.binomtest`.")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dire, pour chaque jeu (entraînement, validation, test), à quoi il sert et ce qu'on n'a "
               "pas le droit d'en faire ?\n"
               "2. Sais-tu repérer une fuite dans un protocole (prétraitement, sélection, choix sur le test, "
               "doublons, données dépendantes) et la corriger ?\n"
               "3. Sais-tu dire si un écart entre deux scores est significatif, et choisir entre hold-out et "
               "validation croisée ?\n\n"
               "**Pour aller plus loin** : les guides « Cross-validation » et « Common pitfalls and recommended "
               "practices » de scikit-learn, cités dans la fiche. La suite : le ch. 9 (overfitting et "
               "underfitting), où ta `cross_val_score` choisira la régularisation, et les courbes d'apprentissage "
               "montreront l'écart entre entraînement et validation.")]


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
