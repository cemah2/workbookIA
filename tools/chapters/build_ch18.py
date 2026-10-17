#!/usr/bin/env python
"""Build the two notebooks of chapter 18 from a single source (used by Claude).

    python tools/chapters/build_ch18.py
    python tools/run_all_notebooks.py chapitres/ch18_backprop/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch18_backprop/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
18.1 to 18.5 and of the 🧮 estimate 18.9 (02_exercices.md). The proofs ∂ 18.6 to 18.8, the reflection (🗣️ 18.10,
📄 18.11) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 18.12 to 18.29) come with the second generation session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Paper, badge, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "18"
FOLDER = "chapitres/ch18_backprop"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np
from scipy import special


def sigma(z):
    return special.expit(z)


# 18.3: L(w) = (a/2)(w - w*)^2 with a = 4, w* = 1, from w0 = 3
A_183, WSTAR_183, W0_183 = 4.0, 1.0, 3.0


def w_183(eta, t, curvature=A_183):
    return WSTAR_183 + (1 - eta * curvature) ** t * (W0_183 - WSTAR_183)


# 18.4 and 18.5: the tiny network X, Y -> A, B -> C, D (no bias), one example, label (1, 0)
X_184 = np.array([1.0, 2.0])
W1_184 = np.array([[0.5, 1.0], [-0.5, 0.25]])     # rows X, Y; columns A, B
W2_184 = np.array([[1.0, -1.0], [0.5, 0.5]])      # rows A, B; columns C, D
Y_184 = np.array([1.0, 0.0])


def tiny_184(W1, W2, f=lambda z: z, df=lambda z: np.ones_like(z)):
    """Forward and backward pass of the tiny network, loss 1/2 sum (out - y)^2, linear outputs."""
    z1 = X_184 @ W1
    a1 = f(z1)
    out = a1 @ W2
    loss = 0.5 * float(np.sum((out - Y_184) ** 2))
    d2 = out - Y_184
    d1 = (W2 @ d2) * df(z1)
    return dict(z1=z1, a1=a1, out=out, loss=loss, d2=d2, d1=d1, g1=np.outer(X_184, d1), g2=np.outer(a1, d2))


LIN_184 = tiny_184(W1_184, W2_184)
SIG_185 = tiny_184(W1_184, W2_184, f=sigma, df=lambda z: sigma(z) * (1 - sigma(z)))
GRAD2_184 = float(np.sum(LIN_184["g1"] ** 2) + np.sum(LIN_184["g2"] ** 2))

# 18.9: a 784-256-128-10 MLP on MNIST
SIZES_189 = [784, 256, 128, 10]
P_189 = sum(a * b + b for a, b in zip(SIZES_189[:-1], SIZES_189[1:]))
F_189 = sum(a * b for a, b in zip(SIZES_189[:-1], SIZES_189[1:]))      # multiplications of one forward pass'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
LETTERS = 'letters glued in alphabetical order, e.g. "BD"'
SENSE = "+1 (up) or -1 (down)"
LIST2 = "a list of 2 numbers, e.g. [1.5, -2]"
LIST4 = "a list of 4 numbers"

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("18.Q1", "Pourquoi apprendre la rétropropagation si PyTorch la fait pour toi ?", [
        ("a", TF, "False",
         r'''mistakes={"relis le §18.1 : qui calcule les gradients dans un projet réel ?": True}'''),
        ("b", LETTER, '"D"',
         r'''mistakes={"relis les quatre raisons du livre : la littérature et les échanges en font partie": "A",
          "relis les quatre raisons du livre : le diagnostic d'un réseau qui apprend mal en fait partie": "B",
          "relis les quatre raisons du livre : les idées qui s'appuient sur elle en font partie": "C"}'''),
        ("c", TF, "False",
         r'''mistakes={"relis le début du §18.1 : un réseau n'est qu'un assemblage de petits calculs": True}'''),
    ]),
    Paper("18.Q2", "Perturber un poids au hasard : pourquoi est-ce si lent ?", [
        ("a", "a whole number", "1_000_000",
         r'''mistakes={"un essai par poids, une passe avant par essai : la passe de départ ne compte pas ici": 1_000_001,
          "un seul essai par poids, et une seule passe avant par essai": 2_000_000,
          "combien de poids faut-il essayer, chacun une fois ?": 1}'''),
        ("b", "a proportion", "0.5",
         r'''decimals=2, mistakes={"deux sens possibles, tirés au hasard : lequel fait baisser l'erreur ?": 0.25,
          "une proportion entre 0 et 1 : combien de sens possibles, et combien de bons ?": 1.0}'''),
        ("c", TF, "True",
         r'''mistakes={"relis la p. 714 : un poids sert à tous les exemples": False}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"relis la fin du §18.2.1 : l'astuce du biais du ch. 10": "A",
          "le biais aussi doit être appris": "C", "c'est l'entrée du biais qui vaut 1, pas le biais lui-même": "D"}'''),
    ]),
    Paper("18.Q3", "Ce que le gradient dit à chaque poids, et le cas du biais", [
        ("a", SENSE, "1",
         r'''mistakes={"une pente négative : que fait l'erreur quand le poids augmente ?": -1,
          "il faut choisir un sens : +1 ou −1": 0}'''),
        ("b", SENSE, "-1",
         r'''mistakes={"une pente positive : que fait l'erreur quand le poids augmente ?": 1,
          "il faut choisir un sens : +1 ou −1": 0}'''),
        ("c", TF, "True",
         r'''mistakes={"relis la fin du §18.2.2 : c'est justement le pari du livre, à petits pas": False}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"les activations usuelles ont une dérivée (presque) partout": "A",
          "le livre remet les activations au §18.10": "C", "relis le ch. 17 : un réseau sans activation s'effondre": "D"}'''),
    ]),
    Paper("18.Q4", "Le delta d'un neurone : signe et effet sur l'erreur", [
        ("a", LETTER, '"A"',
         r'''mistakes={"le produit de deux nombres de même signe est positif : l'erreur change dans quel sens ?": "B",
          "le delta est le facteur entre les deux changements : l'erreur bouge": "C"}'''),
        ("b", SENSE, "1",
         r'''mistakes={"il faut un changement de signe opposé à celui du delta": -1,
          "il faut choisir un sens : +1 ou −1": 0}'''),
        ("c", TF, "False",
         r'''mistakes={"relis le §18.4.1 : de 7 à 7,5 ou de 700 à 700,5, l'effet est-il différent ?": True}'''),
        ("d", "a number", "0.4 * -2.5",
         r'''decimals=2, mistakes={"le delta multiplie le changement de la sortie, pas sa nouvelle valeur": -2501.0,
          "le changement de la sortie est 1 000,4 − 1 000": -0.25}'''),
        ("e", TF, "True",
         r'''mistakes={"relis le §18.4 : deux étapes, dont une seule est la rétropropagation au sens strict": False}'''),
    ]),
    Paper("18.Q5", "Le réseau minuscule : couches, poids et leurs noms", [
        ("a", "a whole number", "8",
         r'''mistakes={"le réseau minuscule n'a pas de biais": 12, "compte les fils : chaque entrée va à A et à B, chaque caché à C et à D": 4,
          "compte les fils entre les couches, pas les neurones": 6}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"le neurone de départ vient en premier dans le nom": "B", "c'est la notation d'un delta": "C",
          "c'est la notation d'une sortie": "D"}'''),
        ("c", "a whole number", "2",
         r'''mistakes={"la couche d'entrée ne calcule rien et ne compte pas": 3, "une couche cachée et une couche de sortie": 1}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"le réseau prédit la plus grande des deux sorties": "A",
          "ce ne sont pas des probabilités, mais la plus grande sortie désigne quand même une classe": "C"}'''),
        ("e", "a list of 4 numbers", "[0, 0, 1, 0]",
         r'''mistakes={"les indices commencent à 0 : l'indice 2 est la troisième case": [0, 1, 0, 0],
          "quatre classes d'indices 0 à 3 : où est l'indice 2 ?": [0, 0, 0, 1]}'''),
    ]),
    Paper("18.Q6", "Delta de sortie, delta caché, mise à jour d'un poids", [
        ("a", LETTER, '"A"',
         r'''mistakes={"un petit changement du poids est multiplié par a avant d'arriver au neurone": "B",
          "relis le §18.7 : la sortie de départ multiplie, elle ne divise pas": "C",
          "c'est le facteur d'un chemin dans le delta d'un neurone caché": "D"}'''),
        ("b", "a number", "1 - 0.1 * 2 * -0.5",
         r'''decimals=2, mistakes={"δ est négatif : retirer η a δ fait monter le poids": 0.9,
          "as-tu multiplié par la sortie a ?": 1.05, "as-tu multiplié par η ?": 2.0}'''),
        ("c", "a number", "0.5 * 2 + -1 * 3",
         r'''decimals=2, mistakes={"le poids −1 garde son signe": 4.0, "il manque un des deux chemins": 1.0,
          "chaque delta est multiplié par le poids de son fil, puis on additionne": 2.5}'''),
        ("d", TF, "True",
         r'''mistakes={"relis le §18.8 : les couches plus lointaines interviennent-elles directement ?": False}'''),
    ]),
    Paper("18.Q7", "Rétropropager couche par couche, puis avec une activation", [
        ("a", LETTER, '"D"',
         r'''mistakes={"les deltas partent de la fin du réseau": "A", "relis le §18.9 : où l'erreur est-elle connue directement ?": "B",
          "relis le §18.9 : quelle couche n'a pas de couche suivante ?": "C"}'''),
        ("b", "a whole number", "3",
         r'''mistakes={"on recule d'une seule couche depuis la sortie": 4, "on recule d'une couche, pas jusqu'au début": 2}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"c'est la valeur de l'activation, pas sa pente": "A", "les dérivées du ch. 17 se prennent en z": "C",
          "une pente forte amplifie : on multiplie par la pente": "D"}'''),
        ("d", "a number", "0.25",
         r'''decimals=2, mistakes={"c'est σ(0) ; le facteur est la dérivée": 0.5, "σ′(z) = σ(z)(1 − σ(z))": 1.0}'''),
        ("e", TF, "True",
         r'''mistakes={"la dérivée de l'identité vaut 1 : que change la multiplication ?": False}'''),
    ]),
    Paper("18.Q8", "Learning rate trop grand ou trop petit : reconnaître les symptômes", [
        ("a", LETTER, '"A"',
         r'''mistakes={"des poids qui sautent d'un bord à l'autre : des pas trop courts ou trop longs ?": "B",
          "une loss qui ne baisse jamais n'est pas un bon réglage": "C",
          "relis le §18.11.1 : le symptôme vient du pas, les données sont les mêmes": "D"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"une loss qui baisse régulièrement ne diverge pas": "A", "bien choisi, il irait plus vite": "C",
          "la loss baisse : les données ne sont pas en cause": "D"}'''),
        ("c", TF, "True",
         r'''mistakes={"deux classes équilibrées : quelle accuracy pour un réseau qui répond toujours pareil ?": False}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"relis la p. 785 : le livre, d'après Dauphin et coll., penche pour une autre explication": "A",
          "le réseau finit par repartir : rien n'est cassé": "C", "c'est un comportement de la surface de la loss": "D"}'''),
        ("e", TF, "True",
         r'''mistakes={"η multiplie chaque correction : que vaut-elle pour η = 0 ?": False}'''),
    ]),
    Paper("18.Q9", "Ce que la rétropropagation propage vraiment", [
        ("a", TF, "False",
         r'''mistakes={"relis le §18.12.2 : l'erreur sert-elle ailleurs qu'au départ ?": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"relis le §18.12.3 : un delta est un taux de variation": "A", "un delta est porté par un neurone, pas par un fil": "C",
          "la sortie et le delta sont deux nombres différents": "D"}'''),
        ("c", 'four letters in order, e.g. "CADB"', '"BDCA"',
         r'''mistakes={"les deltas se calculent de la sortie vers l'entrée": "BCDA", "il faut d'abord une passe avant": "DBCA",
          "les poids ne changent qu'une fois tous les deltas connus": "BDAC"}'''),
        ("d", TF, "False",
         r'''mistakes={"qui fixe la taille du pas : la rétropropagation ou le learning rate ?": True}'''),
    ]),
    Paper("18.Q10", "Neurones saturés et ReLU mortes", [
        ("a", TF, "True",
         r'''mistakes={"σ′(15) = σ(15)(1 − σ(15)) et 1 − σ(15) ≈ e^(−15)": False}'''),
        ("b", "a number", "0.0",
         r'''decimals=2, mistakes={"la pente 1 de ReLU ne vaut que pour z > 0": 1.0, "la pente de ReLU n'est jamais négative": -1.0,
          "c'est σ′(0), la dérivée de la sigmoïde ; on demande celle de ReLU": 0.25, "c'est σ(0) ; on demande la dérivée de ReLU": 0.5}'''),
        ("c", LETTERS, '"ACD"',
         r'''mistakes={"il en manque une : relis les remèdes du §18.12.4": "AC", "il en manque une : relis la fin du §18.12.4": "CD",
          "il en manque une : relis les remèdes proposés par le livre et par la fiche": "AD",
          "un grand learning rate pousse les poids loin de 0": "ABCD", "plus de sigmoïdes, c'est plus de neurones qui peuvent saturer": "ACDE"}'''),
        ("d", TF, "True",
         r'''mistakes={"pour z > 0, la pente de ReLU vaut 1 : y a-t-il un plateau ?": False}'''),
    ]),
    Paper("18.Q11", "Mini-batches, mises à jour parallèles et coût du backward", [
        ("a", "a whole number", "6400 // 32",
         r'''mistakes={"une mise à jour par mini-batch, pas par exemple": 6400, "c'est la taille du mini-batch": 32,
          "6 400 / 32": 199}'''),
        ("b", TF, "False",
         r'''mistakes={"relis le §18.12.5 : quand se fait la mise à jour ?": True}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"les deltas coûtent une passe avant ; et les gradients des poids ?": "A", "c'est l'ordre de grandeur des différences finies, pas de la rétropropagation": "C",
          "le coût reste proportionnel au nombre de poids": "D"}'''),
        ("d", TF, "True",
         r'''mistakes={"relis le §18.12.6 : de quoi dépend la mise à jour d'un poids ?": False}'''),
    ]),
    Paper("18.Q12", "Aucune garantie, une longue histoire, des équations à apprivoiser", [
        ("a", TF, "False",
         r'''mistakes={"relis le §18.12.8 : le perceptron a une preuve, la rétropropagation non": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"relis le §18.12.8 : on commence par calmer les choses": "A", "relis le §18.12.8 : le réseau vient en second": "C",
          "relis le §18.12.8 : le premier réflexe porte sur un réglage": "D"}'''),
        ("c", "a year", "1970",
         r'''mistakes={"c'est l'année de sa publication en anglais ; on demande le mémoire": 1976, "c'est la thèse de Werbos": 1974,
          "c'est l'article de Rumelhart, Hinton et Williams": 1986, "c'est la date des travaux d'Ostrowski": 1971}'''),
        ("d", "a year", "1986",
         r'''mistakes={"c'est la publication de Werbos sur les réseaux": 1982, "c'est le mémoire de Linnainmaa": 1970}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"vérifie les formes : (n, n_out) fois (n_in, n_out) ne se multiplie pas": "A",
          "aucune inverse : on transporte des gradients": "C", "relis le §18.12.10 : un seul produit par une matrice de poids": "D"}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("18.R1", "La dérivée de la sigmoïde à partir de sa sortie", [
        ("a", "a number", "0.8 * 0.2",
         r'''decimals=2, mistakes={"σ′ = σ(1 − σ), avec σ = 0,8": 0.2, "c'est la formule de tanh′": 0.36}'''),
        ("b", "a number", "0.5 * 0.5",
         r'''decimals=2, mistakes={"σ′ = σ(1 − σ), avec σ = 0,5": 0.5}'''),
        ("c", "a number", "1 - 0.6 ** 2",
         r'''decimals=2, mistakes={"tanh′ = 1 − tanh², pas 1 − tanh": 0.4, "c'est la formule de la sigmoïde": 0.24, "c'est tanh²(z), pas encore la dérivée : revois la formule de tanh′ au ch. 17": 0.36}'''),
        ("d", TF, "True",
         r'''mistakes={"σ′ s'écrit avec σ seul : a(1 − a)": False}'''),
        ("e", "a number", "0.0",
         r'''decimals=2, mistakes={"la pente 1 de ReLU ne vaut que pour z > 0": 1.0, "c'est z ; on demande la pente de ReLU en z": -0.3,
          "c'est σ′(−0,3), la dérivée de la sigmoïde ; on demande celle de ReLU": float(sigma(-0.3) * (1 - sigma(-0.3))),
          "c'est σ(−0,3) ; on demande la dérivée de ReLU": float(sigma(-0.3))}'''),
    ]),
    Paper("18.R2", "Chercher le meilleur learning rate par recherche en grille", [
        ("a", "a whole number", "4 * 5 + 1",
         r'''mistakes={"refit=True réentraîne une fois de plus sur tout l'entraînement": 20, "chaque valeur est testée sur chacun des 5 folds": 5,
          "chaque valeur est testée sur 5 folds, puis le meilleur réglage est réentraîné": 4}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"scikit-learn accepte n'importe quelle liste de valeurs": "B", "il y a autant de valeurs que dans une grille linéaire de même taille": "C",
          "GridSearchCV fait la validation croisée dans tous les cas": "D"}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"ce sont les poids appris, pas le réglage": "B", "c'est le premier réglage essayé": "C", "c'est une méthode qui note un modèle": "D"}'''),
        ("d", TF, "True",
         r'''mistakes={"relis la documentation de MLPClassifier : le solveur lbfgs a-t-il un learning rate ?": False}'''),
    ]),
    Paper("18.R3", "Représentation, évaluation, optimisation dans un MLP", [
        ("a", LETTER, '"B"',
         r'''mistakes={"la loss ne dit pas ce que le modèle peut exprimer": "A", "la loss est ce qu'on minimise, pas la méthode": "C"}'''),
        ("b", LETTER, '"C"',
         r'''mistakes={"la descente ne change pas ce que le modèle peut exprimer": "A", "elle ne juge pas une solution, elle en cherche une": "B"}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"l'architecture ne juge pas une solution": "B", "l'architecture ne cherche rien : elle fixe ce qui est possible": "C"}'''),
        ("d", TF, "True",
         r'''mistakes={"ce qu'une représentation ne peut pas exprimer peut-il s'apprendre ? (ch. 11)": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("18.1", "Le delta, amplificateur de changement", [
        ("a", "a number", "-3 * 0.3",
         r'''decimals=2, mistakes={"le delta multiplie le changement de la sortie, pas sa nouvelle valeur": -4.5,
          "le changement de la sortie est 1,5 − 1,2": -0.3}'''),
        ("b", "a number", "0.5 * -0.4",
         r'''decimals=2, mistakes={"le delta multiplie le changement de la sortie, pas sa nouvelle valeur": 1.8,
          "le changement de la sortie est 3,6 − 4": -0.4}'''),
        ("c", "a number", "-0.06 / -1.5",
         r'''decimals=2, mistakes={"un delta négatif : pour faire baisser l'erreur, la sortie doit-elle monter ou baisser ?": -0.04,
          "ΔE = δ × Δo : pour trouver Δo, divise": 0.09}'''),
        ("d", "a number", "2.0 * (3 - 2)",
         r'''decimals=1, mistakes={"la dérivée de (o − 2)² est 2(o − 2)": 1.0, "c'est la dérivée de o − 2 seul ; on dérive un carré": 0.5}'''),
        ("e", "a number", "2 * 0.1",
         r'''decimals=2, mistakes={"multiplie le delta par le changement de la sortie, 0,1": 2.0, "relis la question : on demande la prévision par le delta": 0.21}'''),
        ("f", "a number with 2 decimals", "(3.1 - 2) ** 2 - (3 - 2) ** 2",
         r'''decimals=2, mistakes={"relis la question : on demande le vrai changement, E(3,1) − E(3)": 0.2,
          "c'est E(3,1) ; on demande le changement depuis E(3)": 1.21}'''),
        ("g", TF, "False",
         r'''mistakes={"le delta est une pente : que dit son signe ?": True}'''),
    ]),
    Paper("18.2", "Deltas de sortie avec l'erreur quadratique : attention au signe", [
        ("a", "a number", "0.5 * (0.6 ** 2 + 1.2 ** 2)",
         r'''decimals=2, mistakes={"n'oublie pas le facteur ½": 1.8, "relis le label : y = (0 ; 1)": 0.2}'''),
        ("b", "a number", "0.6 - 0",
         r'''decimals=2, mistakes={"δ = ŷ − y, pas y − ŷ": -0.6, "avec le ½, la dérivée de ½(ŷ − y)² est ŷ − y": 1.2}'''),
        ("c", "a number", "-0.2 - 1",
         r'''decimals=2, mistakes={"δ = ŷ − y, pas y − ŷ": 1.2, "avec le ½, la dérivée de ½(ŷ − y)² est ŷ − y": -2.4}'''),
        ("d", "a number", "-1.2 * 0.01",
         r'''decimals=3, mistakes={"un delta négatif et une sortie qui monte : l'erreur monte-t-elle ?": 0.012,
          "multiplie le delta par le changement, 0,01": -1.2}'''),
        ("e", "a number", "0.5 * ((0.6 - 0.5 * 0.6) ** 2 + (-0.2 + 0.5 * 1.2 - 1) ** 2)",
         r'''decimals=3, mistakes={"n'oublie pas le facteur ½": 0.45, "ŷ ← ŷ − 0,5 δ : le second delta est négatif, ŷ₂ monte": 0.5 * ((0.6 - 0.5 * 0.6) ** 2 + (-0.2 - 0.5 * 1.2 - 1) ** 2)}'''),
        ("f", "a number", "0.5 * ((0.6 + 0.5 * 0.6) ** 2 + (-0.2 - 0.5 * 1.2 - 1) ** 2)",
         r'''decimals=3, mistakes={"n'oublie pas le facteur ½": 4.05, "relis la question : on prend δ′ = y − ŷ": 0.225}'''),
        ("g", "a number", "2 * 0.6",
         r'''decimals=2, mistakes={"sans le ½, la dérivée de (ŷ − y)² double": 0.6, "δ = ŷ − y, pas y − ŷ": -1.2}'''),
    ]),
    Paper("18.3", "Descente sur une parabole : quel learning rate ?", [
        ("a", "a number", "1 - 0.1 * A_183",
         r'''decimals=2, mistakes={"c'est ηa ; le facteur du pas est 1 − ηa": 0.4, "la courbure a = 4 intervient dans le facteur": 0.9}'''),
        ("b", "a number", "w_183(0.1, 1)",
         r'''decimals=2, mistakes={"c'est l'écart w₁ − w* ; on demande w₁": 1.2, "le facteur multiplie l'écart au minimum, pas w₀": 1.8,
          "un pas : w₁ = w₀ − η L′(w₀), avec L′(w) = 4(w − 1)": 2.6}'''),
        ("c", "a number with 3 decimals", "w_183(0.1, 3)",
         r'''decimals=3, mistakes={"c'est l'écart w₃ − w* ; on demande w₃": 0.432, "le facteur multiplie l'écart au minimum, pas w₀": 0.648}'''),
        ("d", "a number", "1 / A_183",
         r'''decimals=3, mistakes={"relis la question : on demande d'atteindre w* en un seul pas": 0.5,
          "un pas qui atteint w* annule le facteur 1 − ηa": 4.0, "le facteur 1 − ηa doit valoir 0": 1.0}'''),
        ("e", "a number", "2 / A_183",
         r'''decimals=3, mistakes={"relis la question : on demande la limite au-delà de laquelle la descente diverge": 0.25,
          "la condition porte sur le facteur : |1 − ηa| < 1": 1.0, "le facteur doit rester au-dessus de −1": 0.125}'''),
        ("f", "a number", "w_183(0.4, 2)",
         r'''decimals=2, mistakes={"relis la question : on demande w₂": -0.2, "le facteur 1 − ηa est négatif : son carré est positif": 0.28}'''),
        ("g", "a number", "2 / 0.1",
         r'''decimals=1, mistakes={"la limite dépend de la courbure a = 0,1": 0.5, "relis la question : la limite de convergence, pas le pas qui atteint le minimum": 10.0,
          "a = 0,1 : refais la condition |1 − ηa| < 1": 2.0,
          "a est le double du coefficient : 0,05 (w − 1)² = (a/2)(w − 1)² avec a = 0,1": 40.0}'''),
        ("h", "a whole number", "math.ceil(math.log(0.01 / 2) / math.log(0.6))",
         r'''mistakes={"vérifie l'écart après le nombre de pas trouvé : il part de 2 et doit passer SOUS 0,01": 10,
          "l'écart de départ vaut |3 − 1| = 2, et le nombre de pas s'arrondit à l'entier supérieur": 9,
          "l'écart est multiplié par 0,6 à chaque pas, pas diminué de 0,6": 4},
          fractional="compte des pas entiers : le premier t pour lequel l'écart passe sous 0,01"'''),
    ]),
    Paper("18.4", "Rétropropagation complète du réseau minuscule, sans activation", [
        ("a", LIST2, "LIN_184['z1'].tolist()",
         r'''decimals=3, mistakes={"une colonne de W₁ par neurone caché : A reçoit XA et YA": [2.5, 0.0]}'''),
        ("b", LIST2, "LIN_184['out'].tolist()",
         r'''decimals=3, mistakes={"une colonne de W₂ par neurone de sortie : C reçoit AC et BC": [-2.0, 0.5]}'''),
        ("c", "a number", "LIN_184['loss']",
         r'''decimals=4, mistakes={"n'oublie pas le facteur ½": 2.125}'''),
        ("d", LIST2, "LIN_184['d2'].tolist()",
         r'''decimals=3, mistakes={"δ = ŷ − y, pas y − ŷ": [0.75, -1.25]}'''),
        ("e", LIST2, "LIN_184['d1'].tolist()",
         r'''decimals=3, mistakes={"le delta de A suit les fils qui PARTENT de A : AC vers C et AD vers D": [-0.125, 1.375],
          "il manque le chemin par D": [-0.75, -0.375], "δ = ŷ − y en sortie, pas y − ŷ": [2.0, -0.25]}'''),
        ("f", LIST4, "LIN_184['g2'].ravel().tolist()",
         r'''decimals=3, mistakes={"le gradient d'un poids multiplie la SORTIE du neurone de départ par le delta d'arrivée": [-0.75, -1.25, -0.375, 0.625],
          "l'ordre est celui de W₂ ligne par ligne : AC, AD, BC, BD": [0.375, -1.125, -0.625, 1.875]}'''),
        ("g", LIST4, "LIN_184['g1'].ravel().tolist()",
         r'''decimals=3, mistakes={"un poids de la première couche reçoit le delta du neurone CACHÉ qu'il alimente": [-0.75, 1.25, -1.5, 2.5],
          "l'ordre est celui de W₁ ligne par ligne : XA, XB, YA, YB": [-2.0, -4.0, 0.25, 0.5]}'''),
        ("h", "a number", "-0.1 * GRAD2_184",
         r'''decimals=4, mistakes={"multiplie par η = 0,1": -25.625, "les gradients se mettent au carré": 0.475,
          "le pas retire η g à chaque poids : la loss baisse": 2.5625}'''),
        ("i", "a number with 4 decimals", "tiny_184(W1_184 - 0.1 * LIN_184['g1'], W2_184 - 0.1 * LIN_184['g2'])['loss']",
         r'''decimals=4, mistakes={"on retire η × gradient : w ← w − η g": tiny_184(W1_184 + 0.1 * LIN_184['g1'], W2_184 + 0.1 * LIN_184['g2'])['loss'],
          "la prévision h) ne vaut que pour de petits pas : refais la passe avant avec les nouveaux poids": 1.0625 - 2.5625}'''),
    ]),
    Paper("18.5", "Le même réseau avec une sigmoïde cachée : le facteur σ'(z)", [
        ("a", "a number with 4 decimals", "SIG_185['a1'][0]",
         r'''decimals=4, mistakes={"c'est z_A ; applique la sigmoïde": -0.5, "relis la question : on demande la sortie σ(z_A), pas sa dérivée": SIG_185['a1'][0] * (1 - SIG_185['a1'][0])}'''),
        ("b", "a number with 4 decimals", "SIG_185['a1'][1]",
         r'''decimals=4, mistakes={"c'est z_B ; applique la sigmoïde": 1.5}'''),
        ("c", "a number with 4 decimals", "SIG_185['out'][0]",
         r'''decimals=4, mistakes={"relis l'énoncé : les neurones cachés ont maintenant une sigmoïde": 0.25,
          "les sorties restent sans activation : C_o = A_o AC + B_o BC": float(sigma(SIG_185['out'][0]))}'''),
        ("d", "a number with 4 decimals", "SIG_185['d2'][0]",
         r'''decimals=4, mistakes={"δ = ŷ − y, pas y − ŷ": -SIG_185['d2'][0], "la sortie C n'a pas d'activation : pas de facteur σ′ en sortie": SIG_185['d2'][0] * float(sigma(SIG_185['out'][0]) * (1 - sigma(SIG_185['out'][0]))),
          "tu as gardé la sortie C_o du réseau sans activation : avec la sigmoïde cachée, A_o et B_o changent, donc C_o aussi": LIN_184['d2'][0]}'''),
        ("e", "a number with 4 decimals", "float(sigma(-0.5) * (1 - sigma(-0.5)))",
         r'''decimals=4, mistakes={"σ′ se prend en z_A (ou σ(1 − σ) avec la sortie), pas en A_o": float(sigma(SIG_185['a1'][0]) * (1 - sigma(SIG_185['a1'][0]))),
          "σ′(z) = σ(z)(1 − σ(z)), pas σ(z)": float(sigma(-0.5))}'''),
        ("f", "a number with 4 decimals", "SIG_185['d1'][0]",
         r'''decimals=4, mistakes={"il manque le facteur σ′(z_A)": float(W2_184[0] @ SIG_185['d2']),
          "les deltas de sortie ne sont plus ceux de l'exercice 18.4 : refais la passe avant avec la sigmoïde": float(LIN_184['d1'][0] * sigma(-0.5) * (1 - sigma(-0.5))),
          "le facteur σ′ se prend en z_A, pas en A_o": float(W2_184[0] @ SIG_185['d2']) * float(sigma(SIG_185['a1'][0]) * (1 - sigma(SIG_185['a1'][0])))}'''),
        ("g", "a number with 4 decimals", "SIG_185['d1'][1]",
         r'''decimals=4, mistakes={"il manque le facteur σ′(z_B)": float(W2_184[1] @ SIG_185['d2'])}'''),
        ("h", "a number with 4 decimals", "SIG_185['g1'][1, 0]",
         r'''decimals=4, mistakes={"le poids YA part de Y : quelle entrée multiplie le delta ?": SIG_185['g1'][0, 0],
          "le poids YA arrive dans A : quel delta ?": SIG_185['g1'][1, 1]}'''),
    ]),
    Paper("18.9", "Fermi : rétropropagation contre force brute", [
        ("a", "a whole number", "P_189",
         r'''mistakes={"les biais sont des paramètres aussi": F_189, "compte aussi la dernière couche, 128 × 10 + 10": P_189 - 1290}'''),
        ("b", "a whole number of thousands", "round(F_189 / 1000)",
         r'''mistakes={"en milliers : divise par 1 000": F_189, "une multiplication par poids, pour UNE image": round(F_189 * 60000 / 1000)}'''),
        ("c", "a whole number", "2 * P_189",
         r'''mistakes={"en différence centrée, deux passes par paramètre": P_189, "deux passes par paramètre, biais compris": 2 * F_189}'''),
        ("d", "a whole number of thousands", "round(2 * P_189 / 3 / 1000)",
         r'''mistakes={"la rétropropagation complète coûte une passe avant ET une passe arrière (deux passes avant)": round(2 * P_189 / 1000),
          "une passe arrière vaut deux passes avant, et il faut aussi la passe avant": round(2 * P_189 / 2 / 1000),
          "en milliers : divise par 1 000": round(2 * P_189 / 3)}'''),
        ("e", "a whole number of milliseconds", "round(3 * F_189 * 60000 / 1e12 * 1000)",
         r'''mistakes={"compte la passe avant ET la passe arrière (deux passes avant)": round(F_189 * 60000 / 1e12 * 1000),
          "la passe arrière vaut deux passes avant : trois en tout": round(2 * F_189 * 60000 / 1e12 * 1000),
          "en millisecondes : 1 s = 1 000 ms": round(3 * F_189 * 60000 / 1e12 * 1e6)}'''),
        ("f", "a number of hours, 1 decimal", "2 * P_189 * F_189 * 60000 / 1e12 / 3600",
         r'''decimals=1, mistakes={"en différence centrée, deux passes avant par paramètre": P_189 * F_189 * 60000 / 1e12 / 3600,
          "en heures : 1 h = 3 600 s": 2 * P_189 * F_189 * 60000 / 1e12 / 60}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 18.1 à 18.5 et 🧮 18.9)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un sens : `+1` (monter) ou `-1` (baisser) ; un choix : la "
               "lettre seule, entre guillemets (`\"E\"`) ; plusieurs choix ou une suite : les lettres collées, dans "
               "l'ordre demandé (`\"CADB\"`) ; plusieurs nombres : une liste (`[4, 7]`). Arrondis comme l'énoncé le "
               "demande, et seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. Les "
               "questions « dans ta copie », les preuves ∂ 18.6 à 18.8, la réflexion (🗣️ 18.10, 📄 18.11) et "
               "l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 18.12 to 18.29): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "18.12 à 18.17", "les losses et leur gradient, la méthode très lente, la passe arrière d'une couche, "
                           "gradient_check, une chasse aux gradients faux"),
    ("B", "18.18 à 18.21", "mlp_backward, le réseau 2-4-4-1 à 37 poids sur les lunes, la même expérience avec "
                           "scikit-learn, les figures du learning rate"),
    ("C", "18.22 à 18.25", "lire des courbes de loss, balayer le learning rate, des neurones saturés et morts sur "
                           "MNIST, un test pytest de gradient"),
    ("D", "18.26 à 18.29", "le micro-autograd Value, quatre gradients comparés, MiniMLP, le défi MNIST"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 18.Q1 à 18.Q12, 18.R1 à 18.R3, 18.1 à 18.5, 18.9 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 | "
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
        title = "# 18 · Rétropropagation — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch18_backprop/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 18`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 18 · Rétropropagation — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Calculer à la main les deltas, les gradients et un pas de mise à jour d'un petit réseau, avec et "
               "sans activation.\n"
               "- Écrire et vérifier contre PyTorch les fonctions de `mylearn.nn.backward` (les losses avec leur "
               "gradient, `dense_backward`, `mlp_backward`, `gradient_check`) et le micro-autograd "
               "`mylearn.autograd_mini` (`Value`, `MiniMLP`).\n"
               "- Entraîner un MLP from scratch et diagnostiquer le learning rate, les plateaux, la saturation et les "
               "neurones morts.\n\n"
               "**Rappel express.** $\\delta_j = \\partial L / \\partial z_j$ ; sortie, loss quadratique : "
               "$\\delta_k = (\\hat{y}_k - y_k)\\,f'(z_k)$ ; caché : $\\delta_j = f'(z_j) \\sum_k w_{jk}\\,\\delta_k$ ; "
               "$\\partial L / \\partial w_{ij} = a_i\\,\\delta_j$, $\\partial L / \\partial b_j = \\delta_j$ ; "
               "$w \\leftarrow w - \\eta\\,\\partial L / \\partial w$. En matrices : $\\boldsymbol{\\Delta} = \\mathbf{G} "
               "\\odot f'(\\mathbf{Z})$, $d\\mathbf{W} = \\mathbf{A}_{\\text{prev}}^\\top\\boldsymbol{\\Delta}$, "
               "$d\\mathbf{b} = \\mathbf{1}^\\top\\boldsymbol{\\Delta}$, $\\mathbf{G}_{\\text{prev}} = "
               "\\boldsymbol{\\Delta}\\mathbf{W}^\\top$. Cross-entropy sur les logits : $(\\mathbf{P} - \\mathbf{Y})/n$ ; "
               "binaire : $(\\sigma(\\mathbf{z}) - \\mathbf{y})/n$. Vérification : $\\frac{L(w + \\varepsilon) - "
               "L(w - \\varepsilon)}{2\\varepsilon}$, erreur relative $\\frac{|a - n|}{\\max(|a| + |n|, 10^{-12})}$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 18.12 à 18.29 (parties A à D, ta librairie `nn/backward.py` et "
             "`autograd_mini.py`) seront ajoutés à la prochaine session de génération ; relance alors "
             "`python tools/start_chapter.py 18` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dérouler à la main la passe avant, les deltas et un pas de mise à jour d'un petit réseau ?\n"
               "2. Sais-tu écrire les gradients d'une couche dense pour un mini-batch, et vérifier leurs formes ?\n"
               "3. Sais-tu vérifier un gradient par différences finies, et reconnaître un learning rate mal réglé ?\n\n"
               "**Pour aller plus loin** : le chapitre 2 du livre de M. Nielsen, les notes de CS231n et micrograd "
               "d'A. Karpathy, cités dans la fiche. La suite : le ch. 19, où les optimiseurs remplaceront la mise à "
               "jour $w \\leftarrow w - \\eta g$." + later)]


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
