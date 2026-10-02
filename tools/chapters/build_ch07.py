#!/usr/bin/env python
"""Build the two notebooks of chapter 7 from a single source (used by Claude).

    python tools/chapters/build_ch07.py
    python tools/run_all_notebooks.py chapitres/ch07_classification/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch07_classification/03_notebook.ipynb

Part 0 checks the short answers of the quizzes (all but Q10), of the recalls, of the ✏️
paper exercises 7.1 to 7.4 and of the ∂ exercises 7.5 and 7.6. Parts A to D (exercises
7.11 to 7.31, mylearn.cluster and mylearn.multiclass) come with the second generation
session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Paper, badge, md, paper_cells, part_cells, setup_cell,  # noqa: E402
                         write_notebook)

CHAPTER = "7"
FOLDER = "chapitres/ch07_classification"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np


def entropy_bits(p):
    """Entropy in bits of a distribution (counts or proportions), with 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def posterior(prior, f1, f0):
    """Bayes rule for two classes: P(class 1 | x) from the prior and the two densities at x."""
    return prior * f1 / (prior * f1 + (1 - prior) * f0)


def n_ovo(k):
    """Number of one-versus-one duels for k classes."""
    return k * (k - 1) // 2


A_R3, B_R3 = np.array([3, -1, 2]), np.array([1, 2, 0])            # 7.R3

# 7.2: the six duels in the order A-B, A-C, A-D, B-C, B-D, C-D and the index of each winner
WINNERS_72A = [1, 0, 3, 1, 1, 3]                                    # first point
WINNERS_72B = [0, 2, 0, 1, 3, 2]                                    # second point


def votes(winners):
    counts = np.zeros(4, dtype=int)
    for winner in winners:
        counts[winner] += 1
    return counts


# 7.3: Lloyd's algorithm by hand, k = 2, starting centres P1 and P3
P_73 = np.array([[1, 1], [2, 1], [1, 3], [5, 4], [6, 5], [7, 4]], dtype=float)


def assign(points, centres):
    return ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def update(points, labels, k):
    return np.array([points[labels == j].mean(axis=0) for j in range(k)])


def inertia(points, centres, labels):
    return float(((points - centres[labels]) ** 2).sum())


C0_73 = P_73[[0, 2]]
L1_73 = assign(P_73, C0_73)
C1_73 = update(P_73, L1_73, 2)
L2_73 = assign(P_73, C1_73)
C2_73 = update(P_73, L2_73, 2)
L3_73 = assign(P_73, C2_73)

# 7.4: densities n / b**d
N_74, B_74 = 360, 6
EMPTY_74 = (124 / 125) ** 10                                        # the book's 10 eggs in 125 small cubes

# 7.6: volume of the unit ball, V_d = 2 pi / d V_(d-2)
V = {1: 2.0, 2: math.pi}
for d in range(3, 101):
    V[d] = 2 * math.pi / d * V[d - 2]'''

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("7.Q1", "Label, prédiction, vérité terrain : le vocabulaire", [
        ("a", 'the letter of the expression that is NOT the expert\'s class, e.g. "A"', '"C"',
         r'''mistakes={"le label (étiquette) est la classe fixée par l'expert": "A",
          "la vérité terrain est la classe que l'on tient pour juste, celle de l'expert": "B",
          "le livre donne la valeur réelle (actual value) comme un autre nom du label": "D"}'''),
        ("b", "True or False", "True",
         r'''mistakes={"le mirage est un savoir-faire humain : un mireur ne peut-il jamais se tromper sur un œuf ambigu ?": False}'''),
        ("c", "True or False", "False",
         r'''mistakes={"ce qui compte est la réussite sur des œufs que le modèle n'a jamais vus (ch. 1)": True}'''),
    ]),
    Paper("7.Q2", "Binaire, multi-classe ou multi-étiquette ? Cinq situations", [
        ("a", '"B", "C" or "E"', '"B"',
         r'''mistakes={"spam ou non : il n'y a que deux classes": "C",
          "un e-mail ne reçoit qu'une seule réponse parmi deux": "E"}'''),
        ("b", '"B", "C" or "E"', '"C"',
         r'''mistakes={"trois espèces possibles : plus de deux classes": "B",
          "un manchot n'a qu'une espèce : une seule classe par exemple": "E"}'''),
        ("c", '"B", "C" or "E"', '"E"',
         r'''mistakes={"une photo peut recevoir plusieurs mots-clés à la fois": "C",
          "il y a bien plus de deux mots-clés possibles, et une photo peut en avoir plusieurs": "B"}'''),
        ("d", '"B", "C" or "E"', '"C"',
         r'''mistakes={"dix chiffres possibles : plus de deux classes": "B",
          "une image ne montre qu'un seul chiffre : une seule classe par exemple": "E"}'''),
        ("e", '"B", "C" or "E"', '"B"',
         r'''mistakes={"fécondé ou non : il n'y a que deux classes": "C",
          "un œuf est fécondé ou ne l'est pas : une seule réponse parmi deux": "E"}'''),
    ]),
    Paper("7.Q3", "Régions et frontières de décision", [
        ("a", "True or False", "False",
         r'''mistakes={"regarde la figure 7.3 du livre, ou le panneau (b) de la figure des trois situations de la fiche": True}'''),
        ("b", "a whole number", "3",
         r'''mistakes={"chaque classe que le classifieur peut prédire occupe au moins un morceau du plan": 2}'''),
        ("c", "True or False", "True",
         r'''mistakes={"rien n'oblige une classe à n'occuper qu'un seul morceau : imagine deux paquets d'œufs viables séparés par des quitters": False}'''),
        ("d", "True or False", "False",
         r'''mistakes={"un classifieur rend une seule classe par point, même sur la frontière : une convention tranche": True}'''),
        ("e", "True or False", "False",
         r'''mistakes={"fais tourner très légèrement la droite entre deux nuages bien séparés : sépare-t-elle encore les deux classes ?": True}'''),
    ]),
    Paper("7.Q4", "Classes qui se recouvrent : probabilités et politique de seuil", [
        ("a", "True or False", "True",
         r'''mistakes={"tout point où P ≥ 0,5 vérifie aussi P ≥ 0,2 : un œuf fécondé déjà repéré peut-il être perdu ?": False}'''),
        ("b", "True or False", "True",
         r'''mistakes={"les œufs déclarés fécondés au seuil 0,5 le restent au seuil 0,2, qu'ils le soient vraiment ou non : un faux positif peut-il disparaître ?": False}'''),
        ("c", '"A", "B" or "C"', '"A"',
         r'''mistakes={"compare les ensembles {P ≥ 0,5} et {P ≥ 0,2} : l'un contient l'autre": "B",
          "tout point de la région au seuil 0,5 y reste au seuil 0,2 : elle ne peut pas rétrécir": "C"}'''),
        ("d", "the threshold, 3 decimals", "1 / (1 + 7)",
         r'''decimals=3, mistakes={"le seuil est C_FP / (C_FP + C_FN) : il BAISSE quand les faux négatifs coûtent cher": 7 / (1 + 7),
          "le dénominateur est la SOMME des deux coûts, pas l'un des deux": 1 / 7,
          "0,5 ne convient que si les deux erreurs coûtent autant : calcule C_FP / (C_FP + C_FN)": 0.5,
          "0,2 est le seuil choisi au début de l'énoncé : calcule celui qui minimise le coût moyen": 0.2}'''),
    ]),
    Paper("7.Q5", "Cinq mesures par œuf : ce que change (ou non) la dimension", [
        ("a", "a whole number", "5",
         r'''mistakes={"une dimension par feature : la classe n'est pas une feature": 6}'''),
        ("b", "True or False", "True",
         r'''mistakes={"une distance euclidienne est la racine de la somme des carrés des écarts, quel que soit le nombre de coordonnées": False}'''),
        ("c", "True or False", "False",
         r'''mistakes={"chaque feature ajoute un terme à chaque distance et à chaque moyenne : le calcul peut-il coûter autant ?": True}'''),
        ("d", "a whole number", "4",
         r'''mistakes={"une frontière a une dimension de MOINS que l'espace (une droite dans le plan)": 5}'''),
    ]),
    Paper("7.Q6", "Un-contre-tous : combien de modèles, quelle décision ?", [
        ("a", "a whole number", "7",
         r'''mistakes={"un-contre-tous : un classifieur par CLASSE, pas un par paire": 21}'''),
        ("b", "a letter", '"B"',
         r'''mistakes={"compare 0,55 et 0,52": "C", "on prédit la classe de plus GRAND score": "D",
          "0,30 n'est pas le plus grand score": "A"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"additionne les quatre probabilités de b)": True}'''),
        ("d", "a letter", '"B"',
         r'''mistakes={"le plus grand de quatre nombres négatifs est le plus proche de 0": "C",
          "compare −1,2 et −0,4 : lequel est le plus grand ?": "A",
          "compare −0,9 et −0,4 : lequel est le plus grand ?": "D"}'''),
    ]),
    Paper("7.Q7", "Un-contre-un : duels, votes et coût", [
        ("a", "a whole number", "n_ovo(6)",
         r'''mistakes={"A-B et B-A sont le même duel : divise par 2": 30,
          "un-contre-un : un duel par PAIRE de classes, pas un par classe": 6,
          "K × K compte aussi les duels d'une classe contre elle-même, et chaque paire deux fois": 36,
          "c'est le nombre de duels pour 5 classes": 10}'''),
        ("b", "True or False", "False",
         r'''mistakes={"un duel ne voit que les échantillons de ses deux classes : les autres sont ignorés": True}'''),
        ("c", "True or False", "True",
         r'''mistakes={"calcule 3 × 2 / 2 et compare à 3": False}'''),
        ("d", "a letter", '"A"',
         r'''mistakes={"A et B sont à égalité : la règle de mylearn garde la classe de plus petit indice": "B",
          "C n'a qu'une voix": "C", "D n'a qu'une voix": "D"}'''),
        ("e", "True or False", "False",
         r'''mistakes={"développe K(K − 1)/2 et compare-le à K²/2": True}'''),
    ]),
    Paper("7.Q8", "Ce que k-means ne peut pas deviner tout seul", [
        ("a", "True or False", "False",
         r'''mistakes={"k-means est non supervisé : il ne voit que les points": True}'''),
        ("b", "a letter", '"B"',
         r'''mistakes={"k est fixé AVANT l'entraînement : il n'est pas appris": "A",
          "k est un nombre de clusters, pas une classe": "C"}'''),
        ("c", "True or False", "False",
         r'''mistakes={"k-means rend exactement k clusters": True}'''),
        ("d", "True or False", "False",
         r'''mistakes={"le résultat dépend des centres de départ : relis « minimum local »": True}'''),
        ("e", "True or False", "False",
         r'''mistakes={"que vaut l'inertie quand chaque point est seul dans son cluster ?": True}'''),
    ]),
    Paper("7.Q9", "Densité d'échantillons quand les features s'accumulent", [
        ("a", "a whole number", "20 // 4",
         r'''fractional="divise le nombre d'échantillons par le nombre de cases"'''),
        ("b", "2 decimals", "20 / 4 ** 2",
         r'''decimals=2, mistakes={"en dimension 2, il y a 4 × 4 cases, pas 4 × 2": 20 / 8}'''),
        ("c", "4 decimals", "20 / 4 ** 3",
         r'''decimals=4, mistakes={"en dimension 3, il y a 4³ = 64 cases, pas 4 × 3": 20 / 12,
          "il y a 4 cases par axe : 4³ cases, pas 3⁴": 20 / 3 ** 4,
          "arrondi trop tôt, ou trop peu de décimales : l'énoncé en demande 4": 0.31}'''),
        ("d", "True or False", "True",
         r'''mistakes={"la densité est le nombre d'échantillons divisé par le nombre de cases": False}'''),
        ("e", "True or False", "True",
         r'''mistakes={"chaque feature multiplie le nombre de cases par le nombre de cases par axe": False}'''),
    ]),
    Paper("7.Q11", "Géométrie déroutante en grande dimension", [
        ("a", "True or False", "1 - 0.97 ** 100 > 0.99",
         r'''mistakes={"calcule 1 − 0,97¹⁰⁰ et compare-le à 0,99": True}'''),
        ("b", "True or False", "math.sqrt(10) - 1 > 2",
         r'''mistakes={"relis les valeurs du livre, ou calcule √10 − 1": False}'''),
        ("c", "True or False", "True",
         r'''mistakes={"le coin a toutes ses coordonnées égales à ±1/2 : calcule sa distance au centre en dimension d": False}'''),
        ("d", "a letter", '"B"',
         r'''mistakes={"en grande dimension, le plus proche voisin s'éloigne vite : relis la concentration des distances": "A",
          "la distance au plus proche voisin ne diminue pas quand la dimension grandit": "C"}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("7.R1", "Ch. 6 : entropie d'un cluster pur et d'un cluster mélangé", [
        ("a", "in bits", "round(entropy_bits([30]))",
         r'''mistakes={"une seule espèce : il n'y a aucune incertitude": 1}'''),
        ("b", "in bits", "round(entropy_bits([20, 20]))",
         r'''fractional="l'entropie se mesure ici en bits (log2, pas ln), avec un terme par espèce"'''),
        ("c", "in bits", "entropy_bits([0.5, 0.25, 0.25])",
         r'''decimals=2, mistakes={"en bits, avec log2 (tu as obtenu des nats, avec ln)": 1.5 * math.log(2)}'''),
        ("d", "in bits, 3 decimals", "entropy_bits([0.7, 0.2, 0.1])",
         r'''decimals=3, mistakes={"en bits, avec log2 (tu as obtenu des nats, avec ln)": entropy_bits([0.7, 0.2, 0.1]) * math.log(2),
          "n'oublie pas le signe moins devant la somme": -entropy_bits([0.7, 0.2, 0.1])}'''),
        ("e", "in bits, 3 decimals", "math.log2(3)",
         r'''decimals=3, mistakes={"en bits, avec log2 (tu as obtenu des nats, avec ln)": math.log(3),
          "avec trois espèces, la distribution uniforme est [1/3, 1/3, 1/3]": 1.0}'''),
    ]),
    Paper("7.R2", "Ch. 4 : probabilité a posteriori « fécondé » par la règle de Bayes", [
        ("a", "3 decimals", "posterior(0.3, 0.08, 0.04)",
         r'''decimals=3, mistakes={"tu as oublié l'a priori π = 0,3 : multiplie chaque densité par la probabilité de sa classe": posterior(0.5, 0.08, 0.04),
          "c'est le numérateur seul : divise par la somme des deux termes": 0.3 * 0.08,
          "c'est la probabilité de NE PAS être fécondé": 1 - posterior(0.3, 0.08, 0.04)}'''),
        ("b", "True or False", "posterior(0.3, 0.08, 0.04) >= 0.5",
         r'''mistakes={"compare ta réponse de a) au seuil de 0,5": True}'''),
        ("c", "3 decimals", "posterior(0.5, 0.08, 0.04)",
         r'''decimals=3, mistakes={"c'est le résultat avec l'a priori de 30 % : refais le calcul avec 0,5 et 0,5": posterior(0.3, 0.08, 0.04)}'''),
        ("d", "3 decimals", "0.7 / 0.3",
         r'''decimals=3, mistakes={"c'est π / (1 − π) : pour une classe RARE, le rapport f1/f0 doit compenser l'a priori défavorable": 0.3 / 0.7,
          "un rapport de 1 ne donne P = 0,5 que si les deux classes sont aussi fréquentes : tiens compte de l'a priori": 1}'''),
    ]),
    Paper("7.R3", "0B : développer ‖a − b‖² avec le produit scalaire", [
        ("a", "a whole number", "int(A_R3 @ A_R3)",
         r'''fractional="‖a‖² est le CARRÉ de la norme : la somme des carrés des coordonnées, sans racine",
          mistakes={"‖a‖² est la somme des CARRÉS des coordonnées": 4}'''),
        ("b", "a whole number", "int(A_R3 @ B_R3)",
         r'''mistakes={"a · b = a1 b1 + a2 b2 + a3 b3 : attention au signe de la deuxième coordonnée de a": 5}'''),
        ("c", "a whole number", "int(B_R3 @ B_R3)",
         r'''fractional="‖b‖² est le CARRÉ de la norme : la somme des carrés des coordonnées, sans racine",
          mistakes={"‖b‖² est la somme des CARRÉS des coordonnées": 3}'''),
        ("d", "a whole number", "int((A_R3 - B_R3) @ (A_R3 - B_R3))",
         r'''fractional="on demande le CARRÉ de la distance : la somme des carrés des coordonnées de a − b, sans racine",
          mistakes={"le terme croisé vaut −2 a·b, pas −a·b": 18, "tu as oublié le terme croisé −2 a·b": 19,
          "le terme croisé est MOINS 2 a·b": 21}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("7.1", "Compter les classifieurs OvR et OvO", [
        ("a", "[N_OvR, N_OvO] for K = 3", "[3, n_ovo(3)]",
         r'''mistakes={"A-B et B-A sont le même duel : divise K(K − 1) par 2": [3, 6]}'''),
        ("b", "[N_OvR, N_OvO] for K = 10", "[10, n_ovo(10)]",
         r'''mistakes={"A-B et B-A sont le même duel : divise K(K − 1) par 2": [10, 90],
          "l'ordre demandé est [N_OvR, N_OvO]": [45, 10],
          "une classe ne se bat pas contre elle-même, et chaque paire ne compte qu'une fois": [10, 100]}'''),
        ("c", "[N_OvR, N_OvO] for K = 26", "[26, n_ovo(26)]",
         r'''mistakes={"A-B et B-A sont le même duel : divise K(K − 1) par 2": [26, 650],
          "l'ordre demandé est [N_OvR, N_OvO]": [325, 26]}'''),
        ("d", "[N_OvR, N_OvO] for K = 100", "[100, n_ovo(100)]",
         r'''mistakes={"A-B et B-A sont le même duel : divise K(K − 1) par 2": [100, 9900],
          "l'ordre demandé est [N_OvR, N_OvO]": [4950, 100],
          "K²/2 compte aussi les duels d'une classe contre elle-même : c'est K(K − 1)/2": [100, 5000]}'''),
        ("e", "the smallest K", "next(k for k in range(2, 1000) if n_ovo(k) > 10_000)",
         r'''fractional="cherche le plus petit ENTIER K tel que K(K − 1)/2 dépasse 10 000",
          mistakes={"avec 141 classes, il n'y a encore que 9 870 duels : il en faut PLUS de 10 000": 141,
          "n'oublie pas de diviser par 2 : il faut K(K − 1)/2 > 10 000, c'est-à-dire K(K − 1) > 20 000": 101}'''),
        ("f", "total number of training examples, one-versus-rest", "10 * 50_000",
         r'''mistakes={"chaque classifieur voit TOUTES les images, et il y a 10 classifieurs": 50_000,
          "chaque classifieur de l'un-contre-tous voit TOUTES les images, pas seulement celles de deux chiffres": 45 * 10_000}'''),
        ("g", "total number of training examples, one-versus-one", "n_ovo(10) * 2 * 5_000",
         r'''mistakes={"un duel ne voit que les images de ses DEUX chiffres (10 000), pas toutes": 45 * 50_000,
          "un duel ne voit que les images de ses deux chiffres : compte les duels, puis les images de chacun": 10 * 50_000,
          "chaque duel voit les images de ses DEUX chiffres : 2 × 5 000": 45 * 5_000,
          "A-B et B-A sont le même duel : il y a K(K − 1)/2 duels, pas K(K − 1)": 90 * 10_000}'''),
        ("h", "how many times cheaper, 1 decimal", "(10 * 50_000 ** 2) / (n_ovo(10) * 10_000 ** 2)",
         r'''decimals=1, mistakes={"c'est le rapport inverse : divise le coût de l'un-contre-tous par celui de l'un-contre-un": (n_ovo(10) * 10_000 ** 2) / (10 * 50_000 ** 2),
          "le coût d'un modèle est m² : élève au carré le nombre d'exemples de CHAQUE modèle avant d'additionner": (10 * 50_000) / (n_ovo(10) * 10_000)}'''),
    ]),
    Paper("7.2", "Dépouiller les votes d'un un-contre-un à quatre classes", [
        ("a", "votes [A, B, C, D], first point", "votes(WINNERS_72A).tolist()",
         r'''mistakes={"tu as compté les défaites : une voix par duel GAGNÉ": (3 - votes(WINNERS_72A)).tolist()}'''),
        ("b", "a letter", '"B"',
         r'''mistakes={"compte les voix de a) : D n'en a pas le plus": "D", "A ne gagne qu'un duel": "A",
          "C ne gagne aucun duel": "C"}'''),
        ("c", "a whole number", "int(votes(WINNERS_72A).sum())",
         r'''mistakes={"chaque duel donne UNE voix, à son vainqueur seulement": 12}'''),
        ("d", "votes [A, B, C, D], second point", "votes(WINNERS_72B).tolist()",
         r'''mistakes={"tu as compté les défaites : une voix par duel GAGNÉ": (3 - votes(WINNERS_72B)).tolist()}'''),
        ("e", "a letter", '"A"',
         r'''mistakes={"A et C sont à égalité : la règle de mylearn garde la classe de plus petit INDICE": "C",
          "B n'a qu'une voix": "B", "D n'a qu'une voix": "D"}'''),
        ("f", "a letter", '"C"',
         r'''mistakes={"regarde qui a gagné le duel A-C": "A", "B n'a qu'une voix": "B", "D n'a qu'une voix": "D"}'''),
        ("g", "True or False", "False",
         r'''mistakes={"une classe qui gagne ses K − 1 duels a K − 1 voix ; combien au plus pour chacune des autres, qui ont toutes perdu contre elle ?": True}'''),
        ("h", "True or False", "False",
         r'''mistakes={"6 voix pour 4 classes : combien chacune en aurait-elle en cas d'égalité parfaite ?": True}'''),
    ]),
    Paper("7.3", "Une itération de k-means à la main", [
        ("a", "cluster (1 or 2) of P1 to P6", "(L1_73 + 1).tolist()",
         r'''mistakes={"numérote les clusters 1 et 2, comme l'énoncé": L1_73.tolist(),
          "P2 est à 1 de c1 et à 5 de c2 (distances au carré)": [1, 2, 2, 2, 2, 2],
          "numérote les clusters comme l'énoncé : le cluster 1 est celui du centre de départ c1 = P1": (2 - L1_73).tolist()}'''),
        ("b", "the inertia with the starting centres", "round(inertia(P_73, C0_73, L1_73))",
         r'''fractional="l'inertie de b) se calcule avec les centres de DÉPART (P1 et P3) et des distances AU CARRÉ : avec ces coordonnées entières, elle est entière",
          mistakes={"l'inertie est une SOMME, pas une moyenne": round(inertia(P_73, C0_73, L1_73) / 6)}'''),
        ("c", "the two new centres [[x1, y1], [x2, y2]], 2 decimals", "C1_73.tolist()",
         r'''decimals=2, mistakes={"c2 est la moyenne de P3, P4, P5 ET P6 : P3 est encore dans le cluster 2": [[1.5, 1.0], [6.0, 13 / 3]],
          "dans l'ordre [c1, c2] : c1 est le centre du cluster de P1": C1_73[::-1].tolist()}'''),
        ("d", "2 decimals", "inertia(P_73, C1_73, L1_73)",
         r'''decimals=2, mistakes={"garde l'affectation de a) : P3 est encore dans le cluster 2 à cette étape": inertia(P_73, C1_73, L2_73)}'''),
        ("e", "the number of the point (1 to 6)", "int(np.flatnonzero(L2_73 != L1_73)[0]) + 1",
         r'''mistakes={"recalcule les distances de P4 aux deux centres de c) : il reste bien plus près de c2": 4}'''),
        ("f", "the centres after the second update, 3 decimals", "C2_73.tolist()",
         r'''decimals=3, mistakes={"ce sont les centres de c) : refais la mise à jour avec la nouvelle affectation": C1_73.tolist(),
          "dans l'ordre [c1, c2] : c1 est le centre du cluster de P1": C2_73[::-1].tolist()}'''),
        ("g", "the final inertia, 2 decimals", "inertia(P_73, C2_73, L2_73)",
         r'''decimals=2, mistakes={"c'est l'inertie de d) : recalcule-la avec les centres de f)": inertia(P_73, C1_73, L1_73),
          "c'est l'inertie avant la dernière mise à jour : recalcule-la avec les centres de f)": inertia(P_73, C1_73, L2_73)}'''),
        ("h", "True or False", "bool((L3_73 != L2_73).any())",
         r'''mistakes={"recalcule les distances de P3 et de P4 aux centres de f) : changent-ils encore de cluster ?": True}'''),
        ("i", "True or False", "True",
         r'''mistakes={"relis les inerties de b), d) et g) : ont-elles augmenté ?": False}'''),
    ]),
    Paper("7.4", "Densité d'échantillons et nombre d'œufs nécessaires", [
        ("a", "[d = 1, d = 2, d = 3, d = 4], 3 decimals", "[N_74 / B_74 ** d for d in range(1, 5)]",
         r'''decimals=3, mistakes={"en dimension d, il y a 6^d cases, pas 6 × d": [N_74 / (B_74 * d) for d in range(1, 5)]}'''),
        ("b", "a whole number", "2 * B_74 ** 3",
         r'''mistakes={"n = ρ × b^d, pas ρ × b × d": 2 * B_74 * 3, "multiplie le nombre de cases par la densité 2": B_74 ** 3}'''),
        ("c", "a whole number", "2 * B_74 ** 6",
         r'''mistakes={"multiplie le nombre de cases par la densité 2": B_74 ** 6, "n = ρ × b^d, pas ρ × b × d": 2 * B_74 * 6}'''),
        ("d", "a whole number", "2 * 10 ** 6",
         r'''mistakes={"multiplie le nombre de cases par la densité 2": 10 ** 6}'''),
        ("e", "the smallest dimension", "next(d for d in range(1, 30) if 10 ** 6 / B_74 ** d < 1)",
         r'''fractional="une dimension est un nombre entier : prends le plus petit entier d tel que 10⁶ / 6^d < 1",
          mistakes={"en dimension 7, la densité vaut encore 10⁶ / 6⁷ ≈ 3,6 : elle n'est pas sous 1": 7}'''),
        ("f", "3 decimals", "EMPTY_74",
         r'''decimals=3, mistakes={"c'est la probabilité qu'il soit OCCUPÉ : on demande celle qu'il soit vide": 1 - EMPTY_74,
          "il y a 10 œufs indépendants : chacun doit éviter le cube": 124 / 125,
          "1 − 0,08 ? la densité n'est pas une probabilité : chaque œuf doit éviter ce cube, indépendamment des autres (si tu as seulement arrondi, donne 3 décimales)": 1 - 10 / 125}'''),
        ("g", "2 decimals", "125 * (1 - EMPTY_74)",
         r'''decimals=2, mistakes={"deux œufs peuvent tomber dans le même cube : multiplie 125 par la probabilité qu'un cube soit occupé": 10.0,
          "c'est le nombre moyen de cubes VIDES": 125 * EMPTY_74,
          "tu as utilisé la valeur arrondie de f) : garde sa valeur exacte jusqu'au bout": 9.62,
          "tu as arrondi f) avant de t'en servir : garde toutes ses décimales jusqu'à la fin du calcul": 9.63}'''),
    ]),
    Paper("7.5", "Le rayon de l'hyper-orange : r(d) = √d − 1", [
        ("a", "r(2), 3 decimals", "math.sqrt(2) - 1",
         r'''decimals=3, mistakes={"le rayon de l'orange est √d − 1 : il faut retirer le rayon d'un ballon": math.sqrt(2),
          "c'est le diamètre de l'orange : on demande son rayon": 2 * (math.sqrt(2) - 1)}'''),
        ("b", "r(3), 3 decimals", "math.sqrt(3) - 1",
         r'''decimals=3, mistakes={"le rayon de l'orange est √d − 1 : il faut retirer le rayon d'un ballon": math.sqrt(3),
          "c'est le diamètre de l'orange : on demande son rayon": 2 * (math.sqrt(3) - 1)}'''),
        ("c", "a whole number", "2 ** 12",
         r'''mistakes={"un ballon par coin : chaque coordonnée d'un coin vaut −1 ou +1, indépendamment des autres : combien de choix en tout ?": 2 * 12,
          "12² ne compte que des paires : chaque coordonnée d'un coin vaut −1 ou +1, indépendamment des autres : combien de choix en tout ?": 12 ** 2}'''),
        ("d", "the smallest dimension", "next(d for d in range(1, 100) if math.sqrt(d) - 1 >= 1.5)",
         r'''fractional="une dimension est un nombre entier : prends le plus petit entier qui convient",
          mistakes={"en dimension 6, r = √6 − 1 ≈ 1,45 : pas encore 1,5": 6,
          "r = √d − 1 : n'oublie pas de retirer le rayon d'un ballon": 3}'''),
        ("e", "the smallest dimension", "next(d for d in range(1, 100) if math.sqrt(d) - 1 > 2.5)",
         r'''fractional="une dimension est un nombre entier : prends le plus petit entier qui convient",
          mistakes={"en dimension 12, r = √12 − 1 ≈ 2,46 : pas encore 2,5": 12,
          "r = √d − 1 : n'oublie pas de retirer le rayon d'un ballon": 7,
          "c'est le seuil r > 2 du livre (l'orange sort de la boîte) : ici, on demande r > 2,5": 10}'''),
        ("f", "r(50), 3 decimals", "math.sqrt(50) - 1",
         r'''decimals=3, mistakes={"le rayon de l'orange est √d − 1 : il faut retirer le rayon d'un ballon": math.sqrt(50),
          "c'est le diamètre de l'orange : on demande son rayon": 2 * (math.sqrt(50) - 1)}'''),
    ]),
    Paper("7.6", "Boule dans un cube : rapport des volumes par récurrence", [
        ("a", "V4, 3 decimals", "V[4]",
         r'''decimals=3, mistakes={"la récurrence saute de deux dimensions : V4 = (2π/4) V2, pas (2π/4) V3": 2 * math.pi / 4 * V[3]}'''),
        ("b", "V5, 3 decimals", "V[5]",
         r'''decimals=3, mistakes={"la récurrence saute de deux dimensions : V5 = (2π/5) V3, pas (2π/5) V4": 2 * math.pi / 5 * V[4]}'''),
        ("c", "q4, 3 decimals", "V[4] / 2 ** 4",
         r'''decimals=3, mistakes={"le cube de côté 2 a pour volume 2⁴ = 16 en dimension 4": V[4] / 4,
          "le cube de côté 2 a pour volume 2^d : ici 2⁴": V[4] / 2 ** 5,
          "la boule de rayon 1 tient dans le cube de côté 2, pas 1 : divise par le volume de ce cube": V[4]}'''),
        ("d", "q10 in percent, 3 decimals", "100 * V[10] / 2 ** 10",
         r'''decimals=3, mistakes={"c'est une proportion : l'énoncé demande un pourcentage, avec 3 décimales": 0.002,
          "multiplie par 100 : l'énoncé demande un pourcentage, pas une proportion": 0.003,
          "la boule de rayon 1 tient dans le cube de côté 2, pas 1 : divise par le volume de ce cube": V[10],
          "le cube qui contient la boule de rayon 1 a pour côté 2 : divise V10 par son volume, puis multiplie par 100": 100 * V[10]}'''),
        ("e", "a dimension", "max(range(1, 30), key=lambda d: V[d])",
         r'''mistakes={"tu n'as peut-être calculé que les dimensions paires : la récurrence saute de deux en deux, calcule aussi V3, V5 et V7": 6,
          "tu t'es peut-être arrêté trop tôt : calcule V1 à V8 avant de chercher le maximum": 4}'''),
        ("f", "3 decimals", "1 - 0.95 ** 100",
         r'''decimals=3, mistakes={"c'est la part du cœur, l'intérieur de rayon 0,95 : on demande celle de la peau": 0.95 ** 100,
          "le volume est proportionnel à r^d : la peau contient bien plus que 5 % du volume (relis « Tout est dans l'écorce »)": 0.05,
          "c'est la valeur de la fiche pour une peau de 10 % en dimension 50 : ici, la peau fait 5 % et d = 100": 0.995}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 7.1 à 7.4, ∂ 7.5 et 7.6)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.375` (en "
               "Python, le séparateur décimal est un **point** ; `0,375` sans guillemets serait un couple de deux "
               "nombres) ; un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets "
               "(`\"D\"`) ; plusieurs nombres : une liste (`[2, 5]`). Arrondis comme l'énoncé le demande, et "
               "seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. "
               "Les questions « dans ta copie », le quiz Q10, ∂ 7.7, la réflexion et l'entretien se corrigent "
               "avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 7.11 to 7.31): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "7.11 à 7.16", "frontières, distances, centroïde le plus proche, carte de probabilité, un-contre-tous avec "
                         "des centroïdes"),
    ("B", "7.17 à 7.21", "k-means, DBSCAN et HDBSCAN avec scikit-learn, puis la grande dimension"),
    ("C", "7.22 à 7.24", "un-contre-tous et un-contre-un génériques dans mylearn.multiclass"),
    ("D", "7.25 à 7.31", "k-means de zéro dans mylearn.cluster, silhouette, choix de k, phénomène de Hughes, défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 7.Q1 à 7.Q11, 7.R1 à 7.R3, 7.1 à 7.6 | vérifier tes réponses courtes | 🧠 🔁 ✏️ ∂ | ★ à ★★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")          # a "|" in a title must not split the table row
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |" for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# 7 · Classification — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch07_classification/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 7`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 7 · Classification — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Lire des régions et des frontières de décision, et régler un seuil d'après le coût des erreurs.\n"
               "- Traiter plusieurs classes avec des classifieurs binaires : un-contre-tous et un-contre-un.\n"
               "- Programmer le centroïde le plus proche et k-means (Lloyd, k-means++), et choisir $k$ avec "
               "l'inertie et la silhouette ; savoir quand préférer DBSCAN ou HDBSCAN.\n"
               "- Mesurer ce que la dimension fait aux densités et aux distances.\n\n"
               "**Rappel express.** $N_{\\text{OvR}} = K$, $N_{\\text{OvO}} = \\frac{K(K-1)}{2}$ ; seuil de coût "
               "minimal $t^* = \\frac{C_{FP}}{C_{FP} + C_{FN}}$ ; "
               "$\\lVert \\mathbf{a} - \\mathbf{b} \\rVert^2 = \\lVert \\mathbf{a} \\rVert^2 - 2\\,\\mathbf{a} \\cdot "
               "\\mathbf{b} + \\lVert \\mathbf{b} \\rVert^2$ ; inertie $J = \\sum_i \\lVert \\mathbf{x}_i - "
               "\\boldsymbol{\\mu}_{c_i} \\rVert^2$ ; silhouette $s = \\frac{b - a}{\\max(a, b)}$ ; densité "
               "$\\rho = \\frac{n}{b^d}$ ; peau d'une boule $1 - (1 - \\varepsilon)^d$. En Python : "
               "`sklearn.cluster.KMeans`, `DBSCAN`, `HDBSCAN`, `sklearn.metrics.silhouette_score`, "
               "`sklearn.multiclass`, `rng.choice(n, p=w / w.sum())`.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 7.11 à 7.31 (parties A à D, tes librairies `mylearn.cluster` et "
             "`mylearn.multiclass`) seront ajoutés à la prochaine session de génération ; relance alors "
             "`python tools/start_chapter.py 7` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu choisir un seuil de décision d'après le coût des deux erreurs, et dire qui doit fixer "
               "ces coûts ?\n"
               "2. Sais-tu compter et dépouiller les votes d'un un-contre-un, et dire quand le préférer à "
               "l'un-contre-tous ?\n"
               "3. Sais-tu dérouler k-means à la main, et expliquer pourquoi la densité des exemples s'effondre "
               "quand on ajoute des features ?\n\n"
               "**Pour aller plus loin** : les guides « Clustering » et « Multiclass and multioutput algorithms » "
               "de scikit-learn, cités dans la fiche. La suite : le ch. 8 (entraînement et test), où un jeu de "
               "validation fixe les seuils et les hyperparamètres comme $k$." + later)]


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
