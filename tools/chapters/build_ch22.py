#!/usr/bin/env python
"""Build the two notebooks of chapter 22 from a single source (used by Claude).

    python tools/chapters/build_ch22.py
    python tools/run_all_notebooks.py chapitres/ch22_rnn/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch22_rnn/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
22.1 to 22.5 and of the 🧮 estimate 22.8 (02_exercices.md). The proofs ∂ 22.6 and 22.7, the reflection
(🗣️ 22.9, ⚖️ 22.10, 📄 22.11) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 22.12 to 22.29): the libraries mylearn.sequences and mylearn.rnn checked against PyTorch,
the vanishing gradient measured with autograd, the sunspot forecast against persistence, the Holmes generator and
the CNN-LSTM. The 20-epoch history of 📈 22.26 comes from tools/chapters/history_ch22.py (history_ch22.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402,F401

CHAPTER = "22"
FOLDER = "chapitres/ch22_rnn"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def n_windows(T, w, h=1, s=1):
    """Number of windows of w values whose target lies h steps after the window (starts s apart)."""
    return (T - w - h) // s + 1


def n_recurrent(d, h, blocks):
    """Parameters of one PyTorch recurrent layer: blocks of H columns, two bias vectors."""
    return blocks * (d * h + h * h + 2 * h)


def softmax_t(z, t):
    e = np.exp((np.asarray(z, float) - max(z)) / t)
    return e / e.sum()


# Ex 22.2: a scalar RNN cell h_t = tanh(u x_t + w h_{t-1} + b), y_t = v h_t
def scalar_rnn(xs, u=1.0, w=0.5, b=0.0, h=0.0):
    states = []
    for x in xs:
        h = math.tanh(u * x + w * h + b)
        states.append(h)
    return states


H_222 = scalar_rnn([2, 0, -1])
H_REV_222 = scalar_rnn([-1, 0, 2])

# Ex 22.4: one LSTM step, gates in the order i, f, g, o
W_IH_224, W_HH_224, B_224 = [-0.5, 1, 1, -1], [1, -2, 0, 3], [0, 1, 0, 0]
A_224 = [W_IH_224[k] * 1 + W_HH_224[k] * 0.5 + B_224[k] for k in range(4)]
I_224, F_224, G_224, O_224 = sigmoid(A_224[0]), sigmoid(A_224[1]), math.tanh(A_224[2]), sigmoid(A_224[3])
C_224 = F_224 * 2 + I_224 * G_224
H_224 = O_224 * math.tanh(C_224)
C_OPEN_224 = sigmoid(A_224[1] + 20) * 2 + sigmoid(A_224[0] - 20) * G_224

# Ex 22.5: the logits of « k », « e », « a » after « Sherloc »
Z_225 = [3.0, 1.0, 0.5]

# Ex 22.8: the CNN-LSTM of the book (128 x 128 frames, 8 filters 5 x 5, stride 2, padding 2, LSTM 128, 9 classes)
SIDE_228 = (128 + 2 * 2 - 5) // 2 + 1
D_228 = SIDE_228 * SIDE_228 * 8
CONV_228 = 8 * (5 * 5 + 1)
LSTM_228 = n_recurrent(D_228, 128, 4)
DENSE_228 = 128 * 9 + 9
MACS_228 = D_228 * 4 * 128'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
SHAPE = 'a shape as a string, e.g. "(3, 5, 1)"'
WHOLE = "a whole number"
DEC4 = "a number, 4 decimals"

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("22.Q1", "Pourquoi un réseau dense ignore l'ordre", [
        ("a", TF, "True",
         r'''mistakes={"une permutation FIXE des entrées : le réseau peut-il permuter ses poids de la même façon ?": False}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"une couche dense ne trie rien : elle multiplie chaque entrée par son poids": "A",
          "la loss dépend bien des entrées, à travers les prédictions": "C",
          "c'est le RNN qui garde un état, pas la couche dense": "D"}'''),
        ("c", TF, "False",
         r'''mistakes={"si l'ordre change d'une fenêtre à l'autre, une entrée porte-t-elle encore toujours la même position ?": True}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"des nombres sans lien entre eux ne forment pas une séquence": "A",
          "la quantité de données ne fait pas une séquence": "C", "les labels ne font pas une séquence": "D"}'''),
        ("e", TF, "True",
         r'''mistakes={"écris les positions des fenêtres qui démarrent en 0 et en 1": False}'''),
    ]),
    Paper("22.Q2", "État ou poids : qu'est-ce qui change en production ?", [
        ("a", TF, "False", r'''mistakes={"sans optimiseur, qu'est-ce qui modifierait les poids ?": True}'''),
        ("b", TF, "True", r'''mistakes={"relis le §22.3 : que calcule la cellule à chaque nouvelle entrée ?": False}'''),
        ("c", TF, "True",
         r'''mistakes={"mêmes poids, mêmes entrées, même état de départ : qu'est-ce qui pourrait différer ?": False}'''),
        ("d", TF, "False",
         r'''mistakes={"après la phrase A, l'état de départ de B est-il encore nul ?": True}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"qu'est-ce qui est recalculé à chaque entrée ?": "A",
          "les gradients ne servent qu'à l'entraînement": "C", "le learning rate est un hyperparamètre fixe": "D"}'''),
    ]),
    Paper("22.Q3", "Délai, déroulement et pas de temps", [
        ("a", LETTER, '"B"',
         r'''mistakes={"l'activation est dans la cellule, pas dans le petit carré": "A",
          "le biais est un poids de la cellule": "C", "le dropout n'apparaît pas dans ce schéma": "D"}'''),
        ("b", WHOLE, "1", r'''mistakes={"les copies du schéma déroulé ont-elles leurs propres poids ?": 12}'''),
        ("c", WHOLE, "12", r'''mistakes={"une application de la cellule par pas de temps": 1}'''),
        ("d", TF, "False", r'''mistakes={"relis le paragraphe « Enroulé et déroulé » : les copies partagent-elles leurs poids ?": True}'''),
        ("e", WHOLE, "2",
         r'''mistakes={"une matrice pour l'entrée, une pour l'état précédent": 1,
          "l'énoncé ne compte pas les biais": 4}'''),
    ]),
    Paper("22.Q4", "Entrée one-hot, taille de l'état et sens de ses valeurs", [
        ("a", WHOLE, "3000", r'''mistakes={"la longueur du vecteur, pas son nombre de 1": 1}'''),
        ("b", WHOLE, "1", r'''mistakes={"un vecteur one-hot n'a qu'un seul 1": 3000}'''),
        ("c", TF, "False", r'''mistakes={"W_ih a la forme (D, H) : D et H sont-ils liés ?": True}'''),
        ("d", WHOLE, "30 * 64",
         r'''mistakes={"un poids par couple (entrée, valeur d'état) : un produit": 30 + 64,
          "c'est la taille de W_hh, (H, H)": 64 * 64, "sans les biais": 30 * 64 + 64}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"personne ne fixe le contenu de l'état à l'avance": "A",
          "l'état résume ce qui a été lu, il ne range pas les caractères un par un": "C",
          "la loss n'est pas stockée dans l'état": "D"}'''),
    ]),
    Paper("22.Q5", "(échantillons, pas de temps, features) : la bonne organisation", [
        ("a", SHAPE, '"(10, 6, 2)"',
         r'''mistakes={"relis l'ordre demandé : (N, T, D), le batch d'abord": "(6, 10, 2)",
          "(N, T, D) : les pas de temps avant les features": "(10, 2, 6)",
          "les nuits ne sont pas mesurées : une séquence par journée": "(1, 60, 2)"}'''),
        ("b", SHAPE, '"(6, 10, 2)"',
         r'''mistakes={"sans batch_first, quel axe vient en premier ?": "(10, 6, 2)",
          "(T, N, D) : les features en dernier": "(6, 2, 10)"}'''),
        ("c", TF, "False",
         r'''mistakes={"reshape relit les nombres dans le même ordre : essaie sur np.arange(12).reshape(2, 3, 2)": True}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"les nuits manquent : 18 h ne précède pas directement 8 h": "A",
          "une mesure seule n'est pas une séquence": "C", "les mesures successives sont des pas de temps, pas des features": "D"}'''),
        ("e", WHOLE, "1", r'''mistakes={"le deuxième nombre de (N, T, D)": 6}'''),
    ]),
    Paper("22.Q6", "BPTT, gradients évanescents et explosifs, dépendances longues", [
        ("a", LETTER, '"B"',
         r'''mistakes={"la BPTT calcule le gradient ; l'optimiseur s'en sert ensuite": "A",
          "l'initialisation précède l'entraînement": "C", "le dropout est une régularisation": "D"}'''),
        ("b", LETTER, '"C"',
         r'''mistakes={"0,9⁵⁰ = e^{50 ln 0,9}, avec ln 0,9 ≈ −0,105": "A", "calcule e^{−5,3}": "B",
          "recalcule e^{−5,3} : tu t'es trompé d'une puissance de 10": "D"}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"1,1⁵⁰ = e^{50 ln 1,1}, avec ln 1,1 ≈ 0,095": "A", "calcule e^{4,8}": "B",
          "recalcule e^{4,8} : tu t'es trompé d'une puissance de 10": "D"}'''),
        ("d", TF, "False", r'''mistakes={"le clipping agrandit-il un gradient trop petit ?": True}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"il s'agit de la séquence, pas des poids": "A", "il s'agit d'une entrée lointaine": "C",
          "relis le paragraphe « Les dépendances à long terme » du §22.6 : mémoire ou apprentissage ?": "D"}'''),
    ]),
    Paper("22.Q7", "Une porte, c'est une multiplication", [
        ("a", WHOLE, "20",
         r'''mistakes={"multiplie par 0,25, ne divise pas": 320, "une porte multiplie la valeur, elle ne la remplace pas": 80}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"la tanh va de −1 à 1": "A", "la ReLU n'est pas bornée": "C",
          "la softmax normalise un vecteur entier": "D"}'''),
        ("c", TF, "False", r'''mistakes={"une porte est entre 0 et 1 : que fait-elle au signe ?": True}'''),
        ("d", TF, "False", r'''mistakes={"un facteur inférieur à 1 peut-il agrandir ?": True}'''),
        ("e", LETTER, '"A"',
         r'''mistakes={"le learning rate n'intervient qu'à l'entraînement": "B",
          "la loss n'est connue qu'après la sortie": "C", "relis l'équation d'une porte : rien n'y est tiré au hasard": "D"}'''),
    ]),
    Paper("22.Q8", "Oublier, se souvenir, sélectionner", [
        ("a", LETTER, '"B"',
         r'''mistakes={"« se souvenir », c'est faire entrer du neuf dans la mémoire": "A",
          "relis ce que fait chaque porte au §22.7.1": "C", "le candidat est la valeur proposée, pas la porte": "D"}'''),
        ("b", LETTER, '"C"',
         r'''mistakes={"« sélectionner » choisit ce que voit le reste du réseau": "A",
          "« se souvenir » et « sélectionner » sont deux portes différentes": "B", "le candidat n'est pas une porte": "D"}'''),
        ("c", TF, "False", r'''mistakes={"relis l'histoire des portes au §22.7.2 : 1997 ou 2000 ?": True}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"avec f ≈ 0, la mémoire est effacée": "A", "avec i ≈ 1, on ajoute du neuf à chaque pas": "C",
          "la porte de sortie ne touche pas la mémoire": "D"}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"le bloc 0 est i : compte à partir de 0": "A", "i, f, g, o : le bloc 1 est f": "B",
          "i, f, g, o : le bloc 3 est o": "D"}'''),
    ]),
    Paper("22.Q9", "LSTM ou GRU ?", [
        ("a", WHOLE, "3",
         r'''mistakes={"4 blocs, c'est le LSTM": 4, "deux portes, plus le candidat n": 2}'''),
        ("b", TF, "False", r'''mistakes={"cherche une variable c_t dans les équations du GRU": True}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"compare le nombre de blocs de poids des deux cellules": "A", "compare le nombre de blocs de poids des deux cellules": "C"}'''),
        ("d", TF, "False", r'''mistakes={"relis le paragraphe « LSTM ou GRU ? »": True}'''),
    ]),
    Paper("22.Q10", "One-to-many, many-to-one, many-to-many", [
        ("a", LETTER, '"B"',
         r'''mistakes={"une critique entière, un seul verdict": "A", "un verdict pour toute la critique, pas par mot": "C",
          "il n'y a qu'une sortie": "D"}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"une seule entrée, une séquence en sortie": "B", "une seule note en entrée": "C",
          "une seule note en entrée": "D"}'''),
        ("c", LETTER, '"D"',
         r'''mistakes={"une phrase entière en entrée": "A", "une phrase entière en sortie": "B",
          "peut-on traduire le premier mot sans avoir lu la fin de la phrase ?": "C"}'''),
        ("d", LETTER, '"C"',
         r'''mistakes={"une vidéo en entrée": "A", "un label par image, pas un pour toute la vidéo": "B",
          "chaque image a son label au même instant : pas besoin d'attendre la fin": "D"}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"40 caractères en entrée": "A", "une seule sortie par fenêtre": "C", "une seule sortie par fenêtre": "D"}'''),
    ]),
    Paper("22.Q11", "RNN profond et bidirectionnel : quand et à quel prix ?", [
        ("a", TF, "False", r'''mistakes={"le sens arrière a besoin de la fin de la séquence : l'a-t-on en temps réel ?": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"une couche RNN lit une séquence, pas une valeur": "A", "les couches ne s'échangent pas leurs poids": "C",
          "une couche reçoit des valeurs de la couche du dessous, à chaque pas": "D"}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"une somme garderait H valeurs par pas, or output en a 2H": "A",
          "une moyenne garderait H valeurs par pas, or output en a 2H": "B", "un produit garderait H valeurs par pas": "D"}'''),
        ("d", TF, "True", r'''mistakes={"compte les paramètres d'une couche bidirectionnelle (weight_ih_l0_reverse…)": False}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"chaque couche et chaque sens ont leurs propres poids": "A", "il s'entraîne toujours par BPTT": "C",
          "rien ne limite la longueur des séquences": "D"}'''),
    ]),
    Paper("22.Q12", "🕰️ Transformers, RNN légers et modèles à espace d'états", [
        ("a", WHOLE, "2017", r'''mistakes={"1997, c'est le LSTM": 1997, "2014, c'est le GRU": 2014}'''),
        ("b", TF, "True", r'''mistakes={"un RNN peut-il calculer h_t avant h_{t−1} ?": False}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"les grands modèles de langage sont des Transformers": "A",
          "pour de très longs textes, l'attention l'emporte": "C", "la génération d'images n'est pas une affaire de RNN": "D"}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"un SSM est défini par sa récurrence linéaire sur un état, pas par des convolutions : relis la section « Après les RNN »": "A", "un SSM est un réseau de neurones": "C",
          "un modèle à espace d'états a, par définition, un état": "D"}'''),
        ("e", TF, "True", r'''mistakes={"relis la première règle de la section sur les séries temporelles": False}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("22.R1", "Une convolution 1D peut-elle lire une séquence ?", [
        ("a", WHOLE, "20 - 4 + 1", r'''mistakes={"n − k + 1 : n'oublie pas le + 1": 16, "sans padding, la sortie raccourcit": 20}'''),
        ("b", TF, "True", r'''mistakes={"relis le partage de poids (ch. 21)": False}'''),
        ("c", WHOLE, "3 + 2", r'''mistakes={"chaque couche ajoute k − 1 = 2": 6, "3 + 2, pas 3 × 3": 9}'''),
        ("d", WHOLE, "500",
         r'''mistakes={"écris le champ récepteur de n couches, puis cherche le plus petit n qui convient": 1000,
          "« au moins 1 001 valeurs » : relis l'inégalité, ≥ et non >": 501, "combien chaque couche de noyau 3 ajoute-t-elle au champ ?": 334}'''),
    ]),
    Paper("22.R2", "Gradient clipping et optimiseurs adaptatifs", [
        ("a", WHOLE, "5", r'''mistakes={"la norme euclidienne, pas la somme": 7}'''),
        ("b", "a number, 2 decimals", "3 / math.hypot(3, 4)",
         r'''decimals=2, mistakes={"la norme dépasse 1 : il faut réduire le gradient": 3, "divise par la norme, pas par 4": 0.75}'''),
        ("c", TF, "False", r'''mistakes={"on multiplie tout le vecteur par un même nombre positif": True}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"RMSprop regarde les gradients, pas les poids": "B", "le pas ne dépend pas de l'epoch": "C",
          "le learning rate est le numérateur du pas, pas ce par quoi on divise": "D"}'''),
        ("e", TF, "True", r'''mistakes={"écris le premier pas d'Adam : g apparaît en haut et en bas": False}'''),
    ]),
    Paper("22.R3", "Ajuster le scaler sur le train seulement", [
        ("a", TF, "False", r'''mistakes={"le test doit venir après l'entraînement, comme en production": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"ajuster sur les 100 valeurs fait entrer le test dans le scaler": "A",
          "le scaler s'ajuste sur l'entraînement": "C", "un seul scaler pour toute la série": "D"}'''),
        ("c", "a number, 2 decimals", "(250 - 0) / (200 - 0)",
         r'''decimals=2, mistakes={"le scaler ne coupe pas les valeurs à 1": 1.0, "(x − min)/(max − min), pas l'inverse": 0.8}'''),
        ("d", TF, "False", r'''mistakes={"en production, réajusterais-tu le scaler sur chaque nouvelle valeur ?": True}'''),
        ("e", 'a word as a string, e.g. "relu"', '"transform"',
         r'''mistakes={"on ne réajuste pas le scaler sur le test": "fit_transform", "fit ajuste ; il faut appliquer": "fit"}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("22.1", "Combien de fenêtres ? Températures, taches solaires et Holmes", [
        ("a", WHOLE, "n_windows(24, 4, 1, 1)",
         r'''mistakes={"la cible doit exister : n'oublie pas l'horizon": 21, "n'oublie pas le + 1": 19}'''),
        ("b", WHOLE, "n_windows(24, 4, 1, 2)", r'''mistakes={"n'oublie pas le + 1 de la formule": 9}'''),
        ("c", WHOLE, "(2024 - 1749 + 1) * 12",
         r'''mistakes={"de 1749 à 2024 inclus, il y a 276 années": 275 * 12, "en mois, pas en années": 276}'''),
        ("d", WHOLE, "n_windows(3312, 120, 6, 1)",
         r'''mistakes={"l'horizon h = 6 compte": 3193, "n'oublie pas le + 1": 3186}'''),
        ("e", WHOLE, "n_windows(3312 - 600, 120, 6, 1)",
         r'''mistakes={"la cible aussi doit être avant le test : l'horizon compte": 2593, "en fenêtres, pas en mois": 2712}'''),
        ("f", WHOLE, "n_windows(575_000, 40, 1, 3)",
         r'''mistakes={"une fenêtre tous les 3 caractères": 574_960, "n'oublie pas le + 1": 191_653}'''),
        ("g", LETTER, '"C"',
         r'''mistakes={"N ≈ T/s : inverse la formule, T ≈ N × s": "A", "chaque fenêtre démarre 3 caractères après la précédente": "B",
          "recompte les zéros : N × s": "D"}'''),
    ]),
    Paper("22.2", "Dérouler à la main une cellule RNN scalaire sur trois pas", [
        ("a", DEC4, "H_222[0]", r'''decimals=4, mistakes={"applique la tanh": 2.0}'''),
        ("b", DEC4, "H_222[1]", r'''decimals=4, mistakes={"x₂ = 0, mais l'état précédent compte": 0.0}'''),
        ("c", DEC4, "H_222[2]", r'''decimals=4, mistakes={"ajoute w h₂ à u x₃": math.tanh(-1)}'''),
        ("d", DEC4, "2 * H_222[2]", r'''decimals=4, mistakes={"y₃ = v h₃ avec v = 2": H_222[2]}'''),
        ("e", DEC4, "H_REV_222[2]", r'''decimals=4, mistakes={"n'oublie pas l'état h₂ dans le dernier pas": math.tanh(2)}'''),
        ("f", "a number, 2 decimals", "0.5 ** 2 * 1",
         r'''decimals=2, mistakes={"compte combien de fois x₁ est multiplié par w jusqu'au pas 3": 0.5, "x₁ entre au pas 1 : combien de pas reste-t-il ensuite ?": 0.125}'''),
    ]),
    Paper("22.3", "Compter les poids : la cellule à 69 poids du livre, puis nn.RNN et nn.LSTM", [
        ("a", WHOLE, "8 * 10", r'''mistakes={"les neurones A n'ont pas de biais": 88, "8 neurones qui lisent 10 entrées : un produit": 18}'''),
        ("b", WHOLE, "8 * 8", r'''mistakes={"les neurones B n'ont pas de biais": 72, "8 neurones qui lisent 8 valeurs : un produit": 16}'''),
        ("c", WHOLE, "3 * 8 + 3", r'''mistakes={"les neurones C ont un biais": 24}'''),
        ("d", WHOLE, "80 + 64 + 8 + 27",
         r'''mistakes={"n'oublie pas les 8 biais de l'état": 171, "ajoute les neurones de sortie C": 152}'''),
        ("e", WHOLE, "n_recurrent(10, 8, 1)",
         r'''mistakes={"PyTorch a deux vecteurs de biais, b_ih et b_hh": 152, "n'oublie pas les biais": 144}'''),
        ("f", WHOLE, "n_recurrent(10, 8, 1) + 8 * 3 + 3 - 179",
         r'''mistakes={"l'écart, pas le total": 187, "PyTorch compte un vecteur de biais de plus": 0}'''),
        ("g", WHOLE, "n_recurrent(10, 8, 4)",
         r'''mistakes={"4 blocs de portes": 160, "PyTorch a deux vecteurs de biais par couche": 608}'''),
        ("h", WHOLE, "n_recurrent(89, 128, 4) + n_recurrent(128, 128, 4) + 128 * 89 + 89",
         r'''mistakes={"la deuxième couche LSTM reçoit 128 valeurs, pas 89": 2 * n_recurrent(89, 128, 4) + 128 * 89 + 89,
          "compte comme PyTorch : deux vecteurs de biais par couche LSTM": 254_681,
          "n'oublie pas la couche dense": n_recurrent(89, 128, 4) + n_recurrent(128, 128, 4)}'''),
        ("i", WHOLE, "2 * 4 * 128",
         r'''mistakes={"il y a deux couches LSTM": 512, "un vecteur de biais a 4 × 128 valeurs (une par porte)": 256, "l'écart entre les deux décomptes exacts, pas avec 250 000": 5705}'''),
    ]),
    Paper("22.4", "Un pas de LSTM à la main", [
        ("a", DEC4, "I_224", r'''decimals=4, mistakes={"σ(0) = 0,5, pas 0": 0.0, "n'oublie pas W_hh h_{t−1}": sigmoid(-0.5)}'''),
        ("b", DEC4, "F_224", r'''decimals=4, mistakes={"le biais de f vaut 1": 0.5, "attention au signe de W_hh (−2)": sigmoid(3)}'''),
        ("c", DEC4, "G_224", r'''decimals=4, mistakes={"le candidat g passe par une tanh, pas une sigmoïde": sigmoid(1)}'''),
        ("d", DEC4, "O_224", r'''decimals=4, mistakes={"n'oublie pas W_hh h_{t−1} = 3 × 0,5": sigmoid(-1)}'''),
        ("e", DEC4, "C_224",
         r'''decimals=4, mistakes={"la porte d'oubli multiplie c_{t−1}": 2 + I_224 * G_224, "ajoute i g": F_224 * 2}'''),
        ("f", DEC4, "H_224",
         r'''decimals=4, mistakes={"h_t = o tanh(c_t) : n'oublie pas la tanh": O_224 * C_224, "multiplie par la porte de sortie": math.tanh(C_224)}'''),
        ("g", DEC4, "C_OPEN_224",
         r'''decimals=4, mistakes={"σ(−20) ≈ 0 : la porte d'entrée est fermée": 2 + G_224}'''),
        ("h", DEC4, "F_224", r'''decimals=4, mistakes={"la dérivée de f c_{t−1} par rapport à c_{t−1} n'est pas 1": 1.0}'''),
        ("i", SHAPE, '"(4, 1)"',
         r'''mistakes={"PyTorch range les 4 blocs dans les LIGNES de weight_ih_l0 : (4H, D)": "(1, 4)",
          "4 blocs de portes": "(1, 1)"}'''),
    ]),
    Paper("22.5", "Softmax avec température à la main", [
        ("a", DEC4, "softmax_t(Z_225, 1)[0]",
         r'''decimals=4, mistakes={"la softmax prend des exponentielles, pas les logits eux-mêmes": 3 / 4.5}'''),
        ("b", DEC4, "softmax_t(Z_225, 0.5)[0]",
         r'''decimals=4, mistakes={"n'oublie pas le troisième caractère": 1 / (1 + math.exp(-4))}'''),
        ("c", DEC4, "softmax_t(Z_225, 2)[0]",
         r'''decimals=4, mistakes={"n'oublie pas le troisième caractère": 1 / (1 + math.exp(-1))}'''),
        ("d", DEC4, "1.0", r'''decimals=4, mistakes={"quand T tend vers 0, le plus probable prend toute la probabilité": 0.0}'''),
        ("e", DEC4, "1 / 3", r'''decimals=4, mistakes={"quand T tend vers l'infini, tous les logits divisés par T tendent vers 0": 0.0}'''),
        ("f", "a number, 2 decimals", "math.exp((3 - 1) / 0.5)",
         r'''decimals=2, mistakes={"à T = 0,5, l'écart 3 − 1 est divisé par 0,5": math.exp(2), "le rapport des probabilités, pas des logits": 3.0}'''),
        ("g", TF, "True", r'''mistakes={"diviser tous les logits par un même T > 0 change-t-il leur ordre ?": False}'''),
    ]),
    Paper("22.8", "Fermi : les 17 millions de poids du CNN-LSTM et le prix d'une epoch", [
        ("a", WHOLE, "D_228",
         r'''mistakes={"le stride 2 divise chaque côté par 2": 128 * 128 * 8, "n'oublie pas les 8 canaux": SIDE_228 ** 2}'''),
        ("b", WHOLE, "CONV_228", r'''mistakes={"un biais par filtre": 8 * 25, "les images n'ont qu'un canal": 8 * (25 * 3 + 1)}'''),
        ("c", WHOLE, "LSTM_228",
         r'''mistakes={"4 blocs de portes": n_recurrent(D_228, 128, 1),
          "deux vecteurs de biais de 4 × 128 valeurs": 4 * (D_228 * 128 + 128 * 128)}'''),
        ("d", WHOLE, "DENSE_228", r'''mistakes={"n'oublie pas les 9 biais": 128 * 9}'''),
        ("e", WHOLE, "CONV_228 + LSTM_228 + DENSE_228",
         r'''mistakes={"additionne les trois couches": LSTM_228}'''),
        ("f", WHOLE, "n_recurrent(D_228, 32, 4)", r'''mistakes={"4 blocs de portes": n_recurrent(D_228, 32, 1)}'''),
        ("g", WHOLE, "MACS_228", r'''mistakes={"4 blocs de 128 colonnes : D × 4 × 128": D_228 * 128}'''),
        ("h", WHOLE, "int(math.log10(MACS_228 * 1000 * 16))",
         r'''fractional="un nombre entier : l'exposant n de l'écriture a × 10ⁿ, avec 1 ≤ a < 10", mistakes={"pour toute l'epoch : 1 000 vidéos de 16 images": 7, "1 000 × 16 = 16 000 images, pas 1 000": 10}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 22.1 à 22.5 et 🧮 22.8)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; une "
               "forme : entre guillemets (`\"(3, 5, 1)\"`) ; un mot : entre guillemets (`\"fit\"`). Arrondis comme "
               "l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. "
               "Les questions « dans ta copie », les preuves ∂ 22.6 et 22.7, la réflexion (🗣️ 22.9, ⚖️ 22.10, "
               "📄 22.11) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Shared tools of parts A to D
# ---------------------------------------------------------------------------
TOOLS = r'''# Tools for the notebook exercises (parts A to D)
import math
import re
import time
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from wb.nbtools import (StopCheck, as_numpy, checked, filled, fr, pair, refuses, returned, run_pytest_group, same,
                        shape_text, verdict, written)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # a computer without PyTorch (Mac Intel): this chapter is done on Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : ce chapitre se fait sur Colab (00_setup/INSTALL_LOCAL.md). Les "
          "cellules qui utilisent PyTorch affichent ⏳ au lieu de s'exécuter ; les tests de tes librairies tournent "
          "sans lui : python -m pytest tests/test_ch22_sequences.py tests/test_ch22_rnn.py -q")

    def _skip_torch_cells(lines):
        """IPython input transformer: without PyTorch, a cell that uses it prints ⏳ instead of stopping "Run all"."""
        if re.search(r"\b(torch|torchvision)\b|(?<![\w.])(nn|F)\.", "".join(lines)):
            return ['print("⏳ PyTorch absent sur cet ordinateur : cette cellule se fait sur Colab.")\n']
        return lines

    try:
        _shell = get_ipython()
    except NameError:                     # not in a notebook
        _shell = None
    if _shell is not None and all(getattr(f, "__name__", "") != "_skip_torch_cells" for f in _shell.input_transformers_cleanup):
        _shell.input_transformers_cleanup.append(_skip_torch_cells)

DEVICE = cfg.device                       # "cuda" if a GPU is available, else "cpu" (wb.setup)
TEST_SEQ = "tests/test_ch22_sequences.py"  # the tests of your library sequences.py
TEST_RNN = "tests/test_ch22_rnn.py"        # the tests of your library rnn.py
HOLMES = wb.datasets.load_holmes()         # The Adventures of Sherlock Holmes, a str of 562 203 characters
VERNE = wb.datasets.load_verne()           # Le Tour du monde en quatre-vingts jours, a str of 421 336 characters
SUNSPOTS = wb.datasets.load_sunspots(definitive_only=True)   # monthly sunspot numbers since January 1749 (SILSO)
SPOTS = SUNSPOTS["sunspots"].to_numpy(dtype=float)          # the series alone, one value per month
I86 = int(np.flatnonzero(SUNSPOTS["year"].to_numpy() >= 1986)[0])   # index of January 1986
I06 = int(np.flatnonzero(SUNSPOTS["year"].to_numpy() >= 2006)[0])   # index of January 2006


def n_params(model):
    """The number of parameters (learned numbers) of a torch module."""
    return sum(p.numel() for p in model.parameters())


def numpy_weights(module):
    """The parameters of a torch module as NumPy arrays, by name (weight_ih_l0, bias_hh_l0...)."""
    return {name: p.detach().cpu().numpy() for name, p in module.named_parameters()}


print(f"tools ready · torch {torch.__version__ if torch is not None else 'absent'} · device {DEVICE} · "
      f"{len(SPOTS)} months, {SUNSPOTS['date'].iloc[0]:%Y-%m} to {SUNSPOTS['date'].iloc[-1]:%Y-%m}")'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="22")   # reload your saved files\n'


def mylearn_howto(file: str) -> str:
    return (f"> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre `mon_travail/mylearn/{file}` (créé par "
            "`python tools/start_chapter.py 22`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de "
            "l'exercice par ton code et **enregistre**. NumPy et la bibliothèque standard seulement. La première cellule "
            "de vérification recharge ta librairie et l'essaie sur quelques cas ; la seconde lance les tests de "
            f"l'exercice. `python -m pytest tests/test_ch22_{file[:-3]}.py -q` les lance tous dans un terminal.")


COMMON_SEQ = ("**Contrat commun de `sequences.py`** (les docstrings, plus ces précisions que vérifient les tests) : tes "
              "fonctions acceptent des listes comme des tableaux et ne modifient **jamais** leurs entrées ; un "
              "argument « entier » refuse un `bool` (`True` n'est pas un nombre de pas) et toute valeur < 1, avec "
              "`ValueError`.")

COMMON_RNN = ("**Contrat commun de `rnn.py`** (les docstrings, plus ces précisions que vérifient les tests) : les "
              "entrées peuvent être des listes et ne sont jamais modifiées ; les résultats sont de **nouveaux** "
              "tableaux `float64`, même pour des entrées entières ou `float32`, jamais une vue d'un autre résultat ; "
              "les poids sont dans la disposition de `mylearn`, `W_ih` de forme `(D, kH)` et `W_hh` de forme "
              "`(H, kH)` (les `weight_ih.T` et `weight_hh.T` de PyTorch, $k = 1$, 4 ou 3), avec des blocs de "
              "$H$ colonnes dans l'ordre de PyTorch ; un biais a la forme `(kH,)` exactement, et `None` vaut zéro ; "
              "`x_t` et les états sont 2-D avec le même nombre de lignes $N$, une séquence `x` est 3-D `(N, T, D)` "
              "avec $T \\geq 1$. Tu vérifies toi-même les formes : celles que NumPy diffuserait sans erreur (un poids "
              "d'une seule colonne, un `x_t` d'une ligne face à un état de deux, un biais `(1,)`) lèvent aussi "
              "`ValueError`, comme la disposition de PyTorch.")


def ref_check(check: str) -> str:
    """The solutions version of a check cell: the tests run on the reference."""
    return check.replace('run_pytest_group(TEST_SEQ, "', 'run_pytest_group(TEST_SEQ, impl="ref", keyword="').replace(
        'run_pytest_group(TEST_RNN, "', 'run_pytest_group(TEST_RNN, impl="ref", keyword="')


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "{" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items() if letter != answer) + "}"


def tests_cell(ex_id: str, probe: str, test_file: str, keyword: str, what: str = "tes fonctions") -> str:
    """The second check cell of a 🔨: the tests of the exercise (no PyTorch needed)."""
    return (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group({test_file}, "{keyword}")\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')


def tests_after(ex_id: str, probe: str, test_file: str, keyword: str, what: str = "tes fonctions") -> list:
    cell = tests_cell(ex_id, probe, test_file, keyword, what)
    return [("check", cell), ("solution", ref_check(cell))]


# ---------------------------------------------------------------------------
# Part A: sequences.py, and what a dense network sees (22.12 to 22.14)
# ---------------------------------------------------------------------------
GIVEN_12 = r'''VERNE_WORDS = re.findall(r"[^\W\d_]+", VERNE.lower())    # the words of Verne: runs of letters, accents kept
print(len(VERNE_WORDS), "words ·", VERNE_WORDS[:8])'''

CHECK_12 = r'''with checked("22.12"):
    S = mylearn.sequences
    stoi_12, itos_12 = pair("22.12", "build_vocab(HOLMES)", S.build_vocab(HOLMES), names=("stoi", "itos"))
    verdict("22.12", isinstance(stoi_12, dict) and isinstance(itos_12, list) and len(stoi_12) == len(itos_12)
            and all(stoi_12.get(token) == i for i, token in enumerate(itos_12)),
            "build_vocab(HOLMES) rend un dict et une list tels que itos[stoi[t]] == t.",
            "build_vocab doit rendre (stoi, itos) : un dict et une list, avec itos[stoi[t]] == t pour chaque token.")
    counts_12 = Counter(HOLMES)
    verdict("22.12", set(itos_12) == set(counts_12)
            and all(counts_12[a] >= counts_12[b] for a, b in zip(itos_12, itos_12[1:])),
            "chaque caractère de HOLMES a un id, du plus fréquent au plus rare.",
            "les ids doivent suivre l'effectif décroissant : chaque caractère de HOLMES une fois, le plus fréquent en 0.")
    ids_12 = S.encode(HOLMES[:5000], stoi_12)
    verdict("22.12", isinstance(ids_12, np.ndarray) and ids_12.dtype == np.int64
            and S.decode(ids_12, itos_12) == HOLMES[:5000],
            "decode(encode(texte)) redonne le texte, avec des ids int64.",
            "encode doit rendre un tableau int64, et decode(encode(texte), itos) doit redonner le texte exact.")
    wb.check("22.12a", len(itos_12), computed=True)
    wb.check("22.12b", stoi_12.get("e"), computed=True)
    wb.check("22.12c", S.encode("Watson", stoi_12), computed=True)
    stoi_w_12, itos_w_12 = pair("22.12", "build_vocab(VERNE_WORDS, max_size=2000, unk_token='<unk>')",
                                S.build_vocab(VERNE_WORDS, max_size=2000, unk_token="<unk>"), names=("stoi", "itos"))
    ids_w_12 = np.asarray(S.encode(VERNE_WORDS, stoi_w_12, unk_token="<unk>"))
    wb.check("22.12d", float(np.mean(ids_w_12 == stoi_w_12.get("<unk>"))), computed=True)
    wb.check("22.12e", itos_w_12[1] if len(itos_w_12) > 1 else None, computed=True)
    refuses("22.12", "encode('Holmes!', stoi) sans unk_token, « ! » absent du vocabulaire",
            lambda: S.encode("Holmes!", S.build_vocab("Holmes")[0]))
    refuses("22.12", "decode([len(itos)], itos)", lambda: S.decode([len(itos_12)], itos_12))
    refuses("22.12", "build_vocab('') (un texte vide)", lambda: S.build_vocab(""))'''

RECORD_12 = r'''stoi_12, itos_12 = mylearn.sequences.build_vocab(HOLMES)
stoi_w_12, itos_w_12 = mylearn.sequences.build_vocab(VERNE_WORDS, max_size=2000, unk_token="<unk>")
ids_w_12 = mylearn.sequences.encode(VERNE_WORDS, stoi_w_12, unk_token="<unk>")
wb.record("22.12a", len(itos_12), mistakes={"le vocabulaire n'a pas de token inconnu ici (unk_token=None)": len(itos_12) + 1})
wb.record("22.12b", stoi_12["e"], mistakes={"les ids partent de 0, comme les indices de Python": stoi_12["e"] + 1,
          "le plus fréquent d'abord : quel caractère est encore plus fréquent que « e » ?": 0})
wb.record("22.12c", mylearn.sequences.encode("Watson", stoi_12))
share_2001_12 = float(np.mean(mylearn.sequences.encode(VERNE_WORDS, mylearn.sequences.build_vocab(VERNE_WORDS, max_size=2001, unk_token="<unk>")[0], unk_token="<unk>") == 0))
wb.record("22.12d", float(np.mean(ids_w_12 == stoi_w_12["<unk>"])), decimals=4,
          mistakes={"unk_token compte dans max_size : 2 000 entrées en tout, <unk> compris": share_2001_12})
wb.record("22.12e", itos_w_12[1], mistakes={"<unk> prend l'id 0, même s'il n'apparaît pas dans le texte : l'id 1 est le mot le plus fréquent": itos_w_12[2]})'''

CHECK_13 = r'''with checked("22.13"):
    S = mylearn.sequences
    X_13, y_13 = pair("22.13", "make_windows(SPOTS, 120, horizon=6)", S.make_windows(SPOTS, 120, horizon=6), names=("X", "y"))
    view_13 = np.lib.stride_tricks.sliding_window_view(SPOTS, 120)[:len(SPOTS) - 125]   # windows that keep a target
    same("22.13", "X de make_windows(SPOTS, 120, horizon=6) contre sliding_window_view, en (N, w, 1)", X_13, view_13[:, :, None])
    same("22.13", "y : la valeur 6 mois après la dernière valeur de chaque fenêtre", y_13, SPOTS[125:])
    X2_13, y2_13 = pair("22.13", "make_windows(SPOTS[:100], 12, stride=12)", S.make_windows(SPOTS[:100], 12, stride=12), names=("X", "y"))
    same("22.13", "les cibles de make_windows(SPOTS[:100], 12, stride=12)", y2_13, SPOTS[12:100:12])
    ids_13 = np.arange(1001)
    X3_13, Y3_13 = pair("22.13", "make_sequence_pairs(np.arange(1001), 40)", S.make_sequence_pairs(ids_13, 40), names=("X", "Y"))
    same("22.13", "X de make_sequence_pairs(np.arange(1001), 40) : 25 morceaux disjoints", X3_13, ids_13[:1000].reshape(25, 40))
    same("22.13", "Y : les mêmes morceaux décalés d'un pas", Y3_13, ids_13[1:1001].reshape(25, 40))
    refuses("22.13", "make_windows(SPOTS[:5], 5) (pas de place pour une cible)", lambda: S.make_windows(SPOTS[:5], 5))
    refuses("22.13", "make_windows(SPOTS, 12, horizon=True) (un bool n'est pas un entier)", lambda: S.make_windows(SPOTS, 12, horizon=True))
    refuses("22.13", "make_sequence_pairs(ids, 40, stride=0)", lambda: S.make_sequence_pairs(ids_13, 40, stride=0))'''

GIVEN_14 = r'''from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

T_14, WAVE_14 = wb.synth.noisy_sine(n=1200, freq=1 / 24, noise=0.1, t_max=1199, seed=2214)   # period: 24 steps
fig, ax = plt.subplots(figsize=(9, 2.2))
ax.plot(T_14[:150], WAVE_14[:150], lw=1)
ax.set_xlabel("pas de temps")
ax.set_title("Les 150 premiers pas de la série")
plt.show()'''

TODO_14 = r'''shuffled_14 = ...    # a) variant 3 (the 8 values of each window in a random order of its own): its test MSE is
                     #    "A" below persistence, "B" between 1 and 3 times persistence, "C" more than 3 times persistence
two_beats_14 = ...   # b) variant 4 (only the last two values, in order): does it beat persistence? True or False
c_vs_mean_14 = ...   # c) is the test MSE of variant 3 less than 70 % of the MSE of the constant prediction (the training mean)?
                     #    True or False'''

SOLUTION_14 = r'''shuffled_14, two_beats_14, c_vs_mean_14 = "C", True, False'''

EXPERIMENT_14 = r'''with checked("22.14"):
    X_14, y_14 = mylearn.sequences.make_windows(WAVE_14, 8)        # your make_windows: (N, 8, 1) and (N,)
    X_14, y_14 = np.asarray(X_14)[:, :, 0], np.asarray(y_14)
    N_TRAIN_14 = 900                                               # the first 900 windows train, the others test
    rng_14 = np.random.default_rng(14)
    variants_14 = {"1 · ordre naturel": X_14,
                   "2 · une même permutation pour toutes les fenêtres": X_14[:, rng_14.permutation(8)],
                   "3 · un ordre au hasard propre à chaque fenêtre": np.array([row[rng_14.permutation(8)] for row in X_14]),
                   "4 · seulement les deux dernières valeurs, dans l'ordre": X_14[:, -2:]}
    target_14 = y_14[N_TRAIN_14:]
    mse_14 = {"persistance (la dernière valeur)": np.mean((X_14[N_TRAIN_14:, -1] - target_14) ** 2),
              "constante (la moyenne d'entraînement)": np.mean((y_14[:N_TRAIN_14].mean() - target_14) ** 2)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for name_14, inputs_14 in variants_14.items():
            mlp_14 = MLPRegressor(hidden_layer_sizes=(32,), max_iter=3000, random_state=0)
            mlp_14.fit(inputs_14[:N_TRAIN_14], y_14[:N_TRAIN_14])
            mse_14[name_14] = np.mean((mlp_14.predict(inputs_14[N_TRAIN_14:]) - target_14) ** 2)
    for name_14, value_14 in mse_14.items():
        print(f"{name_14:55s} MSE de test {value_14:.4f}")
wb.check("22.14a", shuffled_14)
wb.check("22.14b", two_beats_14)
wb.check("22.14c", c_vs_mean_14)'''

RECORD_14 = (r'''wb.record("22.14a", "C", mistakes=''' + letters("C", {
    "A": "dans la variante 3, le réseau a les huit valeurs, mais sait-il si la série monte ou descend ?",
    "B": "compare la variante 3 à la persistance : combien de fois plus grande est son erreur ?"}) + r''')
wb.record("22.14b", True, mistakes={"deux valeurs dans l'ordre donnent la pente : la persistance, elle, n'en a qu'une": False})
wb.record("22.14c", False, mistakes={"compare l'erreur de la variante 3 à celle de la prédiction constante : est-elle beaucoup plus petite ?": True})''')

PART_A = Part("A", "Ta librairie sequences.py : vocabulaire, fenêtres, et ce que voit un réseau dense",
              "Tu écris en NumPy, dans `sequences.py`, ce qui transforme un texte ou une série en exemples : le "
              "vocabulaire et l'encodage (22.12), les fenêtres et les paires « morceau, morceau décalé » (22.13) ; puis "
              "tu mesures ce que perd une couche dense quand l'ordre d'une fenêtre disparaît (22.14). La cellule "
              "suivante importe PyTorch et les outils des parties A à D, et charge les textes de Holmes et de Verne et "
              "la série des taches solaires. Ne recopie jamais un nombre à la main : chaque réponse vérifiée est "
              "calculée par ton code.",
              given=TOOLS, exercises=[
    Ex("22.12", "🔨", 1, 15, "build_vocab, encode et decode sur Holmes et Verne",
       "numéroter les tokens d'un texte, du plus fréquent au plus rare, et passer d'un texte à ses ids et retour.",
       "0A (dictionnaires, `sorted`) · ch. 6 (fréquences des lettres) · fiche §22.9", thread="Holmes/Verne",
       tracks="R, C", mylearn="sequences.py",
       body=mylearn_howto("sequences.py") + "\n\n" + COMMON_SEQ + r"""

Écris `build_vocab(tokens, max_size=None, min_freq=1, unk_token=None)`, `encode(tokens, stoi, unk_token=None)` et `decode(ids, itos, sep="")`. **Contrat** propre à ces trois fonctions :
- `build_vocab` reçoit les caractères d'une `str`, ou n'importe quel itérable de `str` (une liste de mots, un générateur aussi) ; elle rend `(stoi, itos)` : un dictionnaire token → id et la liste id → token ; les ids suivent l'effectif **décroissant**, et deux tokens de même effectif se rangent dans l'ordre de `sorted` ; `unk_token`, s'il est donné, prend l'id 0 (même s'il apparaît dans le texte) et **compte** dans `max_size` ; elle refuse un texte vide, un token vide, et un `max_size` ou un `min_freq` qui n'est pas un entier ≥ 1 ;
- `encode` rend un tableau `int64` (de forme `(0,)` pour un texte vide) ; un token absent de `stoi` est remplacé par `unk_token` s'il est donné, sinon c'est une `ValueError` (pas une `KeyError`) ; un `unk_token` absent de `stoi` est aussi une `ValueError` ;
- `decode` recolle les tokens avec `sep` entre eux ; elle refuse des ids négatifs ou ≥ `len(itos)` (Python lirait `itos[-1]` sans rien dire), des ids qui ne sont pas 1-D, et des ids qui ne sont pas des entiers (`1.7`, `True`).

La cellule suivante découpe le roman de Verne en mots. La première cellule de vérification construit le vocabulaire des caractères de Holmes, puis celui des 2 000 mots les plus fréquents de Verne (token inconnu compris), et vérifie, calculés par tes fonctions : a) le nombre de caractères distincts de Holmes, b) l'id de « e », c) l'encodage de « Watson », d) la part des mots de Verne remplacés par `<unk>`, e) le mot d'id 1 du vocabulaire de Verne. La seconde lance les tests (`-k "test_build_vocab_ or test_encode_ or test_decode_"`).

Dans tes notes : avec un vocabulaire de 2 000 mots, quelle part du texte de Verne devient inconnue ? Pourquoi le livre préfère-t-il les caractères pour son générateur (fiche §22.9), et que proposent les modèles de langue actuels entre les deux ?""",
       given=GIVEN_12, check=RELOAD + CHECK_12, solution=CHECK_12, record=RECORD_12,
       after=tests_after("22.12", 'mylearn.sequences.build_vocab("ab")', "TEST_SEQ",
                         "test_build_vocab_ or test_encode_ or test_decode_")),

    Ex("22.13", "🔨", 2, 20, "make_windows et make_sequence_pairs",
       "découper une série en fenêtres et leurs cibles, et un texte en morceaux et leurs suites, au format (N, T, D).",
       "✏️ 22.1 · Ex 22.12 · fiche §22.2, §22.5", thread="taches solaires", tracks="R, M, C", mylearn="sequences.py",
       body=mylearn_howto("sequences.py") + "\n\n" + COMMON_SEQ + r"""

Écris `make_windows(series, window, horizon=1, stride=1)` et `make_sequence_pairs(seq, seq_len, stride=None)`. **Contrat** :
- `make_windows` : l'exemple $i$ démarre en $i \cdot s$ ; son entrée est `series[start:start + w]` et sa cible la valeur `horizon` pas après la dernière valeur de la fenêtre ; il rend `(X, y)` avec `X` de forme `(N, w, D)` (D = 1 pour une série 1-D, au format batch-first de la fiche) et `y` de forme `(N,)` ou `(N, D)`, $N = \lfloor (T - w - h)/s \rfloor + 1$ (✏️ 22.1) ;
- `make_sequence_pairs` : le morceau $i$ démarre en $i \cdot s$ (`stride=None` vaut `seq_len`, des morceaux sans recouvrement) ; `X[i]` contient `seq_len` valeurs et `Y[i]` les mêmes, décalées d'un pas (`Y[:, :-1] == X[:, 1:]`) ;
- les deux rendent de **nouveaux** tableaux, du dtype de la série ; elles n'acceptent qu'une série `(T,)` ou `(T, D)` (une série 3-D est refusée) ; une série trop courte pour un seul exemple lève `ValueError`.

La première cellule compare tes fenêtres des taches solaires à `sliding_window_view` de NumPy, essaie un pas de 12 et des paires de morceaux, puis trois refus ; la seconde lance les tests (`-k "test_make_windows_ or test_make_sequence_pairs_"`).

Dans tes notes : pourquoi `make_windows` rend-elle des copies plutôt que des vues de la série (pense à ce qui arriverait si l'on normalisait ensuite `X` sur place) ? Combien d'exemples d'entraînement par morceau de 40 caractères donne `make_sequence_pairs`, contre combien pour une fenêtre de `make_windows` ?""",
       check=RELOAD + CHECK_13, solution=CHECK_13,
       after=tests_after("22.13", "mylearn.sequences.make_windows(np.arange(5.0), 2)", "TEST_SEQ",
                         "test_make_windows_ or test_make_sequence_pairs_")),

    Ex("22.14", "🔮", 2, 20, "Un MLP sur des fenêtres : l'ordre compte-t-il ?",
       "prédire ce que perd un réseau dense quand on mélange l'ordre des valeurs de ses fenêtres, puis le mesurer.",
       "Ex 22.13 · ch. 16 (réseau dense) · ch. 15 (`MLPRegressor`) · fiche §22.2", thread="synthétique", tracks="C",
       hypothesis=True,
       body=r"""Une série synthétique : un sinus de période 24 pas, plus un bruit d'écart-type 0,1 (1 200 pas). On prédit chaque valeur à partir des 8 précédentes, avec un petit réseau dense de scikit-learn (`MLPRegressor`, 32 neurones cachés), entraîné sur les 900 premières fenêtres et testé sur les autres. Quatre façons de lui présenter les fenêtres :
- **1** : l'ordre naturel ;
- **2** : une même permutation des 8 positions pour toutes les fenêtres (fiche §22.2 : le réseau peut l'apprendre aussi bien) ;
- **3** : les 8 valeurs de chaque fenêtre dans un ordre au hasard, **différent pour chaque fenêtre** ;
- **4** : seulement les deux dernières valeurs, dans l'ordre.

On compare l'erreur quadratique moyenne de test (MSE) à deux références : la **persistance** (prédire la dernière valeur de la fenêtre) et la **constante** (prédire la moyenne des cibles d'entraînement). Prédis, **avant** d'exécuter :
a) `shuffled_14` : l'erreur de la variante 3 est `"A"` plus petite que celle de la persistance, `"B"` entre 1 et 3 fois celle de la persistance, `"C"` plus de 3 fois celle de la persistance ;
b) `two_beats_14` : la variante 4 bat-elle la persistance ? (`True` ou `False`)
c) `c_vs_mean_14` : l'erreur de la variante 3 est-elle plus petite que 70 % de celle de la constante ? (`True` ou `False`)

L'expérience utilise ta `make_windows` (22.13), entraîne les quatre réseaux (quelques secondes) et affiche les six erreurs.""",
       given=GIVEN_14, todo=TODO_14, solution=SOLUTION_14,
       after=[("code", guarded(EXPERIMENT_14, ["shuffled_14", "two_beats_14", "c_vs_mean_14"],
                               "⏳ Ex 22.14 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Que garde la "
                     "variante 3 des huit valeurs, et que perd-elle ? Pourquoi deux valeurs dans l'ordre valent-elles "
                     "mieux que huit valeurs en vrac ?")],
       record=RECORD_14,
       note="Mélangées fenêtre par fenêtre, les huit valeurs disent à quelle hauteur se trouve la série, mais plus si "
            "elle monte ou si elle descend : deux tronçons symétriques du sinus donnent le même ensemble de valeurs et "
            "deux suites opposées. Le réseau ne peut que viser la moyenne des deux, et fait à peine mieux que la "
            "constante, environ huit fois et demie plus mal que la persistance. Avec seulement les deux dernières valeurs dans "
            "l'ordre, il connaît la pente et extrapole mieux que la persistance. Ce n'est pas le nombre de valeurs qui "
            "compte, c'est l'ordre : ce que la couche dense ne sait pas exploiter d'elle-même et que l'état d'un RNN "
            "porte naturellement."),
])

# ---------------------------------------------------------------------------
# Part B: rnn.py, a benchmark and the vanishing gradient (22.15 to 22.19)
# ---------------------------------------------------------------------------
CHECK_15 = r'''with checked("22.15"):
    R = mylearn.rnn
    torch.manual_seed(2215)
    torch_15 = nn.RNN(3, 4, batch_first=True).double()           # the oracle; its weights are copied with .T
    w_15 = numpy_weights(torch_15)
    x_15 = torch.randn(2, 6, 3, dtype=torch.float64)
    h0_15 = torch.randn(1, 2, 4, dtype=torch.float64)
    outputs_15, h_T_15 = pair("22.15", "rnn_forward", R.rnn_forward(
        x_15.numpy(), w_15["weight_ih_l0"].T, w_15["weight_hh_l0"].T, w_15["bias_ih_l0"], w_15["bias_hh_l0"],
        h0=h0_15[0].numpy()), names=("outputs", "h_T"))
    want_out_15, want_h_15 = torch_15(x_15, h0_15)
    same("22.15", "outputs de rnn_forward contre nn.RNN(batch_first=True), (N, T, H)", outputs_15, want_out_15.detach())
    same("22.15", "h_T contre h_n[0] de nn.RNN", h_T_15, want_h_15[0].detach())
    cell_15 = nn.RNNCell(3, 4, nonlinearity="relu").double()
    wc_15 = numpy_weights(cell_15)
    same("22.15", "rnn_cell_forward(..., nonlinearity='relu') contre nn.RNNCell",
         R.rnn_cell_forward(x_15[:, 0].numpy(), h0_15[0].numpy(), wc_15["weight_ih"].T, wc_15["weight_hh"].T,
                            wc_15["bias_ih"], wc_15["bias_hh"], nonlinearity="relu"),
         cell_15(x_15[:, 0], h0_15[0]).detach())
    refuses("22.15", "W_ih dans la disposition de PyTorch, (H, D) = (4, 3)",
            lambda: R.rnn_cell_forward(x_15[:, 0].numpy(), h0_15[0].numpy(), w_15["weight_ih_l0"], w_15["weight_hh_l0"].T))
    refuses("22.15", "nonlinearity='sigmoid'",
            lambda: R.rnn_cell_forward(np.ones((1, 3)), np.zeros((1, 4)), np.ones((3, 4)), np.ones((4, 4)), nonlinearity="sigmoid"))'''

CHECK_16 = r'''with checked("22.16"):
    R = mylearn.rnn
    torch.manual_seed(2216)
    torch_16 = nn.LSTM(3, 5, batch_first=True).double()
    w_16 = numpy_weights(torch_16)
    x_16 = torch.randn(2, 7, 3, dtype=torch.float64)
    h0_16, c0_16 = torch.randn(1, 2, 5, dtype=torch.float64), torch.randn(1, 2, 5, dtype=torch.float64)
    result_16 = R.lstm_forward(x_16.numpy(), w_16["weight_ih_l0"].T, w_16["weight_hh_l0"].T, w_16["bias_ih_l0"],
                               w_16["bias_hh_l0"], h0=h0_16[0].numpy(), c0=c0_16[0].numpy())
    outputs_16, final_16 = pair("22.16", "lstm_forward", result_16, names=("outputs", "(h_T, c_T)"))
    verdict("22.16", isinstance(final_16, tuple), "le second élément de lstm_forward est un tuple (h_T, c_T), comme nn.LSTM.",
            f"le second élément de lstm_forward doit être le tuple (h_T, c_T), reçu un {type(final_16).__name__}.")
    h_T_16, c_T_16 = pair("22.16", "le second élément de lstm_forward", final_16, names=("h_T", "c_T"))
    want_out_16, (want_h_16, want_c_16) = torch_16(x_16, (h0_16, c0_16))
    same("22.16", "outputs de lstm_forward contre nn.LSTM(batch_first=True)", outputs_16, want_out_16.detach())
    same("22.16", "h_T contre h_n[0]", h_T_16, want_h_16[0].detach())
    same("22.16", "c_T contre c_n[0]", c_T_16, want_c_16[0].detach())
    cell_16 = nn.LSTMCell(3, 5).double()
    wc_16 = numpy_weights(cell_16)
    h_t_16, c_t_16 = pair("22.16", "lstm_cell_forward", R.lstm_cell_forward(
        x_16[:, 0].numpy(), h0_16[0].numpy(), c0_16[0].numpy(), wc_16["weight_ih"].T, wc_16["weight_hh"].T,
        wc_16["bias_ih"], wc_16["bias_hh"]), names=("h_t", "c_t"))
    want_ht_16, want_ct_16 = cell_16(x_16[:, 0], (h0_16[0], c0_16[0]))
    same("22.16", "h_t de lstm_cell_forward contre nn.LSTMCell", h_t_16, want_ht_16.detach())
    same("22.16", "c_t de lstm_cell_forward contre nn.LSTMCell", c_t_16, want_ct_16.detach())
    refuses("22.16", "W_ih avec 3H colonnes au lieu de 4H",
            lambda: R.lstm_cell_forward(np.ones((1, 3)), np.zeros((1, 5)), np.zeros((1, 5)), np.ones((3, 15)), np.ones((5, 20))))'''

CHECK_17 = r'''with checked("22.17"):
    R = mylearn.rnn
    torch.manual_seed(2217)
    cell_17 = nn.GRUCell(3, 4).double()
    wc_17 = numpy_weights(cell_17)
    x_17 = torch.randn(5, 3, dtype=torch.float64)
    h_17 = torch.randn(5, 4, dtype=torch.float64)
    got_17 = R.gru_cell_forward(x_17.numpy(), h_17.numpy(), wc_17["weight_ih"].T, wc_17["weight_hh"].T,
                                wc_17["bias_ih"], wc_17["bias_hh"])
    same("22.17", "gru_cell_forward contre nn.GRUCell", got_17, cell_17(x_17, h_17).detach())
    big_17 = np.zeros(12)
    big_17[8:] = 3.0                                   # b_hh only on the n block: r multiplies it
    same("22.17", "gru_cell_forward avec un b_hh non nul sur le bloc n seulement",
         R.gru_cell_forward(np.zeros((1, 3)), np.ones((1, 4)), np.zeros((3, 12)), np.zeros((4, 12)), None, big_17),
         np.full((1, 4), 0.5 * np.tanh(0.5 * 3.0) + 0.5))
    refuses("22.17", "W_hh de forme (4, 16) : un GRU a 3H colonnes",
            lambda: R.gru_cell_forward(np.ones((1, 3)), np.zeros((1, 4)), np.ones((3, 12)), np.ones((4, 16))))'''

TODO_18 = r'''def best_time_18(fn, *args, repeat=5):
    """Call fn(*args) once to warm up, then `repeat` more times, each one timed with time.perf_counter.
    Return the shortest of these `repeat` durations, in seconds (a Python float)."""
    raise NotImplementedError("best_time_18")'''

SOLUTION_18 = r'''def best_time_18(fn, *args, repeat=5):
    """Call fn(*args) once to warm up, then `repeat` more times, each one timed with time.perf_counter.
    Return the shortest of these `repeat` durations, in seconds (a Python float)."""
    fn(*args)                                       # warm-up: caches, memory allocation, lazy initialisation
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        durations.append(time.perf_counter() - start)
    return float(min(durations))                    # the least disturbed run: the others only add noise'''

CHECK_18 = r'''bench_18 = None
with checked("22.18"):
    sleeps_18 = iter([0.03, 0.02, 0.01, 0.025, 0.015, 0.02, 0.05, 0.05])
    calls_18 = []

    def fake_18(x):
        calls_18.append(x)
        time.sleep(next(sleeps_18))

    best_18 = best_time_18(fake_18, "x", repeat=5)
    if returned("22.18", "best_time_18", best_18):
        verdict("22.18", len(calls_18) == 6, "best_time_18 appelle la fonction 6 fois : une de chauffe, puis 5 mesures.",
                f"best_time_18(fn, x, repeat=5) a appelé fn {len(calls_18)} fois : attendu 6 (une de chauffe, puis repeat).")
        verdict("22.18", isinstance(best_18, float) and 0.009 <= best_18 < 0.0145,
                "best_time_18 rend la plus courte des 5 durées mesurées, un float Python.",
                f"best_time_18 rend {best_18!r} pour des durées de 0,020, 0,010, 0,025, 0,015 et 0,020 s après une chauffe de "
                "0,030 s : attendu la plus courte des mesures, en float Python.")
        R = mylearn.rnn
        torch.manual_seed(2218)
        lstm_18 = nn.LSTM(32, 64, batch_first=True).double()
        w_18 = numpy_weights(lstm_18)
        x_18 = torch.randn(32, 100, 32, dtype=torch.float64)
        args_18 = (x_18.numpy(), w_18["weight_ih_l0"].T, w_18["weight_hh_l0"].T, w_18["bias_ih_l0"], w_18["bias_hh_l0"])
        with torch.no_grad():
            same("22.18", "ta lstm_forward contre nn.LSTM sur les entrées du benchmark", R.lstm_forward(*args_18)[0],
                 lstm_18(x_18)[0])
            t_numpy_18 = best_time_18(R.lstm_forward, *args_18, repeat=5)
            t_torch_18 = best_time_18(lstm_18, x_18, repeat=5)
        bench_18 = (t_numpy_18, t_torch_18)
        print(f"lstm_forward (NumPy) : {t_numpy_18 * 1e3:.1f} ms · nn.LSTM : {t_torch_18 * 1e3:.1f} ms · "
              f"rapport {t_numpy_18 / t_torch_18:.1f} (N = 32, T = 100, D = 32, H = 64, float64)")'''

TODO_19 = r'''def input_grad_norms_19(model, x):
    """For a batch-first recurrent module (nn.RNN, nn.LSTM or nn.GRU) and inputs x of shape (N, T, D), return the
    T norms ||dL/dx_t||, a NumPy array of shape (T,), where L = output[:, -1].sum() is the sum of the outputs of the
    last step and the norm of step t is taken over the whole slice dL/dx[:, t, :]. x itself is left unchanged."""
    raise NotImplementedError("input_grad_norms_19")'''

SOLUTION_19 = r'''def input_grad_norms_19(model, x):
    """For a batch-first recurrent module (nn.RNN, nn.LSTM or nn.GRU) and inputs x of shape (N, T, D), return the
    T norms ||dL/dx_t||, a NumPy array of shape (T,), where L = output[:, -1].sum() is the sum of the outputs of the
    last step and the norm of step t is taken over the whole slice dL/dx[:, t, :]. x itself is left unchanged."""
    inputs = x.detach().clone().requires_grad_(True)      # a leaf of its own: x keeps no gradient
    output, _ = model(inputs)
    output[:, -1].sum().backward()
    return inputs.grad.norm(dim=(0, 2)).numpy()           # one norm per time step, over the batch and the features'''

CHECK_19 = r'''grads_ok_19 = False
with checked("22.19"):
    torch.manual_seed(2219)
    small_19 = nn.RNN(3, 4, batch_first=True)
    x_small_19 = torch.randn(2, 5, 3)
    jacobian_19 = torch.autograd.functional.jacobian(lambda inputs: small_19(inputs)[0][:, -1].sum(), x_small_19)
    got_19 = input_grad_norms_19(small_19, x_small_19)
    if returned("22.19", "input_grad_norms_19", got_19):
        grads_ok_19 = same("22.19", "input_grad_norms_19 contre la jacobienne de PyTorch (5 pas)", got_19,
                           jacobian_19.norm(dim=(0, 2)), rtol=1e-5, atol=1e-8)
        verdict("22.19", not x_small_19.requires_grad and x_small_19.grad is None,
                "x n'a pas été modifié (ni requires_grad, ni gradient).",
                "input_grad_norms_19 a modifié x (requires_grad ou .grad) : travaille sur une copie détachée.")'''

EXPERIMENT_19 = r'''if not grads_ok_19:
    print("⏳ Ex 22.19 : l'expérience attend une input_grad_norms_19 juste.")
else:
    T_19, H_19 = 80, 32
    torch.manual_seed(2219)
    x_19 = torch.randn(16, T_19, 1)
    rnn_19 = nn.RNN(1, H_19, batch_first=True)
    rnn_orth_19 = nn.RNN(1, H_19, batch_first=True)
    nn.init.orthogonal_(rnn_orth_19.weight_hh_l0)                  # every singular value of W_hh equals 1
    lstm_19 = nn.LSTM(1, H_19, batch_first=True)
    lstm_open_19 = nn.LSTM(1, H_19, batch_first=True)
    lstm_open_19.load_state_dict(lstm_19.state_dict())
    with torch.no_grad():
        lstm_open_19.bias_ih_l0[H_19:2 * H_19] = 1.0               # forget gate bias: 1 in total (fiche §22.7.2)
        lstm_open_19.bias_hh_l0[H_19:2 * H_19] = 0.0
    curves_19 = {"RNN tanh, initialisation par défaut": rnn_19, "RNN tanh, W_hh orthogonale": rnn_orth_19,
                 "LSTM, initialisation par défaut": lstm_19, "LSTM, biais d'oubli à 1": lstm_open_19}
    fig, ax = plt.subplots(figsize=(8, 4))
    decay_19 = {}
    for name_19, model_19 in curves_19.items():
        norms_19 = np.asarray(input_grad_norms_19(model_19, x_19))[::-1]      # distance 0 = the last step
        decay_19[name_19] = np.log10(norms_19 / norms_19[0])
        ax.plot(decay_19[name_19], label=name_19)
    ax.set_xlabel("distance au dernier pas (T - t)")
    ax.set_ylabel("log10( ||dL/dx_t|| / ||dL/dx_T|| )")
    ax.set_title("Le gradient qui remonte le temps, à l'initialisation")
    ax.legend()
    plt.show()
    for name_19, curve_19 in decay_19.items():
        print(f"{name_19:38s} distance 10 : {curve_19[10]:6.1f} · distance 40 : {curve_19[40]:6.1f} · distance 79 : {curve_19[79]:6.1f}")'''

PART_B = Part("B", "Ta librairie rnn.py : RNN, LSTM et GRU contre PyTorch, et le gradient qui s'évanouit",
              "Tu écris en NumPy les trois cellules du chapitre et leur boucle sur le temps (22.15 à 22.17), avec les "
              "conventions de PyTorch, pour pouvoir y copier ses poids et comparer au milliardième près ; tu mesures "
              "ce que coûte ta version face à `nn.LSTM` (22.18), puis, avec l'autograd de PyTorch, ce que devient le "
              "gradient qui remonte 80 pas (22.19).",
              exercises=[
    Ex("22.15", "🔨", 2, 30, "rnn_cell_forward et rnn_forward",
       "écrire la cellule d'Elman et sa boucle sur le temps, avec les poids et le format de PyTorch.",
       "✏️ 22.2 · Ex 22.13 · fiche §22.4, §22.4.1, §22.5", tracks="R, M, C", mylearn="rnn.py",
       body=mylearn_howto("rnn.py") + "\n\n" + COMMON_RNN + r"""

Écris `rnn_cell_forward(x_t, h_prev, W_ih, W_hh, b_ih=None, b_hh=None, nonlinearity="tanh")` et `rnn_forward(x, W_ih, W_hh, b_ih=None, b_hh=None, h0=None, nonlinearity="tanh")`. **Contrat** propre à ces deux fonctions :
- la cellule calcule $h_t = f(x_t W_{ih} + b_{ih} + h_{t-1} W_{hh} + b_{hh})$, avec $f$ = tanh ou ReLU (`nonlinearity` vaut `"tanh"` ou `"relu"`, rien d'autre) ; elle rend `h_t`, de forme `(N, H)` ;
- `rnn_forward` applique la cellule à chaque pas de `x` `(N, T, D)`, en partant de `h0` (des zéros si `None`, sinon de forme `(N, H)`) ; elle rend `(outputs, h_T)` : `outputs` `(N, T, H)` empile les états, `h_T` `(N, H)` est le dernier, un tableau **à part** (pas une vue de `outputs`).

La première cellule copie les poids d'une `nn.RNN` (et d'une `nn.RNNCell` en ReLU), compare tes sorties aux siennes, et essaie deux refus ; la seconde lance les tests (`-k "test_rnn_cell_forward_ or test_rnn_forward_"`).""",
       check=RELOAD + CHECK_15, solution=CHECK_15,
       after=tests_after("22.15", "mylearn.rnn.rnn_cell_forward([[1.0]], [[0.0]], [[1.0]], [[1.0]])", "TEST_RNN",
                         "test_rnn_cell_forward_ or test_rnn_forward_")),

    Ex("22.16", "🔨", 3, 45, "lstm_cell_forward et lstm_forward (portes i, f, g, o)",
       "écrire un pas de LSTM et sa boucle sur le temps, portes rangées comme dans PyTorch.",
       "Ex 22.15 · ✏️ 22.4 · fiche §22.7.1, §22.7.2", tracks="M, C", mylearn="rnn.py",
       body=mylearn_howto("rnn.py") + "\n\n" + COMMON_RNN + r"""

Écris `lstm_cell_forward(x_t, h_prev, c_prev, W_ih, W_hh, b_ih=None, b_hh=None)` et `lstm_forward(x, W_ih, W_hh, b_ih=None, b_hh=None, h0=None, c0=None)`. **Contrat** :
- `W_ih` a la forme `(D, 4H)` et `W_hh` la forme `(H, 4H)`, leurs colonnes en quatre blocs de $H$ dans l'ordre $i, f, g, o$ ; $i, f, o$ passent par une sigmoïde, $g$ par une tanh ; $c_t = f \odot c_{t-1} + i \odot g$ et $h_t = o \odot \tanh(c_t)$ (fiche §22.7.2) ;
- ta sigmoïde est **stable** : des pré-activations de ±1 000 ne déclenchent aucun `RuntimeWarning` ;
- `lstm_cell_forward` rend le couple `(h_t, c_t)`, dans cet ordre ; `c_prev` a la forme de `h_prev` ;
- `lstm_forward` part de `h0` et `c0` (des zéros si `None`) et rend `(outputs, (h_T, c_T))`, comme `nn.LSTM` : la paire finale est un **tuple**, et `h_T`, `c_T` sont des tableaux à part.

La première cellule copie les poids d'une `nn.LSTM` et d'une `nn.LSTMCell` et compare tes résultats, puis essaie un refus ; la seconde lance les tests (`-k "test_lstm_cell_forward_ or test_lstm_forward_"`).

Dans tes notes : écris l'expression de `outputs[:, -1]` en fonction de `h_T`. Que deviendrait ta cellule si l'on oubliait la tanh de $h_t$ ?""",
       check=RELOAD + CHECK_16, solution=CHECK_16,
       after=tests_after("22.16", "mylearn.rnn.lstm_cell_forward([[1.0]], [[0.0]], [[0.0]], np.ones((1, 4)), np.ones((1, 4)))",
                         "TEST_RNN", "test_lstm_cell_forward_ or test_lstm_forward_")),

    Ex("22.17", "🔨", 2, 25, "gru_cell_forward (portes r, z, n)",
       "écrire un pas de GRU dans la convention de PyTorch, où la porte de réinitialisation agit après $W_{hn}$.",
       "Ex 22.16 · fiche §22.7.2 (le GRU)", tracks="C", mylearn="rnn.py",
       body=mylearn_howto("rnn.py") + "\n\n" + COMMON_RNN + r"""

Écris `gru_cell_forward(x_t, h_prev, W_ih, W_hh, b_ih=None, b_hh=None)`. **Contrat** :
- `W_ih` `(D, 3H)` et `W_hh` `(H, 3H)` ont trois blocs de $H$ colonnes dans l'ordre $r, z, n$ ;
- $r$ et $z$ sont des sigmoïdes de la somme des deux contributions ; le candidat est $n = \tanh\big(x_t W_{in} + b_{in} + r \odot (h_{t-1} W_{hn} + b_{hn})\big)$ : la porte $r$ multiplie **aussi** $b_{hn}$, c'est pourquoi les deux biais ne se fusionnent pas ;
- elle rend $h_t = (1 - z) \odot n + z \odot h_{t-1}$, de forme `(N, H)`.

La première cellule compare ta cellule à `nn.GRUCell`, puis à un cas où seul $b_{hn}$ est non nul (la porte $r$ vaut 0,5 et doit le multiplier), et essaie un refus ; la seconde lance les tests (`-k "test_gru_cell_forward_"`).""",
       check=RELOAD + CHECK_17, solution=CHECK_17,
       after=tests_after("22.17", "mylearn.rnn.gru_cell_forward([[1.0]], [[0.0]], np.ones((1, 3)), np.ones((1, 3)))",
                         "TEST_RNN", "test_gru_cell_forward_", "ta fonction")),

    Ex("22.18", "🛠️", 2, 20, "Benchmark : ta cellule NumPy contre nn.LSTM",
       "mesurer proprement le temps d'une fonction (chauffe, plusieurs mesures, la plus courte) et comparer ta lstm_forward à celle de PyTorch.",
       "Ex 22.16 · ch. 0B (ordre des produits, mesure du temps) · fiche §22.7.2", tracks="C",
       body=r"""Mesurer un temps d'exécution est une compétence de métier : une seule mesure ne veut rien dire (le premier appel paie l'allocation de la mémoire et les caches, et le système d'exploitation peut interrompre n'importe quel appel). La méthode courante, celle du module `timeit` de Python : un appel de chauffe, puis plusieurs mesures, et l'on garde la **plus courte**, la moins perturbée.

**Contrat**
- `best_time_18(fn, *args, repeat=5)` appelle `fn(*args)` une fois sans le mesurer, puis `repeat` fois en mesurant chaque appel avec `time.perf_counter()` ; elle rend la plus courte de ces `repeat` durées, en secondes (un `float` Python).

La vérification essaie ta fonction sur un faux calcul aux durées connues, compare ta `lstm_forward` (22.16) à `nn.LSTM` sur les entrées du benchmark (32 séquences de 100 pas, $D = 32$, $H = 64$, en `float64`), puis mesure les deux et affiche leur rapport.

Dans tes notes : quel est le rapport entre le temps de ta version et celui de `nn.LSTM` sur ta machine ? Lance la cellule deux fois : le rapport bouge-t-il ? Où passe le temps dans ta boucle (pense au nombre de produits matriciels par pas, et à ce que fait l'interpréteur Python entre deux) ? Pourquoi garde-t-on la plus courte des mesures plutôt que leur moyenne ?""",
       todo=TODO_18, check=CHECK_18, solution=SOLUTION_18 + "\n\n\n" + CHECK_18,
       note="Sur deux cœurs, `nn.LSTM` est en général plusieurs fois plus rapide que la boucle NumPy : PyTorch "
            "calcule les quatre portes en un seul produit matriciel par pas, comme toi, mais dans une boucle compilée en "
            "C++ qui évite de passer par l'interpréteur et d'allouer de nouveaux tableaux à chaque pas ; sur GPU, cuDNN "
            "va plus loin en calculant d'un coup les produits $x_t W_{ih}$ de tous les pas, qui ne dépendent pas de "
            "l'état. Le rapport varie d'une exécution à l'autre et d'une machine à l'autre : on le donne en ordre de "
            "grandeur. La plus courte mesure est la meilleure estimation du coût du calcul lui-même : les autres y "
            "ajoutent des interruptions, jamais l'inverse."),

    Ex("22.19", "🔬", 2, 30, "Mesurer la disparition du gradient : RNN tanh contre LSTM",
       "mesurer avec l'autograd la norme du gradient par rapport à chaque entrée, et la comparer d'une cellule et d'une initialisation à l'autre.",
       "Ex 22.15 · ∂ 22.6 · ∂ 22.7 · ch. 20 (autograd) · fiche §22.6, §22.7.2", tracks="M, C",
       body=r"""∂ 22.6 et ∂ 22.7 ont montré sur le papier que le gradient qui remonte le temps est un produit de facteurs. Ici, tu le mesures. On prend la somme des sorties du **dernier** pas, $L = \sum$ `output[:, -1]`, et l'on regarde, pour chaque pas $t$, la taille du gradient de $L$ par rapport à l'entrée de ce pas.

**Contrat**
- `input_grad_norms_19(model, x)` reçoit un module récurrent batch-first de PyTorch (`nn.RNN`, `nn.LSTM` ou `nn.GRU`, qui rend `(output, état)`) et des entrées `x` de forme `(N, T, D)` ; elle rend un tableau NumPy de forme `(T,)` : pour chaque pas $t$, la norme de $\partial L / \partial x[:, t, :]$, prise sur toute la tranche (tous les exemples et toutes les features) ;
- elle ne modifie pas `x` (pas de `requires_grad` ni de `.grad` laissé sur lui).

La vérification compare ta fonction à la jacobienne calculée par `torch.autograd.functional.jacobian` sur un petit RNN. L'expérience la lance ensuite sur quatre réseaux **non entraînés**, à 80 pas et 32 valeurs d'état : un RNN tanh avec l'initialisation par défaut de PyTorch, le même avec $W_{hh}$ orthogonale (toutes ses valeurs singulières valent 1), un LSTM par défaut et le même LSTM avec un biais d'oubli de 1. Elle trace $\log_{10}$ du rapport entre la norme à la distance $T - t$ et celle du dernier pas.

Dans tes notes : à quelle distance le gradient du RNN par défaut a-t-il perdu un facteur $10^{6}$ ? Le LSTM par défaut fait-il beaucoup mieux, et pourquoi (que vaut $f$ au départ, fiche §22.7.2) ? Que change le biais d'oubli ? L'initialisation orthogonale règle-t-elle le problème pour de bon, une fois l'entraînement lancé (pense à ce que deviennent les valeurs singulières et à la saturation) ?""",
       todo=TODO_19, check=CHECK_19, solution=SOLUTION_19 + "\n\n\n" + CHECK_19,
       after=[("code", EXPERIMENT_19)],
       note="À l'initialisation, le RNN par défaut perd un facteur 10 environ tous les 4 pas : $10^{-10}$ à 40 pas. Le "
            "LSTM par défaut ne fait guère mieux, car ses portes d'oubli valent environ $\\sigma(0) = 0{,}5$ : son chemin "
            "additif divise le gradient par deux à chaque pas ($0{,}5^{40} \\approx 10^{-12}$ sur ce seul chemin). Avec "
            "un biais d'oubli de 1, $f \\approx 0{,}73$ et la pente devient bien plus douce. Le RNN à $W_{hh}$ "
            "orthogonale garde le gradient le plus longtemps, mais seulement au départ : l'entraînement déplace les "
            "valeurs singulières et peut saturer la tanh, alors que les portes du LSTM apprennent à rester ouvertes. "
            "Les nombres exacts dépendent du tirage des poids (graine 2219)."),
])

# ---------------------------------------------------------------------------
# Part C: sampling, a leaky series, deep and bidirectional shapes, the sunspots (22.20 to 22.24)
# ---------------------------------------------------------------------------
CHECK_20 = r'''with checked("22.20"):
    S = mylearn.sequences
    logits_20 = np.array([[2.0, 1.0, 0.1, -1.0], [1000.0, 999.0, 0.0, -1000.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)          # an overflow becomes an error, shown as ❌
        probs_20 = S.temperature_softmax(logits_20, temperature=0.5)
    same("22.20", "temperature_softmax(logits, 0.5) contre scipy.special.softmax(logits / 0.5, axis=-1), lignes de ±1 000 comprises",
         probs_20, softmax(logits_20 / 0.5, axis=-1))
    greedy_20 = S.sample_from_logits([0.0, 3.0, 3.0, 1.0], temperature=0)
    verdict("22.20", type(greedy_20) is int and greedy_20 == 1,
            "temperature=0 : le premier argmax, un int Python.",
            f"sample_from_logits([0, 3, 3, 1], temperature=0) rend {greedy_20!r} : attendu le PREMIER indice du maximum, en int Python.")
    draws_20 = np.asarray(S.sample_from_logits(np.tile([0.0, 1.0, 2.0], (20000, 1)), rng=np.random.default_rng(20)))
    freq_20 = np.bincount(draws_20.astype(int), minlength=3)[:3] / draws_20.size if draws_20.shape == (20000,) else None
    want_20 = softmax([0.0, 1.0, 2.0])
    verdict("22.20", freq_20 is not None and np.abs(freq_20 - want_20).max() < 0.015,
            "20 000 tirages (une ligne par tirage) suivent softmax([0, 1, 2]) à 0,015 près.",
            f"20 000 lignes de logits [0, 1, 2] : attendu un tableau (20000,) dont les fréquences valent environ "
            f"{np.round(want_20, 3).tolist()}" + ("" if freq_20 is None else f", reçu {np.round(freq_20, 3).tolist()}") + ".")
    top_20 = np.asarray(S.sample_from_logits(np.tile([0.0, 2.0, 1.0, 1.0], (2000, 1)), top_k=2, rng=np.random.default_rng(21)))
    verdict("22.20", set(np.unique(top_20).tolist()) == {1, 2, 3},
            "top_k=2 garde les deux plus grands logits et les ex aequo du deuxième (ids 1, 2 et 3).",
            f"top_k=2 sur [0, 2, 1, 1] : attendu les ids {{1, 2, 3}} (les ex aequo du k-ième logit restent), reçu {sorted(set(np.unique(top_20).tolist()))}.")
    again_20 = [S.sample_from_logits([0.0, 0.5, 1.0], rng=np.random.default_rng(7)) for _ in range(2)]
    verdict("22.20", again_20[0] == again_20[1], "la même graine redonne le même id.",
            "deux générateurs de même graine doivent donner le même id : tire avec le rng reçu, rien d'autre.")
    refuses("22.20", "temperature=-1", lambda: S.sample_from_logits([0.0, 1.0], temperature=-1))
    refuses("22.20", "top_k=0", lambda: S.sample_from_logits([0.0, 1.0], top_k=0))
    refuses("22.20", "temperature_softmax(..., temperature=0)", lambda: S.temperature_softmax([0.0, 1.0], temperature=0))'''

TODO_21 = r'''from sklearn.model_selection import train_test_split


def prepare_21(series, window=24, horizon=6, n_test=240):
    """Scaled windows to forecast `horizon` months ahead from `window` months; the last `n_test` targets are the test.
    Return X_train, y_train, X_test, y_test (float32 tensors, X of shape (N, window, 1)) and (lo, hi), the min-max scaler."""
    lo, hi = series.min(), series.max()
    scaled = (series - lo) / (hi - lo)
    blocks = np.lib.stride_tricks.sliding_window_view(scaled, window + horizon)   # one row = inputs, then the gap, then the target
    X = blocks[:, -window:]
    y = blocks[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=0)
    to_tensor = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    return to_tensor(X_train)[..., None], to_tensor(y_train), to_tensor(X_test)[..., None], to_tensor(y_test), (lo, hi)


class Forecaster21(nn.Module):
    """An LSTM that reads a window (N, T, 1) and predicts one value per window (N,)."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)'''

SOLUTION_21 = r'''def prepare_21(series, window=24, horizon=6, n_test=240):
    """Scaled windows to forecast `horizon` months ahead from `window` months; the last `n_test` targets are the test.
    Return X_train, y_train, X_test, y_test (float32 tensors, X of shape (N, window, 1)) and (lo, hi), the min-max scaler."""
    lo, hi = series[:-n_test].min(), series[:-n_test].max()     # 3. the scaler sees the training period only
    scaled = (series - lo) / (hi - lo)
    blocks = np.lib.stride_tricks.sliding_window_view(scaled, window + horizon)
    X = blocks[:, :window]                                       # 1. the first `window` values: the target stays outside
    y = blocks[:, -1]
    cut = len(y) - n_test                                        # 2. a cut in time, no shuffling
    X_train, X_test, y_train, y_test = X[:cut], X[cut:], y[:cut], y[cut:]
    to_tensor = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    return to_tensor(X_train)[..., None], to_tensor(y_train), to_tensor(X_test)[..., None], to_tensor(y_test), (lo, hi)


class Forecaster21(nn.Module):
    """An LSTM that reads a window (N, T, 1) and predicts one value per window (N,)."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)    # 4. the windows arrive as (N, T, 1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)'''

CHECK_21 = r'''def fit_21(model, X, y, epochs=15, lr=3e-3, seed=0):
    """Given: Adam, MSE, mini-batches of 64 in a new random order at each epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(X), generator=generator)
        for start in range(0, len(X), 64):
            batch = order[start:start + 64]
            optimizer.zero_grad()
            nn.functional.mse_loss(model(X[batch]), y[batch]).backward()
            optimizer.step()
    return model


def point_21(k, ok, success, failure):
    """One diagnostic of the chain: ⏳ while an earlier point fails."""
    if not all(STATE_21):
        print(f"⏳ Ex 22.21 point {k} : corrige d'abord le point {STATE_21.index(False) + 1}.")
        STATE_21.append(False)
        return
    verdict("22.21", ok, f"point {k} : {success}", f"point {k} : {failure}")
    STATE_21.append(bool(ok))


with checked("22.21"):
    parts_21 = prepare_21(SPOTS)
    if returned("22.21", "prepare_21", parts_21):
        X_tr_21, y_tr_21, X_te_21, y_te_21, (lo_21, hi_21) = parts_21
        torch.manual_seed(2221)
        model_21 = fit_21(Forecaster21(), X_tr_21, y_tr_21)
        with torch.no_grad():
            pred_21 = model_21(X_te_21)
        unit_21 = hi_21 - lo_21
        rmse_21 = float(torch.sqrt(torch.mean((pred_21 - y_te_21) ** 2))) * unit_21
        pers_21 = float(torch.sqrt(torch.mean((X_te_21[:, -1, 0] - y_te_21) ** 2))) * unit_21
        print(f"Le pipeline actuel : RMSE de test {rmse_21:.1f} taches · persistance {pers_21:.1f} taches.")

        ramp_21 = np.arange(400.0)                    # a ramp: every diagnostic can read where each value comes from
        X_a, y_a, X_b, y_b, (lo_a, hi_a) = (np.asarray(v, dtype=float) if not isinstance(v, tuple) else v
                                            for v in prepare_21(ramp_21, window=24, horizon=6, n_test=40))
        STATE_21 = []
        with np.errstate(divide="ignore", invalid="ignore"):
            gap_21 = np.concatenate([(y_a - X_a[:, -1, 0]) / (X_a[:, 1, 0] - X_a[:, 0, 0]),
                                     (y_b - X_b[:, -1, 0]) / (X_b[:, 1, 0] - X_b[:, 0, 0])])
        point_21(1, np.allclose(gap_21, 6, atol=0.01), "chaque cible est 6 mois après la dernière valeur de sa fenêtre.",
                 "sur une rampe, la cible d'une fenêtre doit être 6 pas après sa dernière valeur : relis la "
                 "construction de X et de y.")
        point_21(2, len(y_b) == 40 and np.all(np.diff(y_a) > 0) and np.all(np.diff(y_b) > 0) and y_a.max() < y_b.min(),
                 "les fenêtres d'entraînement précèdent toutes celles du test, dans l'ordre du temps.",
                 "sur une rampe, les cibles d'entraînement doivent être toutes avant celles du test, dans l'ordre : "
                 "regarde comment le découpage choisit les fenêtres.")
        point_21(3, np.isclose(lo_a, 0) and np.isclose(hi_a, 359),
                 "le min et le max du scaler viennent de la période d'entraînement seulement.",
                 f"sur une rampe de 0 à 399 dont les 40 dernières valeurs sont le test, le scaler a (lo, hi) = "
                 f"({float(lo_a):g}, {float(hi_a):g}) : d'où viennent ces deux nombres ?")
        probe_21 = Forecaster21()
        x_probe_21 = torch.randn(5, 24, 1)
        with torch.no_grad():
            base_21 = probe_21(x_probe_21)
            moved_21 = x_probe_21.clone()
            moved_21[0] += 1.0                         # change the first window only
            changed_21 = (probe_21(moved_21) - base_21).abs() > 1e-7
        point_21(4, tuple(base_21.shape) == (5,) and bool(changed_21[0]) and not bool(changed_21[1:].any()),
                 "la prédiction d'une fenêtre ne dépend que de cette fenêtre.",
                 f"en changeant la seule fenêtre 0 d'un mini-batch de 5, les prédictions qui bougent sont "
                 f"{changed_21.nonzero().flatten().tolist()} : attendu [0] ; demande-toi quel axe le LSTM lit comme le temps.")'''

TODO_22 = r'''output_shape_22 = ...   # a) the shape of output, as a string like "(3, 5, 1)"
h_n_shape_22 = ...      # b) the shape of h_n
params_22 = ...         # c) the number of parameters of lstm_22 (a whole number)
index_22 = ...          # d) the index k such that h_n[k] is the final state of the LAST layer, FORWARD direction
time_22 = ...           # e) the time index t such that output[:, t, 20:] equals h_n[5] (backward direction of the last layer)


def final_state_22(h_n, num_layers, bidirectional):
    """The final state of the last layer: h_n[-1] of shape (N, H) for one direction, the concatenation
    [forward | backward] of shape (N, 2H) for a bidirectional network."""
    raise NotImplementedError("final_state_22")


def lstm_params_22(input_size, hidden_size, num_layers=1, bidirectional=False):
    """The number of parameters of nn.LSTM(input_size, hidden_size, num_layers, bidirectional=...)."""
    raise NotImplementedError("lstm_params_22")'''

SOLUTION_22 = r'''output_shape_22 = "(4, 7, 40)"      # (N, T, 2H): both directions at each step, concatenated
h_n_shape_22 = "(6, 4, 20)"         # (num_layers * 2, N, H): never batch-first
params_22 = 24_960
index_22 = 4                        # layer 2 (the third), direction 0: 2 * 2 + 0
time_22 = 0                         # the backward direction finishes at the first step


def final_state_22(h_n, num_layers, bidirectional):
    """The final state of the last layer: h_n[-1] of shape (N, H) for one direction, the concatenation
    [forward | backward] of shape (N, 2H) for a bidirectional network."""
    if not bidirectional:
        return h_n[-1]
    last = h_n.view(num_layers, 2, *h_n.shape[1:])[-1]     # (2, N, H): forward, then backward
    return torch.cat([last[0], last[1]], dim=1)


def lstm_params_22(input_size, hidden_size, num_layers=1, bidirectional=False):
    """The number of parameters of nn.LSTM(input_size, hidden_size, num_layers, bidirectional=...)."""
    directions = 2 if bidirectional else 1
    total = 0
    for layer in range(num_layers):
        d = input_size if layer == 0 else directions * hidden_size      # a layer reads the outputs of the one below
        total += directions * 4 * (d * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
    return total'''

CHECK_22 = r'''torch.manual_seed(2222)
lstm_22 = nn.LSTM(10, 20, num_layers=3, bidirectional=True, batch_first=True)
x_22 = torch.randn(4, 7, 10)
with torch.no_grad():
    output_22, (h_n_22, c_n_22) = lstm_22(x_22)
wb.check("22.22a", shape_text(output_shape_22))
wb.check("22.22b", shape_text(h_n_shape_22))
wb.check("22.22c", params_22)
wb.check("22.22d", index_22)
wb.check("22.22e", time_22)
if filled(output_shape_22, h_n_shape_22, params_22, index_22, time_22):
    print(f"output {tuple(output_22.shape)} · h_n {tuple(h_n_22.shape)} · c_n {tuple(c_n_22.shape)} · {n_params(lstm_22)} paramètres")
with checked("22.22"):
    final_22 = final_state_22(h_n_22, 3, True)
    same("22.22", "final_state_22(h_n, 3, True) contre [output[:, -1, :20] | output[:, 0, 20:]]", final_22,
         torch.cat([output_22[:, -1, :20], output_22[:, 0, 20:]], dim=1))
    torch.manual_seed(2223)
    gru_22 = nn.GRU(5, 8, num_layers=2, batch_first=True)
    with torch.no_grad():
        out_gru_22, h_gru_22 = gru_22(torch.randn(3, 6, 5))
    same("22.22", "final_state_22(h_n, 2, False) d'un GRU à deux couches contre output[:, -1]",
         final_state_22(h_gru_22, 2, False), out_gru_22[:, -1])
    configs_22 = [(10, 20, 3, True), (5, 8, 2, False), (3, 4, 1, True), (7, 6, 1, False)]
    counts_22 = [lstm_params_22(*cfg) for cfg in configs_22]
    want_counts_22 = [n_params(nn.LSTM(d, h, num_layers=l, bidirectional=b)) for d, h, l, b in configs_22]
    verdict("22.22", counts_22 == want_counts_22, "lstm_params_22 retrouve le nombre de paramètres de quatre nn.LSTM.",
            f"lstm_params_22 rend {counts_22} pour (D, H, couches, bidirectionnel) = {configs_22} : attendu {want_counts_22}.")'''

RECORD_22 = r'''wb.record("22.22a", "(4, 7, 40)", mistakes={"sans batch_first, l'ordre serait (T, N, ...) : ici batch_first=True": "(7, 4, 40)",
          "les deux sens sont concaténés à chaque pas": "(4, 7, 20)", "output ne garde que la dernière couche": "(4, 7, 120)"})
wb.record("22.22b", "(6, 4, 20)", mistakes={"h_n garde H valeurs par ligne : les deux sens sont rangés sur le premier axe, jamais concaténés": "(6, 4, 40)",
          "h_n garde la forme (couches × sens, N, H), même avec batch_first=True": "(4, 6, 20)",
          "une ligne par couche ET par sens": "(3, 4, 20)", "h_n ne concatène pas les deux sens : il les range sur le premier axe": "(3, 4, 40)"})
wb.record("22.22c", 24_960, mistakes={"les couches 2 et 3 lisent les deux sens de la couche du dessous : D = 2H": 3 * 2 * 4 * (10 * 20 + 400 + 40),
          "un réseau bidirectionnel a deux jeux de poids par couche": 24_960 // 2,
          "PyTorch compte deux vecteurs de biais par couche et par sens": 24_960 - 3 * 2 * 4 * 20})
wb.record("22.22d", 4, mistakes={"h_n range les couches à l'extérieur et les sens à l'intérieur : couche × 2 + sens": 2,
          "le sens avant a l'indice 0 dans chaque couche": 5})
wb.record("22.22e", 0, mistakes={"le sens arrière lit la séquence à l'envers : où finit-il ?": 6, "les pas se comptent à partir de 0": 1})'''

TODO_23 = r'''scale_23 = ...                       # the largest monthly value before 1986 (the training period)
X_train_23 = ...                     # make_windows of SPOTS[:I86] / scale_23: 60 months, horizon 6 (the inputs)...
y_train_23 = ...                     # ... and their targets
X_val_23 = ...                       # the windows whose TARGET is in 1986-2005 (indices I86 to I06 - 1), same scale...
y_val_23 = ...                       # ... and their targets
n_train_23 = ...                     # a) the number of training windows
persistence_23 = ...                 # b) the RMSE of persistence on the validation targets, in sunspots (not scaled)


class SunspotLSTM_23(nn.Module):
    """nn.LSTM(1, hidden_size, batch_first=True), then nn.Linear(hidden_size, 1) on the output of the last step:
    windows (N, T, 1) -> predictions (N,)."""

    def __init__(self, hidden_size=32):
        super().__init__()
        raise NotImplementedError("SunspotLSTM_23")

    def forward(self, x):
        raise NotImplementedError("SunspotLSTM_23.forward")


def fit_23(model, X, y, epochs=20, lr=3e-3, batch_size=64, seed=0):
    """Train `model` in place with Adam(lr) and nn.MSELoss, gradient clipping at a norm of 1.0, on mini-batches in a new
    random order at each epoch (torch.randperm with torch.Generator().manual_seed(seed), created once). X (N, T, 1) and
    y (N,) are float32 tensors. Return the list of the mean training loss of each epoch (Python floats)."""
    raise NotImplementedError("fit_23")'''

SOLUTION_23 = r'''scale_23 = SPOTS[:I86].max()
X_train_23, y_train_23 = mylearn.sequences.make_windows(SPOTS[:I86] / scale_23, 60, horizon=6)
X_val_23, y_val_23 = mylearn.sequences.make_windows(SPOTS[I86 - 65:I06] / scale_23, 60, horizon=6)
n_train_23 = len(X_train_23)
persistence_23 = float(np.sqrt(np.mean((SPOTS[I86 - 6:I06 - 6] - SPOTS[I86:I06]) ** 2)))   # the value 6 months earlier


class SunspotLSTM_23(nn.Module):
    """nn.LSTM(1, hidden_size, batch_first=True), then nn.Linear(hidden_size, 1) on the output of the last step:
    windows (N, T, 1) -> predictions (N,)."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)                # many-to-one: the last step only


def fit_23(model, X, y, epochs=20, lr=3e-3, batch_size=64, seed=0):
    """Train `model` in place with Adam(lr) and nn.MSELoss, gradient clipping at a norm of 1.0, on mini-batches in a new
    random order at each epoch (torch.randperm with torch.Generator().manual_seed(seed), created once). X (N, T, 1) and
    y (N,) are float32 tensors. Return the list of the mean training loss of each epoch (Python floats)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        total = 0.0
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X[batch]), y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(X))
    return history'''

CHECK_23 = r'''wb.check("22.23a", n_train_23)
wb.check("22.23b", persistence_23)
model_23 = None
with checked("22.23"):
    ok_data_23 = filled(scale_23, X_train_23, y_train_23, X_val_23, y_val_23)
    if not ok_data_23:
        print("⏳ Ex 22.23 : prépare d'abord scale_23 et les fenêtres.")
    else:
        Xv_23, yv_23 = np.asarray(X_val_23), np.asarray(y_val_23)
        same("22.23", "les cibles de validation, remises à l'échelle, contre SPOTS[I86:I06]", yv_23 * scale_23, SPOTS[I86:I06],
             rtol=1e-5)                                         # float32 windows are fine
        same("22.23", "la dernière valeur de chaque fenêtre de validation : 6 mois avant sa cible",
             Xv_23[:, -1, 0] * scale_23, SPOTS[I86 - 6:I06 - 6], rtol=1e-5)
        verdict("22.23", np.asarray(y_train_23).max() * scale_23 <= SPOTS[:I86].max() and len(y_train_23) + 65 == I86,
                "les fenêtres d'entraînement, cibles comprises, restent avant 1986.",
                "les fenêtres d'entraînement doivent rester avant 1986, cible comprise : prends-les dans SPOTS[:I86].")
        torch.manual_seed(2223)
        model_23 = SunspotLSTM_23()
        to32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        history_23 = fit_23(model_23, to32(X_train_23), to32(y_train_23))
        if returned("22.23", "fit_23", history_23):
            verdict("22.23", isinstance(history_23, list) and len(history_23) == 20 and history_23[-1] < history_23[0],
                    "fit_23 rend 20 loss moyennes, qui baissent.",
                    f"fit_23 doit rendre une liste de 20 loss moyennes qui baissent ; reçu {history_23!r:.120}.")
            model_23.eval()
            with torch.no_grad():
                pred_23 = model_23(to32(Xv_23)).numpy().astype(float) * scale_23
            rmse_23 = float(np.sqrt(np.mean((pred_23 - SPOTS[I86:I06]) ** 2)))
            ratio_23 = rmse_23 / float(np.sqrt(np.mean((SPOTS[I86 - 6:I06 - 6] - SPOTS[I86:I06]) ** 2)))
            print(f"RMSE de validation (1986-2005) : LSTM {rmse_23:.1f} taches · rapport à la persistance {ratio_23:.2f}")
            verdict("22.23", ratio_23 < 0.9, "le LSTM bat la persistance de plus de 10 % en validation.",
                    f"rapport {ratio_23:.2f} à la persistance : attendu moins de 0,9 ; relis le contrat de fit_23 et la mise à l'échelle.")
            fig, ax = plt.subplots(figsize=(10, 3.2))
            dates_23 = SUNSPOTS["date"].iloc[I86:I06]
            ax.plot(dates_23, SPOTS[I86:I06], lw=0.8, color="0.5", label="valeur réelle")
            ax.plot(dates_23, SPOTS[I86 - 6:I06 - 6], lw=1, label="persistance (6 mois avant)")
            ax.plot(dates_23, pred_23, lw=1.5, label="LSTM")
            ax.set_ylabel("taches (moyenne mensuelle)")
            ax.set_title("Prévision à 6 mois sur la validation 1986-2005")
            ax.legend()
            plt.show()'''

RECORD_23 = r'''pers_h1_23 = float(np.sqrt(np.mean((SPOTS[I86 - 1:I06 - 1] - SPOTS[I86:I06]) ** 2)))
wb.record("22.23a", n_train_23, mistakes={"N = (T - w - h) // s + 1 : n'oublie pas le + 1": n_train_23 - 1,
          "l'horizon vaut 6 : la cible est 6 mois après la fin de la fenêtre": I86 - 60})
wb.record("22.23b", persistence_23, decimals=1, mistakes={"la persistance à 6 mois répète la valeur d'il y a 6 mois, pas celle du mois précédent": pers_h1_23,
          "en taches, pas à l'échelle de l'entraînement": persistence_23 / scale_23})'''

GIVEN_24 = r'''CHARS_24 = sorted(set(HOLMES))                               # a character vocabulary of its own (88 characters)
INDEX_24 = {c: i for i, c in enumerate(CHARS_24)}
ids_24 = np.array([INDEX_24[c] for c in HOLMES])
counts_24 = np.ones((len(CHARS_24), len(CHARS_24)))            # +1: no character is impossible after another
np.add.at(counts_24, (ids_24[:-1], ids_24[1:]), 1)
LOGITS_24 = np.log(counts_24 / counts_24.sum(axis=1, keepdims=True))   # row c: the logits of the character after c
WORDS_24 = set(re.findall(r"[a-z]+", HOLMES.lower()))           # every word of Holmes, lower case


def generate_bigram_24(temperature, n_chars=2000, seed=24):
    """2 000 characters drawn one after the other by YOUR sample_from_logits, from the bigram logits, starting at "T"."""
    rng = np.random.default_rng(seed)
    current, out = INDEX_24["T"], ["T"]
    for _ in range(n_chars):
        current = int(mylearn.sequences.sample_from_logits(LOGITS_24[current], temperature=temperature, rng=rng))
        out.append(CHARS_24[current])
    return "".join(out)'''

TODO_24 = r'''greedy_distinct_24 = ...   # a) temperature 0: how many DISTINCT characters among the 2 000 generated?
                           #    "A" 1 to 10, "B" 11 to 40, "C" 41 to 80, "D" more than 80
hot_distinct_24 = ...      # b) same question at temperature 2 (same four letters)
cool_real_24 = ...         # c) temperature 0.3: the share of generated words that exist in Holmes
                           #    "A" less than 50 %, "B" 50 to 90 %, "C" more than 90 %'''

SOLUTION_24 = r'''greedy_distinct_24, hot_distinct_24, cool_real_24 = "A", "D", "B"'''

EXPERIMENT_24 = r'''with checked("22.24"):
    for temperature_24 in (0, 0.3, 1.0, 2.0):
        text_24 = generate_bigram_24(temperature_24)
        words_24 = re.findall(r"[a-z]+", text_24.lower())
        real_24 = np.mean([w in WORDS_24 for w in words_24]) if words_24 else 0.0
        print(f"T = {temperature_24} · {len(set(text_24))} caractères distincts · {real_24:.0%} de mots qui existent")
        print("   ", repr(text_24[:110]))
wb.check("22.24a", greedy_distinct_24)
wb.check("22.24b", hot_distinct_24)
wb.check("22.24c", cool_real_24)'''

_RANGES_24 = {"A": "de 1 à 10", "B": "de 11 à 40", "C": "de 41 à 80", "D": "plus de 80"}
RECORD_24 = (r'''wb.record("22.24a", "A", mistakes=''' + letters("A", {
    k: f"{_RANGES_24[k]} caractères à température 0 ? Le même caractère suit toujours le même caractère : combien de "
       "caractères avant que la suite se répète ?" for k in "BCD"}) + r''')
wb.record("22.24b", "D", mistakes=''' + letters("D", {
    k: f"{_RANGES_24[k]} caractères à température 2 ? Les logits sont divisés par 2 : un caractère rare garde-t-il "
       "une chance sur 2 000 tirages ?" for k in "ABC"}) + r''')
wb.record("22.24c", "B", mistakes=''' + letters("B", {
    "A": "à 0,3, le modèle suit surtout les paires de lettres les plus fréquentes : quels mots courts en sortent ?",
    "C": "un modèle de paires de lettres ne connaît pas les mots : peut-il n'écrire que des mots qui existent ?"}) + r''')''')


PART_C = Part("C", "Échantillonner, valider une série sans fuite, empiler les couches, prévoir les taches solaires",
              "Tu termines `sequences.py` avec la température et l'échantillonnage (22.20) ; tu répares une prévision "
              "trop belle pour être vraie (22.21) ; tu lis les formes que rendent les RNN profonds et bidirectionnels "
              "de PyTorch (22.22) ; tu prévois les taches solaires à 6 mois avec un LSTM, face à la persistance "
              "(22.23) ; puis tu prédis l'allure d'un texte selon sa température (22.24).",
              exercises=[
    Ex("22.20", "🔨", 2, 25, "temperature_softmax et sample_from_logits",
       "diviser des logits par une température et tirer le token suivant, avec ou sans top-k.",
       "✏️ 22.5 · Ex 22.12 · fiche §22.9", thread="Holmes", tracks="R, C", mylearn="sequences.py",
       body=mylearn_howto("sequences.py") + "\n\n" + COMMON_SEQ + r"""

Écris `temperature_softmax(logits, temperature=1.0)` et `sample_from_logits(logits, temperature=1.0, top_k=None, rng=None)`. **Contrat** :
- `temperature_softmax` calcule $\mathrm{softmax}(z / T)$ le long du **dernier** axe (forme gardée), de façon stable : des logits de ±1 000 ne déclenchent aucun `RuntimeWarning` (on retire le maximum de **chaque ligne**) ; $T \leq 0$ lève `ValueError` ;
- `sample_from_logits` rend un `int` Python pour des logits `(V,)`, un tableau `int64` `(N,)` pour des logits `(N, V)`, chaque ligne étant tirée **indépendamment** ; avec `temperature=0`, aucun hasard : le **premier** indice du maximum ; avec `top_k`, les logits strictement plus petits que le $k$-ième plus grand sont retirés avant de tirer, ligne par ligne (les ex aequo du $k$-ième restent, et `top_k = V` garde tout) ; `temperature < 0`, ou un `top_k` qui n'est pas un entier de $[1, V]$, lève `ValueError` ;
- le tirage utilise `rng` (un `np.random.Generator`) et lui seul : la même graine redonne les mêmes ids ; `rng=None` crée un nouveau générateur, sans graine fixe. La façon de tirer (CDF inverse, `rng.choice`…) est libre : les tests ne regardent que les fréquences.

La première cellule compare ta softmax à celle de SciPy, essaie le cas glouton, les fréquences de 20 000 tirages, le top-k avec ex aequo, la reproductibilité et trois refus ; la seconde lance les tests (`-k "test_temperature_softmax_ or test_sample_from_logits_"`).""",
       check=RELOAD + CHECK_20, solution=CHECK_20,
       after=tests_after("22.20", "mylearn.sequences.temperature_softmax([0.0, 1.0])", "TEST_SEQ",
                         "test_temperature_softmax_ or test_sample_from_logits_")),

    Ex("22.21", "🐛", 2, 25, "Série temporelle piégée : fuite, mélange et batch_first",
       "réparer une prévision de série temporelle dont le score de test est trop beau, en suivant la règle des fenêtres de la fiche.",
       "Ex 22.13 · ch. 8 (découpage) · ch. 12 (min-max) · 22.R3 · fiche §22.5, section « Au-delà du livre : prévoir une série temporelle »",
       thread="taches solaires", tracks="R, C",
       body=r"""Un collègue prévoit les taches solaires à 6 mois à partir des 24 derniers mois, avec un petit LSTM. Ses 240 derniers mois de cibles servent de test, et il annonce un résultat spectaculaire. Sa fonction `prepare_21` et sa classe `Forecaster21` contiennent **quatre** bugs : trois fuites de données et un mauvais ordre des axes. Corrige-les **sur place**, dans la cellule, sans changer leurs signatures, la taille de fenêtre, l'horizon ni l'architecture.

Le contrat que la version corrigée doit tenir (fiche, « Au-delà du livre », règles 2 et 3) : la cible de chaque fenêtre est la valeur `horizon` mois **après** sa dernière valeur ; les `n_test` dernières cibles sont le test, et le découpage suit le temps (aucun mélange) : une fenêtre de test peut lire la fin de la période d'entraînement, mais aucune fenêtre d'entraînement ne touche une cible de test ; le min-max s'ajuste sur la période d'entraînement, c'est-à-dire tout ce qui précède les `n_test` dernières valeurs ; le LSTM reçoit des fenêtres `(N, T, 1)`.

La vérification entraîne d'abord le pipeline actuel et affiche ses erreurs (le symptôme), puis donne quatre diagnostics, chacun sur une rampe (0, 1, 2…, 399) ou un mini-batch choisis pour qu'on lise d'où vient chaque nombre ; tant qu'un point échoue, les suivants attendent (⏳). Elle s'exécute dès le premier « Tout exécuter » : les diagnostics du code fautif sont ton point de départ.

Dans tes notes : quel bug produit, à lui seul, une persistance d'erreur nulle ? Aucun ne lève d'erreur : lesquels se voient dans les scores affichés, et lesquels seulement dans les diagnostics ? Une fois tout corrigé, le LSTM bat-il encore la persistance ?""",
       todo=TODO_21, check=CHECK_21, solution=SOLUTION_21 + "\n\n\n" + CHECK_21,
       note="Les quatre bugs : `X = blocks[:, -window:]` met la cible dans la fenêtre (la persistance devient exacte, "
            "et le modèle n'a qu'à recopier une entrée) ; `train_test_split` mélange les fenêtres, si bien que les "
            "fenêtres d'entraînement entourent chaque fenêtre de test et partagent presque toutes ses valeurs ; le "
            "min-max voit le maximum de la période de test ; `nn.LSTM(1, H)` sans `batch_first` lit l'axe des "
            "exemples comme le temps, et chaque prédiction mélange les fenêtres du mini-batch, sans aucune erreur. "
            "Corrigé, le petit LSTM sur 24 mois ne bat la persistance que d'une dizaine de pour cent : une fenêtre "
            "courte et 15 epochs ne suffisent pas pour mieux (22.23 fait mieux avec 60 mois)."),

    Ex("22.22", "📦", 2, 25, "RNN profonds et bidirectionnels : formes de output et h_n",
       "lire ce que rendent output et h_n d'un RNN profond et bidirectionnel de PyTorch, et en tirer l'état final de la dernière couche.",
       "Ex 22.15 · fiche §22.8.2 à §22.8.4, encadré 🕰️ de §22.5", tracks="R, C",
       body=r"""On construit `lstm_22 = nn.LSTM(10, 20, num_layers=3, bidirectional=True, batch_first=True)` et on lui donne `x_22`, un mini-batch de 4 séquences de 7 pas, de forme `(4, 7, 10)`. Elle rend `output_22, (h_n_22, c_n_22)`.

**Contrat**
- a) `output_shape_22` et b) `h_n_shape_22` : les formes de `output` et de `h_n`, en chaînes (`"(3, 5, 1)"`) ; c) `params_22` : le nombre de paramètres de `lstm_22` ; d) `index_22` : l'indice $k$ tel que `h_n[k]` est l'état final de la **dernière** couche, sens **avant** ; e) `time_22` : l'indice de temps $t$ tel que `output[:, t, 20:]` (la moitié « arrière » de la sortie) est égal à `h_n[5]` ;
- `final_state_22(h_n, num_layers, bidirectional)` rend l'état final de la dernière couche : `h_n[-1]`, de forme `(N, H)`, pour un seul sens ; la concaténation `[avant | arrière]`, de forme `(N, 2H)`, pour un réseau bidirectionnel (c'est ce que lit une couche dense dans une structure many-to-one) ;
- `lstm_params_22(input_size, hidden_size, num_layers=1, bidirectional=False)` rend le nombre de paramètres de la `nn.LSTM` correspondante.

Réponds a) à e) **avant** d'exécuter la vérification, d'après la documentation de `nn.LSTM` et la fiche (sans lancer `lstm_22`) ; la cellule vérifie tes réponses, puis affiche les vraies formes, et compare tes deux fonctions à PyTorch.

Dans tes notes : pourquoi `output[:, -1]` n'est-il pas l'état final d'un réseau bidirectionnel ? Combien de paramètres aurait ce réseau avec un seul sens ?""",
       todo=TODO_22, check=CHECK_22, solution=SOLUTION_22 + "\n\n\n" + CHECK_22, record=RECORD_22,
       note="`output` a la forme `(N, T, 2H)` = `(4, 7, 40)` : à chaque pas, les états des deux sens de la **dernière** "
            "couche, concaténés ; `h_n` a la forme `(couches × 2, N, H)` = `(6, 4, 20)`, même avec `batch_first=True`, "
            "rangée couche par couche puis sens par sens : l'indice $2\\ell + s$. Le sens arrière lit la séquence à "
            "l'envers : il finit au pas 0, d'où `output[:, 0, 20:] == h_n[5]` ; `output[:, -1]` mélange donc l'état "
            "final du sens avant et le **premier** état du sens arrière. Paramètres : $2 \\times 4\\,(10 \\cdot 20 + "
            "20^2 + 40) = 5\\,120$ pour la première couche, $2 \\times 4\\,(40 \\cdot 20 + 20^2 + 40) = 9\\,920$ pour "
            "chacune des deux autres, qui lisent les $2H = 40$ valeurs de la couche du dessous : 24 960 en tout ; "
            "avec un seul sens, $2\\,560 + 2 \\times 4\\,(20 \\cdot 20 + 400 + 40) = 9\\,280$."),

    Ex("22.23", "📦", 3, 45, "Prévoir les taches solaires avec nn.LSTM, face à la persistance",
       "préparer une série sans fuite, entraîner un LSTM many-to-one et le juger face à la persistance.",
       "Ex 22.13 · Ex 22.21 · ch. 20 (boucle d'entraînement) · ✏️ 22.1 · fiche §22.5, §22.8.1, « Au-delà du livre »",
       thread="taches solaires", tracks="R, C",
       body=r"""On prévoit la moyenne mensuelle des taches solaires **6 mois** à l'avance, à partir des **60** mois qui précèdent (cinq ans). L'entraînement utilise tout ce qui précède janvier 1986 (`SPOTS[:I86]`), la validation les cibles de janvier 1986 à décembre 2005 (`SPOTS[I86:I06]`) ; les mois suivants restent cachés pour le défi 🏆 22.29.

**Contrat**
- `scale_23` : la plus grande valeur mensuelle avant 1986 ; toutes les fenêtres sont divisées par `scale_23` (un min-max dont le minimum vaut 0, ajusté sur l'entraînement) ;
- `X_train_23, y_train_23` : les fenêtres de 60 mois et leurs cibles à 6 mois, prises **dans** `SPOTS[:I86]` (ta `make_windows`) ;
- `X_val_23, y_val_23` : toutes les fenêtres dont la **cible** tombe entre `I86` et `I06 - 1` (une par mois de validation) ; leurs entrées peuvent lire la fin de 1985 ;
- a) `n_train_23` : le nombre de fenêtres d'entraînement ; b) `persistence_23` : la RMSE de la persistance (prédire la valeur d'il y a 6 mois) sur les 240 mois de validation, en taches, pas à l'échelle ;
- `SunspotLSTM_23(hidden_size=32)` : un `nn.LSTM(1, hidden_size, batch_first=True)`, puis un `nn.Linear(hidden_size, 1)` sur la sortie du dernier pas (many-to-one) ; il transforme `(N, T, 1)` en `(N,)` ;
- `fit_23(model, X, y, epochs=20, lr=3e-3, batch_size=64, seed=0)` entraîne en place avec `Adam(lr)`, `nn.MSELoss()` et `clip_grad_norm_` à 1,0 (fiche §22.6), dans un nouvel ordre à chaque epoch (`torch.randperm` et un seul `torch.Generator().manual_seed(seed)`) ; elle rend la liste des loss moyennes de chaque epoch (`float` Python).

La vérification contrôle tes fenêtres de validation, entraîne ton modèle (quelques secondes), affiche sa RMSE et son rapport à la persistance (attendu : moins de 0,9), et trace les deux prévisions sur 1986-2005.

Dans tes notes : le LSTM bat-il la persistance partout, ou surtout à certains moments du cycle (regarde la figure) ? Pourquoi n'a-t-on pas le droit de choisir `hidden_size` ou le nombre d'epochs en regardant cette validation, puis d'annoncer cette même RMSE comme performance attendue ?""",
       todo=TODO_23, check=CHECK_23, solution=SOLUTION_23 + "\n\n\n" + CHECK_23, record=RECORD_23,
       note="Le LSTM fait de l'ordre d'un quart de mieux que la persistance sur 1986-2005 (le rapport exact dépend de "
            "la graine) ; il gagne surtout dans les montées et les descentes du cycle, où la valeur d'il y a 6 mois "
            "est la plus fausse, et beaucoup moins au sommet, où le bruit mensuel domine. Choisir des réglages sur "
            "cette validation, puis annoncer sa RMSE, c'est la même fuite que choisir un modèle sur le test : la RMSE "
            "annoncée serait optimiste. D'où le test caché du défi 22.29, ouvert une seule fois."),

    Ex("22.24", "🔮", 1, 10, "Température : prédire l'allure du texte généré",
       "prédire l'effet de la température sur un générateur de texte, puis le mesurer avec ton sample_from_logits.",
       "Ex 22.20 · ✏️ 22.5 · fiche §22.9", thread="Holmes", tracks="C", hypothesis=True,
       body=r"""Le plus simple des générateurs de texte : un **modèle de paires de lettres** (bigrammes, ch. 6). Pour chaque caractère de Holmes, on compte les caractères qui le suivent ; les logits du caractère suivant sont les logarithmes de ces fréquences (avec +1 partout, pour qu'aucun caractère ne soit impossible). La cellule suivante le construit et définit `generate_bigram_24(temperature)`, qui tire 2 000 caractères l'un après l'autre avec **ton** `sample_from_logits` (22.20), en partant de « T ».

Prédis, **avant** d'exécuter :
a) `greedy_distinct_24` : à la température 0, combien de caractères **distincts** parmi les 2 000 ? `"A"` de 1 à 10, `"B"` de 11 à 40, `"C"` de 41 à 80, `"D"` plus de 80 ;
b) `hot_distinct_24` : même question à la température 2 (les mêmes quatre lettres) ;
c) `cool_real_24` : à la température 0,3, la part des mots générés qui existent dans Holmes : `"A"` moins de 50 %, `"B"` de 50 à 90 %, `"C"` plus de 90 %.

L'expérience génère à 0 ; 0,3 ; 1 et 2, affiche pour chaque température le nombre de caractères distincts, la part de mots qui existent (un mot : une suite de lettres a à z) et le début du texte.""",
       given=GIVEN_24, todo=TODO_24, solution=SOLUTION_24,
       after=[("code", guarded(EXPERIMENT_24, ["greedy_distinct_24", "hot_distinct_24", "cool_real_24"],
                               "⏳ Ex 22.24 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Laquelle des quatre "
                     "températures choisirais-tu pour produire un texte qui « ressemble à de l'anglais », et que "
                     "faudrait-il au modèle pour aller plus loin que des mots courts ?")],
       record=RECORD_24,
       note="À la température 0, le choix est déterministe : après « T » vient « h », puis « e », puis l'espace, puis "
            "« t »… et la suite « the the the » se répète : une poignée de caractères. À 2, les logits divisés par 2 "
            "rapprochent les probabilités de l'uniforme : en 2 000 tirages, les 88 caractères sortent, "
            "chiffres et ponctuation compris. À 0,3, le modèle suit surtout les paires les plus fréquentes et écrit "
            "beaucoup de petits mots qui existent (« the », « he », « a »), mais aussi des fragments : un modèle "
            "de paires de lettres ne connaît pas les mots, il faut plus de contexte (le LSTM de 22.25 lit 40 caractères)."),
])

# ---------------------------------------------------------------------------
# Part D: the character generator, Holmes against Verne, the CNN-LSTM, the challenge (22.25 to 22.29)
# ---------------------------------------------------------------------------
GIVEN_25 = r'''WINDOW_25 = 40                                               # the network reads 40 characters...
STRIDE_25 = 3                                                # ...and a window starts every 3 characters
CUT_25 = int(0.9 * len(HOLMES))                              # the last 10 % of Holmes: the validation
N_TRAIN_25 = 240_000 if FAST_MODE else CUT_25                # characters of training text
EPOCHS_25 = 2 if FAST_MODE else 20
TARGET_25 = 2.1 if FAST_MODE else 1.6                        # validation loss expected (nats per character)
SEED_25 = "To Sherlock Holmes she is always the wom"          # 40 characters: the start of the first story
print(f"training on {N_TRAIN_25:_} characters, {EPOCHS_25} epochs · validation on {len(HOLMES) - CUT_25:_} characters".replace("_", " "))'''

TODO_25 = r'''class CharLSTM_25(nn.Module):
    """Character-level generator: ids (N, T) int64 -> one-hot (N, T, V) -> nn.LSTM(V, hidden_size, num_layers,
    batch_first=True) -> nn.Linear(hidden_size, V) on the output of the LAST step -> logits (N, V)."""

    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super().__init__()
        raise NotImplementedError("CharLSTM_25")

    def forward(self, ids):
        raise NotImplementedError("CharLSTM_25.forward")


def train_25(model, X, y, epochs=2, batch_size=128, seed=0):
    """Train `model` in place on the windows X (N, T) int64 and their next characters y (N,) int64: cross-entropy,
    RMSprop(lr=0.01, alpha=0.9), gradient clipping at a norm of 1.0, mini-batches in a new random order at each epoch
    (torch.randperm with one torch.Generator().manual_seed(seed)), each batch moved to the device of the model.
    Return the list of the mean training loss of each epoch (Python floats)."""
    raise NotImplementedError("train_25")


def generate_25(model, seed_text, n_chars, stoi, itos, temperature=1.0, rng=None):
    """Generate n_chars characters after seed_text, one at a time: the model reads the last 40 characters (all of
    them while there are fewer), mylearn.sequences.sample_from_logits draws the next id. Return the n_chars new
    characters as a str (without seed_text)."""
    raise NotImplementedError("generate_25")'''

SOLUTION_25 = r'''class CharLSTM_25(nn.Module):
    """Character-level generator: ids (N, T) int64 -> one-hot (N, T, V) -> nn.LSTM(V, hidden_size, num_layers,
    batch_first=True) -> nn.Linear(hidden_size, V) on the output of the LAST step -> logits (N, V)."""

    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, ids):
        out, _ = self.lstm(F.one_hot(ids, self.vocab_size).float())   # one-hot: (N, T) -> (N, T, V)
        return self.head(out[:, -1])                                   # many-to-one: the last step only


def train_25(model, X, y, epochs=2, batch_size=128, seed=0):
    """Train `model` in place on the windows X (N, T) int64 and their next characters y (N,) int64: cross-entropy,
    RMSprop(lr=0.01, alpha=0.9), gradient clipping at a norm of 1.0, mini-batches in a new random order at each epoch
    (torch.randperm with one torch.Generator().manual_seed(seed)), each batch moved to the device of the model.
    Return the list of the mean training loss of each epoch (Python floats)."""
    device = next(model.parameters()).device
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(X), generator=generator).split(batch_size):
            xb, yb = X[batch].to(device), y[batch].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(X))
    return history


def generate_25(model, seed_text, n_chars, stoi, itos, temperature=1.0, rng=None):
    """Generate n_chars characters after seed_text, one at a time: the model reads the last 40 characters (all of
    them while there are fewer), mylearn.sequences.sample_from_logits draws the next id. Return the n_chars new
    characters as a str (without seed_text)."""
    device = next(model.parameters()).device
    rng = np.random.default_rng() if rng is None else rng
    ids = list(mylearn.sequences.encode(seed_text, stoi))
    new = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_chars):
            window = torch.tensor([ids[-40:]], device=device)
            logits = model(window)[0].cpu().numpy().astype(np.float64)
            nxt = mylearn.sequences.sample_from_logits(logits, temperature=temperature, rng=rng)
            ids.append(nxt)
            new.append(nxt)
    return mylearn.sequences.decode(np.array(new, dtype=np.int64), itos)'''

CHECK_25 = r'''def val_loss_25(model, X, y, batch_size=2000):
    """Given: the mean cross-entropy (nats per character) of a model on windows X and next characters y."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        total = sum(F.cross_entropy(model(X[i:i + batch_size].to(device)), y[i:i + batch_size].to(device),
                                    reduction="sum").item() for i in range(0, len(X), batch_size))
    return total / len(X)


model_25 = None
with checked("22.25"):
    S = mylearn.sequences
    torch.manual_seed(2225)
    probe_25 = CharLSTM_25(88)
    want_params_25 = n_params(nn.LSTM(88, 128, num_layers=2)) + 128 * 88 + 88
    verdict("22.25", n_params(probe_25) == want_params_25,
            "CharLSTM_25(88) a " + f"{want_params_25:,}".replace(",", " ") + " paramètres : deux couches LSTM de 128 et la couche dense.",
            "CharLSTM_25(88) a " + f"{n_params(probe_25):,}".replace(",", " ") + " paramètres, attendu "
            + f"{want_params_25:,}".replace(",", " ") + " : relis l'architecture du contrat.")
    with torch.no_grad():
        logits_probe_25 = probe_25(torch.randint(0, 88, (3, 40)))
    verdict("22.25", tuple(logits_probe_25.shape) == (3, 88),
            "des ids (3, 40) donnent des logits (3, 88).",
            f"des ids (3, 40) donnent {shape_text(logits_probe_25)} : attendu (3, 88), un vecteur de logits par fenêtre.")

    stoi_25, itos_25 = pair("22.25", "build_vocab(HOLMES)", S.build_vocab(HOLMES), names=("stoi", "itos"))
    ids_25 = S.encode(HOLMES, stoi_25)
    X_25, y_25 = S.make_windows(ids_25[:N_TRAIN_25], WINDOW_25, horizon=1, stride=STRIDE_25)
    Xv_25, yv_25 = S.make_windows(ids_25[CUT_25:], WINDOW_25, horizon=1, stride=STRIDE_25)
    X_25, y_25, Xv_25, yv_25 = (torch.from_numpy(np.ascontiguousarray(a)) for a in (X_25[:, :, 0], y_25, Xv_25[:, :, 0], yv_25))
    print(f"{len(X_25):_} training windows · {len(Xv_25):_} validation windows · {len(itos_25)} characters".replace("_", " "))

    torch.manual_seed(2225)
    model_25 = CharLSTM_25(len(itos_25)).to(DEVICE)
    started_25 = time.perf_counter()
    losses_25 = train_25(model_25, X_25, y_25, epochs=EPOCHS_25)
    if returned("22.25", "train_25", losses_25):
        val_25 = val_loss_25(model_25, Xv_25, yv_25)
        print(f"training: {time.perf_counter() - started_25:.0f} s · mean training loss per epoch "
              f"{[round(v, 3) for v in losses_25]} · validation loss {val_25:.3f} nats")
        verdict("22.25", isinstance(losses_25, list) and len(losses_25) == EPOCHS_25,
                f"train_25 rend {EPOCHS_25} loss moyennes.", f"train_25 doit rendre une liste de {EPOCHS_25} loss moyennes.")
        verdict("22.25", val_25 < TARGET_25, f"loss de validation sous {TARGET_25} nats par caractère.",
                f"loss de validation {val_25:.3f} : attendu moins de {TARGET_25} ; relis le contrat de train_25 "
                "(optimiseur, clipping, ordre des mini-batches).")

        text_a_25 = generate_25(model_25, SEED_25, 60, stoi_25, itos_25, temperature=0)
        text_b_25 = generate_25(model_25, SEED_25, 60, stoi_25, itos_25, temperature=0)
        verdict("22.25", isinstance(text_a_25, str) and len(text_a_25) == 60 and text_a_25 == text_b_25,
                "generate_25 rend 60 nouveaux caractères, les mêmes deux fois à température 0.",
                f"generate_25(..., 60, temperature=0) doit rendre une str de 60 caractères, la même à chaque appel ; "
                f"reçu {text_a_25!r:.80}.")
        short_25 = generate_25(model_25, "Wat", 5, stoi_25, itos_25, temperature=1.0, rng=np.random.default_rng(0))
        verdict("22.25", isinstance(short_25, str) and len(short_25) == 5,
                "une amorce de 3 caractères suffit.", "une amorce plus courte que 40 caractères doit fonctionner.")
        for temperature_25 in (0, 0.5, 1.0):
            print(f"\n--- température {temperature_25} ---")
            print(SEED_25 + generate_25(model_25, SEED_25, 300, stoi_25, itos_25, temperature=temperature_25,
                                        rng=np.random.default_rng(25)))'''

# 22.26: the given cell is built from tools/chapters/history_ch22.json (precomputed by history_ch22.py)
_HISTORY_26 = json.loads(Path(__file__).with_name("history_ch22.json").read_text(encoding="utf-8"))


def _given_26() -> str:
    h = _HISTORY_26["history"]
    shown = sorted({1, 3, 10, len(h)})
    lines = ["# The generator of Ex 22.25 in its full mode, 20 epochs on 90 % of Holmes (precomputed: about 25 minutes on a CPU).",
             f"# {_HISTORY_26['n_train_windows']} training windows, {_HISTORY_26['n_val_windows']} validation windows (the last 10 %).",
             "# After each epoch: the mean training loss and the validation loss, in nats per character.",
             f"TRAIN_LOSS_26 = {[e['train_loss'] for e in h]}",
             f"VAL_LOSS_26 = {[e['val_loss'] for e in h]}",
             "SAMPLES_26 = {   # 200 characters generated after the seed, at the temperatures 0 and 0.5, after some epochs"]
    for e in h:
        if e["epoch"] in shown:
            lines.append(f"    {e['epoch']}: {{0.0: {e['samples']['0.0']!r},")
            lines.append(f"         0.5: {e['samples']['0.5']!r}}},")
    lines.append("}")
    lines.append(GIVEN_26_PLOT)
    return "\n".join(lines)


GIVEN_26_PLOT = r'''epochs_26 = np.arange(1, len(VAL_LOSS_26) + 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs_26, TRAIN_LOSS_26, "--o", ms=3, label="entraînement")
ax.plot(epochs_26, VAL_LOSS_26, "-o", ms=3, label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("loss (nats par caractère)")
ax.set_xticks(epochs_26)
ax.legend()
plt.show()
for epoch_26, texts_26 in SAMPLES_26.items():
    print(f"--- après l'epoch {epoch_26} ---")
    for temperature_26, text_26 in texts_26.items():
        print(f"T = {temperature_26} :", repr(text_26[:120]))'''

TODO_26 = r'''best_epoch_26 = ...    # a) the epoch (the first is 1) with the lowest validation loss
perplexity_26 = ...    # b) the validation perplexity at that epoch (2 decimals)
bits_26 = ...          # c) the validation loss at that epoch in bits per character (3 decimals)
unigram_26 = ...       # d) the perplexity of a model WITHOUT context, that predicts every character with its frequency
                       #    in HOLMES (2 decimals)
overfit_26 = ...       # e) "A" no sign of overfitting in 20 epochs; "B" the validation loss stops falling, or rises,
                       #    while the training loss keeps falling; "C" both losses rise at the end'''

SOLUTION_26 = r'''best_epoch_26 = int(np.argmin(VAL_LOSS_26)) + 1               # the epochs count from 1
perplexity_26 = float(np.exp(min(VAL_LOSS_26)))                # the loss is in nats: e to the loss
bits_26 = min(VAL_LOSS_26) / np.log(2)
freqs_26 = np.array(list(Counter(HOLMES).values())) / len(HOLMES)
unigram_26 = float(np.exp(-(freqs_26 * np.log(freqs_26)).sum()))   # e to the entropy (nats), as for the network
overfit_26 = "B"'''

CHECK_26 = r'''wb.check("22.26a", best_epoch_26)
wb.check("22.26b", perplexity_26)
wb.check("22.26c", bits_26)
wb.check("22.26d", unigram_26)
wb.check("22.26e", overfit_26)'''

CHECK_28 = r'''wb.check("22.28a", params_28)
model_28 = None
with checked("22.28"):
    torch.manual_seed(2228)
    model_28 = BallCNNLSTM_28()
    with torch.no_grad():
        out_probe_28 = model_28(VIDEOS_TRAIN_28[:5])
    verdict("22.28", tuple(out_probe_28.shape) == (5, 3), "5 vidéos (5, 16, 1, 32, 32) donnent des logits (5, 3).",
            f"5 vidéos (5, 16, 1, 32, 32) donnent {shape_text(out_probe_28)} : attendu (5, 3).")
    verdict("22.28", n_params(model_28) == 21_043, "21 043 paramètres, comme le contrat.",
            f"{n_params(model_28):_} paramètres : le contrat en donne 21 043 ; relis les tailles des couches.".replace("_", " "))
    with torch.no_grad():                                      # a frame must be read alone by the convolution
        moved_28 = VIDEOS_TRAIN_28[:2].clone()
        moved_28[1] = VIDEOS_TRAIN_28[2]
        same_first_28 = torch.allclose(model_28(moved_28)[0], out_probe_28[0], atol=1e-6)
    verdict("22.28", same_first_28, "la prédiction d'une vidéo ne dépend que de cette vidéo.",
            "changer la vidéo 1 d'un mini-batch change la prédiction de la vidéo 0 : regarde comment tu remets les "
            "images en (N, T, features) après la convolution.")
    started_28 = time.perf_counter()
    fit_28(model_28, VIDEOS_TRAIN_28, LABELS_TRAIN_28)
    model_28.eval()
    with torch.no_grad():
        acc_28 = (model_28(VIDEOS_TEST_28).argmax(1) == LABELS_TEST_28).float().mean().item()
        order_28 = torch.randperm(16, generator=torch.Generator().manual_seed(1))
        acc_shuffled_28 = (model_28(VIDEOS_TEST_28[:, order_28]).argmax(1) == LABELS_TEST_28).float().mean().item()
        acc_reversed_28 = (model_28(VIDEOS_TEST_28.flip(1)).argmax(1) == LABELS_TEST_28).float().mean().item()
    print(f"training: {time.perf_counter() - started_28:.0f} s")
    print(f"test accuracy: {acc_28:.3f} · frames shuffled (same order for every video): {acc_shuffled_28:.3f} · "
          f"frames reversed: {acc_reversed_28:.3f}")
    verdict("22.28", acc_28 >= 0.9, "au moins 90 % d'accuracy sur les 600 vidéos de test.",
            f"accuracy {acc_28:.3f} : attendu au moins 0,9 ; relis le contrat (le LSTM lit la sortie du dernier pas).")'''

GIVEN_28 = r'''def make_ball_videos_28(n, frames=16, size=32, noise=0.15, seed=0):
    """n synthetic videos (n, frames, 1, size, size) of a ball crossing the image from left to right, and their labels:
    0 the ball goes up, 1 it goes down, 2 it flies flat. The middle height is random, so one frame alone cannot tell."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    t = np.arange(frames)
    xs = rng.uniform(3, 8, n)[:, None] + rng.uniform(0.9, 1.4, n)[:, None] * t           # left to right
    slope = rng.uniform(0.5, 1.0, n) * np.select([labels == 0, labels == 1], [-1, 1], 0)  # rows grow downwards
    ys = rng.uniform(10, 22, n)[:, None] + slope[:, None] * (t - (frames - 1) / 2)
    grid = np.arange(size)
    gx = np.exp(-(grid - xs[..., None]) ** 2 / 2)                                           # (n, frames, size)
    gy = np.exp(-(grid - ys[..., None]) ** 2 / 2)
    videos = gy[..., :, None] * gx[..., None, :] + noise * rng.normal(size=(n, frames, size, size))
    return torch.tensor(videos[:, :, None], dtype=torch.float32), torch.tensor(labels)


def fit_28(model, X, y, epochs=8, lr=3e-3, seed=0):
    """Given: Adam, cross-entropy, mini-batches of 64 in a new random order at each epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for batch in torch.randperm(len(X), generator=generator).split(64):
            optimizer.zero_grad()
            F.cross_entropy(model(X[batch]), y[batch]).backward()
            optimizer.step()
    return model


VIDEOS_TRAIN_28, LABELS_TRAIN_28 = make_ball_videos_28(1500, seed=28)
VIDEOS_TEST_28, LABELS_TEST_28 = make_ball_videos_28(600, seed=29)
fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
for row_28, label_28 in enumerate((0, 1, 2)):
    video_28 = VIDEOS_TRAIN_28[int(np.flatnonzero(LABELS_TRAIN_28.numpy() == label_28)[0])]
    for col_28, frame_28 in enumerate(range(0, 16, 2)):
        axes[row_28, col_28].imshow(video_28[frame_28, 0], cmap="gray")
        axes[row_28, col_28].set_title(f"{['monte', 'descend', 'à plat'][label_28]} · image {frame_28}", fontsize=7)
        axes[row_28, col_28].axis("off")
plt.tight_layout()
plt.show()'''

TODO_28 = r'''params_28 = ...    # the number of parameters of the network of the contract (compute it before writing the class)


class BallCNNLSTM_28(nn.Module):
    """Videos (N, T, 1, 32, 32) -> logits (N, 3). Each frame alone: nn.Conv2d(1, 8, 5, stride=2, padding=2), ReLU,
    nn.AdaptiveMaxPool2d(4), flattened to 128 values; the T vectors of a video: nn.LSTM(128, 32, batch_first=True);
    the output of the last step: nn.Linear(32, 3)."""

    def __init__(self):
        super().__init__()
        raise NotImplementedError("BallCNNLSTM_28")

    def forward(self, videos):
        raise NotImplementedError("BallCNNLSTM_28.forward")'''

SOLUTION_28 = r'''params_28 = (8 * 1 * 5 * 5 + 8) + 4 * (128 * 32 + 32 * 32 + 2 * 32) + (32 * 3 + 3)   # 208 + 20 736 + 99


class BallCNNLSTM_28(nn.Module):
    """Videos (N, T, 1, 32, 32) -> logits (N, 3). Each frame alone: nn.Conv2d(1, 8, 5, stride=2, padding=2), ReLU,
    nn.AdaptiveMaxPool2d(4), flattened to 128 values; the T vectors of a video: nn.LSTM(128, 32, batch_first=True);
    the output of the last step: nn.Linear(32, 3)."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(), nn.AdaptiveMaxPool2d(4), nn.Flatten())
        self.lstm = nn.LSTM(128, 32, batch_first=True)
        self.head = nn.Linear(32, 3)

    def forward(self, videos):
        n, t = videos.shape[:2]
        frames = videos.reshape(n * t, *videos.shape[2:])       # every frame of every video: (N T, 1, 32, 32)
        features = self.cnn(frames).reshape(n, t, -1)           # back to one sequence per video: (N, T, 128)
        out, _ = self.lstm(features)
        return self.head(out[:, -1])'''

RECORD_28 = r'''wb.record("22.28a", 21_043, mistakes={"PyTorch donne deux vecteurs de biais à chaque porte du LSTM": 21_043 - 4 * 32,
          "le LSTM reçoit 8 × 4 × 4 = 128 valeurs par image, pas 8": 208 + 4 * (8 * 32 + 32 * 32 + 64) + 99,
          "n'oublie pas les biais de la convolution et de la couche dense": 21_043 - 8 - 3})'''

GIVEN_29 = r'''SERIES_29 = SPOTS.copy()                       # a private copy: the evaluation reads this one
GOAL_29, STAR_29 = 0.80, 0.78
HORIZON_29 = 6


def score_29(make_forecaster, train_end, start, stop):
    """RMSE ratio to persistence of the forecasts of the targets start..stop-1, by a forecaster made from the months
    before train_end. While it runs, the global SPOTS and SUNSPOTS hide every month from train_end on (NaN)."""
    global SPOTS, SUNSPOTS
    saved = SPOTS, SUNSPOTS
    SPOTS = np.where(np.arange(len(SERIES_29)) < train_end, SERIES_29, np.nan)
    SUNSPOTS = saved[1].assign(sunspots=SPOTS)
    try:
        forecast = make_forecaster(SERIES_29[:train_end].copy())
        preds = np.array([float(forecast(SERIES_29[:t - HORIZON_29 + 1].copy())) for t in range(start, stop)])
    finally:
        SPOTS, SUNSPOTS = saved
    truth = SERIES_29[start:stop]
    rmse = np.sqrt(np.mean((preds - truth) ** 2))
    persistence = np.sqrt(np.mean((SERIES_29[start - HORIZON_29:stop - HORIZON_29] - truth) ** 2))
    return rmse / persistence, preds'''

TODO_29 = r'''def make_forecaster_29(train):
    """Fit a forecaster on `train` (the monthly values of a period, a float array) and return forecast(past): the
    value 6 months after the last value of `past` (all the months known so far, at least those of `train`)."""
    raise NotImplementedError("make_forecaster_29")'''

SOLUTION_29 = r'''from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features_29(v):
    """Features of a past v (square-root scale, at least 132 months): the recent level at several scales, the trend
    over a year, and the level 9 to 11 years ago (the cycle lasts about 11 years)."""
    return np.array([v[-1], v[-2:].mean(), v[-3:].mean(), v[-6:].mean(), v[-12:].mean(), v[-24:].mean(), v[-48:].mean(),
                     v[-1] - v[-13], v[-120:-108].mean(), v[-132:-120].mean()])


def make_forecaster_29(train):
    """Fit a forecaster on `train` (the monthly values of a period, a float array) and return forecast(past): the
    value 6 months after the last value of `past` (all the months known so far, at least those of `train`)."""
    s = np.sqrt(np.asarray(train, dtype=float))                 # the square root tames the peaks of the cycle
    first = 132 + HORIZON_29 - 1                                # the first target with 132 months of past
    X = np.array([features_29(s[:t - HORIZON_29 + 1]) for t in range(first, len(s))])
    y = s[first:]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    trees = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(X, y)

    def forecast(past):
        f = features_29(np.sqrt(np.asarray(past, dtype=float)))[None]
        return float(max((ridge.predict(f)[0] + trees.predict(f)[0]) / 2, 0.0) ** 2)

    return forecast'''

CHECK_29 = r'''with checked("22.29"):
    started_29 = time.perf_counter()
    public_29, preds_29 = score_29(make_forecaster_29, I86, I86, I06)
    print(f"public score (trained before 1986, targets 1986-2005): {public_29:.3f} of the persistence RMSE "
          f"({time.perf_counter() - started_29:.0f} s)")
    if not np.all(np.isfinite(preds_29)):
        print("❌ Ex 22.29 : des prévisions ne sont pas des nombres finis (NaN) : ta fonction lit-elle SPOTS ou "
              "SUNSPOTS au lieu de train et past ?")
    elif public_29 > GOAL_29:
        print(f"❌ Ex 22.29 : {public_29:.3f} sur la période publique : il faut au plus {fr(GOAL_29, 2)} avant de révéler le test caché.")
    else:
        REVEALED_29 = globals().get("REVEALED_29")
        if REVEALED_29 is None:                                 # revealed once per session (an honour rule)
            REVEALED_29 = score_29(make_forecaster_29, I06, I06, len(SERIES_29))[0]
        hidden_29 = REVEALED_29
        print(f"hidden test (trained before 2006, targets {SUNSPOTS['date'].iloc[I06]:%Y-%m} to "
              f"{SUNSPOTS['date'].iloc[-1]:%Y-%m}): {hidden_29:.3f} (revealed once per session: a new run shows the "
              "same frozen result)")
        verdict("22.29", hidden_29 <= GOAL_29,
                f"défi réussi : au plus {fr(GOAL_29, 2)} de la RMSE de la persistance sur les deux périodes."
                + (f" 🌟 Palier atteint : au plus {fr(STAR_29, 2)} sur le test caché." if hidden_29 <= STAR_29 else ""),
                f"pas encore : {hidden_29:.3f} sur le test caché, il faut au plus {fr(GOAL_29, 2)}. Le résultat est figé "
                "pour cette session : repense ta méthode sur une validation prise dans train, sans la régler au plus près de la période publique.")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    dates_29 = SUNSPOTS["date"].iloc[I86:I06]
    ax.plot(dates_29, SERIES_29[I86:I06], lw=0.8, color="0.5", label="valeur réelle")
    ax.plot(dates_29, SERIES_29[I86 - 6:I06 - 6], lw=1, label="persistance")
    ax.plot(dates_29, preds_29, lw=1.5, label="ta prévision")
    ax.set_title("Période publique : prévisions à 6 mois")
    ax.legend()
    plt.show()'''

GIVEN_27 = r'''N_TRAIN_27 = 120_000 if FAST_MODE else 360_000   # training characters of EACH text (the same number for both)
EPOCHS_27 = 2 if FAST_MODE else 10
N_VAL_27 = 20_000                                  # the last 20 000 characters of each text: the validation
TEXTS_27 = {"Holmes (anglais)": HOLMES, "Verne (français)": VERNE}
print(f"{N_TRAIN_27:_} training characters per text, {EPOCHS_27} epochs".replace("_", " "))'''

TODO_27 = r'''def unigram_bits_27(text):
    """The entropy, in bits, of the characters of `text` taken one at a time (their frequencies, ch. 6): what a model
    without context pays per character on average."""
    raise NotImplementedError("unigram_bits_27")


def bits_per_char_27(model, X, y, batch_size=2000):
    """The mean cross-entropy of `model` on the windows X (N, T) and their next characters y (N,), in BITS per
    character, computed by mini-batches in evaluation mode without autograd."""
    raise NotImplementedError("bits_per_char_27")'''

SOLUTION_27 = r'''def unigram_bits_27(text):
    """The entropy, in bits, of the characters of `text` taken one at a time (their frequencies, ch. 6): what a model
    without context pays per character on average."""
    p = np.array(list(Counter(text).values()), dtype=float) / len(text)
    return float(-(p * np.log2(p)).sum())


def bits_per_char_27(model, X, y, batch_size=2000):
    """The mean cross-entropy of `model` on the windows X (N, T) and their next characters y (N,), in BITS per
    character, computed by mini-batches in evaluation mode without autograd."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        total = sum(F.cross_entropy(model(X[i:i + batch_size].to(device)), y[i:i + batch_size].to(device),
                                    reduction="sum").item() for i in range(0, len(X), batch_size))
    return total / len(X) / np.log(2)                     # nats -> bits'''

CHECK_27 = r'''with checked("22.27"):
    same("22.27", "unigram_bits_27 sur « aabb », « abcd » et « aaaa »",
         [unigram_bits_27("aabb"), unigram_bits_27("abcd"), unigram_bits_27("aaaa")], [1.0, 2.0, 0.0], atol=1e-9)
    from scipy.stats import entropy
    same("22.27", "unigram_bits_27(HOLMES) contre scipy.stats.entropy(effectifs, base=2)", unigram_bits_27(HOLMES),
         entropy(list(Counter(HOLMES).values()), base=2))
    torch.manual_seed(2227)
    probe_27 = nn.Sequential(nn.Embedding(5, 5), nn.Flatten(), nn.Linear(15, 5))      # windows of 3 characters
    x_probe_27, y_probe_27 = torch.randint(0, 5, (4100, 3)), torch.randint(0, 5, (4100,))
    with torch.no_grad():
        want_27 = F.cross_entropy(probe_27(x_probe_27), y_probe_27).item() / math.log(2)
    same("22.27", "bits_per_char_27 sur 4 100 fenêtres (trois mini-batches inégaux) contre la cross-entropy en bits",
         bits_per_char_27(probe_27, x_probe_27, y_probe_27), want_27, rtol=1e-5)

    S = mylearn.sequences
    rows_27 = []
    for name_27, text_27 in TEXTS_27.items():
        stoi_27, itos_27 = S.build_vocab(text_27)
        ids_27 = S.encode(text_27, stoi_27)
        X_27, y_27 = S.make_windows(ids_27[:N_TRAIN_27], 40, horizon=1, stride=3)
        Xv_27, yv_27 = S.make_windows(ids_27[-N_VAL_27:], 40, horizon=1, stride=1)
        X_27, y_27, Xv_27, yv_27 = (torch.from_numpy(np.ascontiguousarray(a)) for a in (X_27[:, :, 0], y_27, Xv_27[:, :, 0], yv_27))
        torch.manual_seed(2227)
        model_27 = CharLSTM_25(len(itos_27)).to(DEVICE)
        started_27 = time.perf_counter()
        curve_27 = []
        for epoch_27 in range(EPOCHS_27):                   # one epoch at a time, to follow the validation
            train_25(model_27, X_27, y_27, epochs=1, seed=epoch_27)
            curve_27.append(bits_per_char_27(model_27, Xv_27, yv_27))
        unigram_27 = unigram_bits_27(text_27)
        rows_27.append((name_27, len(itos_27), unigram_27, curve_27))
        print(f"{name_27}: {len(itos_27)} characters · {time.perf_counter() - started_27:.0f} s · "
              + repr(generate_25(model_27, text_27[:40], 120, stoi_27, itos_27, temperature=0.5, rng=np.random.default_rng(27))))
    print(f"\n{'texte':18} {'V':>4} {'unigramme':>10} " + " ".join(f"{'epoch ' + str(k + 1):>8}" for k in range(EPOCHS_27))
          + f" {'gain':>6}  (bits par caractère, sur les 20 000 derniers caractères)")
    for name_27, v_27, unigram_27, curve_27 in rows_27:
        print(f"{name_27:18} {v_27:4d} {unigram_27:10.3f} " + " ".join(f"{b:8.3f}" for b in curve_27)
              + f" {unigram_27 - curve_27[-1]:6.3f}")'''

RECORD_26 = r'''wrong_a_26 = {"les epochs se comptent à partir de 1 : l'indice 0 est l'epoch 1": best_epoch_26 - 1}
if int(np.argmin(TRAIN_LOSS_26)) + 1 not in (best_epoch_26, best_epoch_26 - 1):
    wrong_a_26["c'est la loss de validation qui choisit l'epoch, pas celle d'entraînement"] = int(np.argmin(TRAIN_LOSS_26)) + 1
wb.record("22.26a", best_epoch_26, mistakes=wrong_a_26)
wrong_b_26 = {"la loss est en nats : la perplexité est e puissance la loss, pas 2 puissance la loss": 2 ** min(VAL_LOSS_26)}
if abs(np.exp(VAL_LOSS_26[-1]) - perplexity_26) > 0.01:
    wrong_b_26["prends l'epoch de la plus basse loss de validation, pas la dernière"] = float(np.exp(VAL_LOSS_26[-1]))
wb.record("22.26b", perplexity_26, decimals=2, mistakes=wrong_b_26)
wb.record("22.26c", bits_26, decimals=3, mistakes={"pour passer des nats aux bits, on divise par ln 2": min(VAL_LOSS_26) * np.log(2),
          "les bits se calculent sur la loss, pas sur la perplexité : log2 de la perplexité, ou loss / ln 2": float(np.log(perplexity_26))})
entropy_nats_26 = float(-(freqs_26 * np.log(freqs_26)).sum())
wb.record("22.26d", unigram_26, decimals=2, mistakes={"la perplexité est l'exponentielle de l'entropie, pas l'entropie elle-même": entropy_nats_26,
          "un modèle sans contexte utilise les fréquences des caractères, pas la même probabilité pour tous": 88.0,
          "l'entropie calculée avec ln est en nats : la perplexité est e puissance l'entropie, pas 2 puissance": 2 ** entropy_nats_26})
wb.record("22.26e", "B", mistakes=''' + letters("B", {
    "A": "compare les deux courbes sur les dernières epochs : l'écart entre entraînement et validation reste-t-il le même ?",
    "C": "regarde la loss d'entraînement des dernières epochs : monte-t-elle ?"}) + ")"

PART_D = Part("D", "Générer du texte, comparer deux langues, lire une vidéo, prévoir le cycle solaire",
              "Le générateur de caractères du livre, en PyTorch : tu l'écris et tu l'entraînes sur Holmes (22.25), tu lis "
              "sa courbe de loss sur 20 epochs (22.26), tu compares l'anglais de Holmes et le français de Verne avec "
              "le même réseau (22.27) ; puis le CNN-LSTM du livre, sur des vidéos de balle synthétiques (22.28), et le "
              "défi des taches solaires (22.29). Ces exercices utilisent ta librairie `sequences.py` (parties A et C).",
              exercises=[
    Ex("22.25", "📦", 3, 60, "Générer du Holmes caractère par caractère (deux LSTM, many-to-one)",
       "écrire, entraîner et faire parler le générateur de texte du livre : deux couches LSTM, une couche dense, un caractère à la fois.",
       "Ex 22.12 · Ex 22.13 · Ex 22.20 · Ex 22.22 · ch. 20 (boucle d'entraînement) · fiche §22.9, §22.8.2",
       thread="Holmes", tracks="C",
       body=r"""Le générateur du §22.9 : il lit 40 caractères et prédit le suivant (many-to-one) ; on le fait ensuite écrire en lui redonnant sans cesse ses propres caractères. Les données : les fenêtres de 40 caractères de Holmes, une tous les 3 caractères, chacune avec le caractère qui la suit (ta `make_windows` avec `horizon=1, stride=3`) ; l'entraînement lit les `N_TRAIN_25` premiers caractères, la validation les 10 % de la fin.

**Contrat**
- `CharLSTM_25(vocab_size, hidden_size=128, num_layers=2)` : des ids `(N, T)` int64 passent en one-hot `(N, T, V)` (`F.one_hot(...).float()`), puis dans un `nn.LSTM(V, hidden_size, num_layers, batch_first=True)` ; un `nn.Linear(hidden_size, V)` lit la sortie du **dernier** pas et rend des logits `(N, V)` (pas de softmax : `F.cross_entropy` l'applique) ;
- `train_25(model, X, y, epochs=2, batch_size=128, seed=0)` entraîne en place avec la cross-entropy, `torch.optim.RMSprop(lr=0.01, alpha=0.9)` (le réglage du code Keras du livre) et `clip_grad_norm_` à 1,0, dans un nouvel ordre à chaque epoch (un seul `torch.Generator().manual_seed(seed)`) ; chaque mini-batch va sur l'appareil du modèle ; elle rend la liste des loss moyennes de chaque epoch ;
- `generate_25(model, seed_text, n_chars, stoi, itos, temperature=1.0, rng=None)` encode l'amorce (ta `encode`), puis répète `n_chars` fois : le modèle lit les 40 derniers ids (tous, s'il y en a moins), en mode évaluation et sans autograd, et ta `sample_from_logits` tire l'id suivant (logits en `float64`) ; elle rend les `n_chars` **nouveaux** caractères (ta `decode`), sans l'amorce.

La vérification contrôle l'architecture (nombre de paramètres, formes), prépare les données avec ta librairie, entraîne (une à deux minutes sur un CPU peu chargé en mode rapide, bien plus si d'autres programmes tournent : 240 000 caractères, 2 epochs), mesure la loss de validation (attendu : moins de `TARGET_25`, 2,1 nats en mode rapide, 1,6 en mode complet), puis fait écrire le réseau à trois températures. 🚀 Le mode complet (tout l'entraînement, 20 epochs) prend une demi-heure sur un CPU : un GPU Colab est conseillé ; 📈 22.26 lit son historique.

Dans tes notes : que fait le texte à la température 0 (le choix du livre, le caractère le plus probable) ? Quelle température donne le texte le plus « Holmes » ? Pourquoi une fenêtre tous les 3 caractères, plutôt qu'à chaque caractère ?""",
       given=GIVEN_25, todo=TODO_25, check=CHECK_25, solution=SOLUTION_25 + "\n\n\n" + CHECK_25,
       note="À la température 0, le réseau finit presque toujours dans une boucle (« that that that… », ou « to the street to the street… ») : "
            "le caractère le plus probable après une suite donnée est toujours le même, et une suite qui revient "
            "ramène le même texte. Vers 0,5, le texte a des mots anglais, de la ponctuation et des guillemets à leur "
            "place, avec des phrases sans suite ; à 1, il invente des mots. Une fenêtre tous les 3 caractères divise "
            "le temps d'une epoch par 3 : des fenêtres voisines se recouvrent de 37 caractères sur 40 et "
            "apprendraient presque la même chose."),

    Ex("22.26", "📈", 2, 15, "Lire la courbe de loss et la perplexité du générateur",
       "lire une courbe de loss de générateur, la traduire en perplexité et en bits par caractère, et la comparer à un modèle sans contexte.",
       "Ex 22.25 · ch. 6 (entropie) · ch. 19 (courbes d'apprentissage) · fiche §22.9", thread="Holmes", tracks="C",
       body=r"""Le générateur de 22.25 en mode complet, entraîné 20 epochs sur 90 % de Holmes (environ 25 minutes sur un CPU, calculées à l'avance) : la cellule suivante donne ses loss moyennes d'entraînement et ses loss de validation après chaque epoch, en **nats par caractère** (la cross-entropy de PyTorch utilise le logarithme naturel), et des textes générés après quelques epochs.

La **perplexité** d'un modèle de langue est l'exponentielle de sa cross-entropy moyenne : $e^{\text{loss}}$ quand la loss est en nats. Elle se lit comme un nombre de choix : une perplexité de 5, c'est en moyenne la même incertitude qu'un choix au hasard entre 5 caractères. La même loss, en **bits par caractère**, est la loss divisée par $\ln 2$ (ch. 6).

Calcule, avec NumPy, à partir de `TRAIN_LOSS_26`, `VAL_LOSS_26` et `HOLMES` :
a) `best_epoch_26` : l'epoch (la première est 1) de la plus basse loss de validation ;
b) `perplexity_26` : la perplexité de validation à cette epoch ;
c) `bits_26` : la loss de validation à cette epoch, en bits par caractère ;
d) `unigram_26` : la perplexité d'un modèle **sans contexte**, qui donne à chaque caractère sa fréquence dans `HOLMES` (son entropie, ch. 6, passée à l'exponentielle) ;
e) `overfit_26` : `"A"` aucun signe d'overfitting en 20 epochs ; `"B"` la loss de validation cesse de baisser, ou remonte, alors que celle d'entraînement continue de baisser ; `"C"` les deux loss remontent à la fin.

Dans tes notes : combien de « choix » le contexte de 40 caractères retire-t-il, par rapport au modèle sans contexte, et par rapport à un modèle uniforme sur les 88 caractères ? Quelle epoch garderais-tu, et comment le ferais-tu automatiquement (ch. 19) ? Les textes générés s'améliorent-ils encore quand la loss de validation ne baisse plus ?""",
       given=_given_26(), todo=TODO_26, check=CHECK_26, solution=SOLUTION_26 + "\n" + CHECK_26, record=RECORD_26,
       note="La loss de validation la plus basse est atteinte à l'epoch 9 (1,549 nats, une perplexité de 4,71, soit 2,24 "
            "bits par caractère) ; ensuite elle oscille et remonte, alors que la loss d'entraînement baisse jusqu'au "
            "bout : le réseau commence à apprendre le texte d'entraînement par cœur (B). Un modèle sans contexte a une "
            "perplexité de 22,4 (une entropie de 4,49 bits), un modèle uniforme de 88 : le contexte de 40 caractères "
            "divise le nombre de « choix » par près de cinq par rapport aux seules fréquences. À la température 0, les "
            "textes tournent en boucle à toutes les epochs ; à 0,5, ils gagnent en vocabulaire même après l'epoch 9, "
            "ce qui ne veut pas dire que le modèle généralise mieux."),

    Ex("22.27", "🔬", 3, 40, "Holmes contre Verne : même réseau, deux langues",
       "comparer, en bits par caractère, ce que le même réseau apprend de l'anglais de Holmes et du français de Verne, face à un modèle sans contexte.",
       "Ex 22.25 · Ex 22.26 · ch. 6 (entropie) · fiche §22.9, §22.4.2", thread="Holmes/Verne", tracks="C",
       body=r"""Le même générateur (`CharLSTM_25` et `train_25`, mêmes réglages, même graine) appris sur le même nombre de caractères d'anglais (Holmes) et de français (Verne) : lequel apprend le mieux ? On compare en **bits par caractère** sur les 20 000 derniers caractères de chaque texte, et face au modèle sans contexte de chaque langue.

**Contrat**
- `unigram_bits_27(text)` : l'entropie, en bits, des caractères de `text` pris un par un (leurs fréquences, ch. 6) ;
- `bits_per_char_27(model, X, y, batch_size=2000)` : la cross-entropy moyenne du modèle sur les fenêtres `X` et leurs caractères suivants `y`, **en bits**, par mini-batches, en mode évaluation et sans autograd (attention au dernier mini-batch, plus petit : c'est une moyenne sur toutes les fenêtres, pas sur les mini-batches).

**Avant d'exécuter**, écris dans tes notes ton hypothèse : quel texte aura la plus basse loss en bits par caractère ? Lequel gagnera le plus par rapport à son modèle sans contexte ? Pense à la taille du vocabulaire (accents, majuscules accentuées), à la longueur des mots, aux noms propres qui reviennent.

La vérification essaie tes deux fonctions, puis, pour chaque texte, construit son vocabulaire avec ta librairie, entraîne un réseau `EPOCHS_27` epochs sur `N_TRAIN_27` caractères (environ une minute par texte en mode rapide ; 🚀 le mode complet, 360 000 caractères et 10 epochs, prend une dizaine de minutes par texte sur un CPU), affiche un texte généré à la température 0,5 et un tableau : taille du vocabulaire $V$, entropie sans contexte, bits par caractère après chaque epoch, gain.

Dans tes notes : ton hypothèse tient-elle ? La comparaison est-elle juste (mêmes caractères d'entraînement, mais pas les mêmes livres) ? Que changerais-tu pour qu'elle le soit davantage ? Que deviennent les écarts en mode complet, si tu as pu le lancer ?""",
       given=GIVEN_27, todo=TODO_27, check=CHECK_27, solution=SOLUTION_27 + "\n\n\n" + CHECK_27,
       note="Lors de nos essais (mode rapide), Verne descend plus bas que Holmes (2,85 contre 3,07 bits par caractère "
            "après deux epochs) alors que son modèle sans contexte part plus haut (4,64 contre 4,49 bits : 103 "
            "caractères, avec les accents) : le gain du contexte est plus grand pour Verne (1,79 bit contre 1,42). "
            "Une explication plausible : un seul roman, aux mots et aux noms qui reviennent sans cesse (Phileas Fogg, "
            "Passepartout, Fix), contre douze nouvelles aux personnages et aux lieux changeants ; le français a aussi "
            "des mots plus longs et des terminaisons très prévisibles. La comparaison reste fragile : deux livres, "
            "pas deux langues, un seul entraînement court et une seule graine."),

    Ex("22.28", "📦", 3, 50, "CNN-LSTM : la balle monte-t-elle ou descend-elle ?",
       "assembler un CNN-LSTM : une convolution lit chaque image, un LSTM lit la suite des images.",
       "Ex 22.22 · ch. 21 (convolution, pooling) · 🧮 22.8 · fiche §22.8.1", thread="synth", tracks="C",
       body=r"""Le CNN-LSTM du §22.8.1, en petit. Chaque vidéo a 16 images de 32 × 32 : une balle floue traverse l'image de gauche à droite en montant (classe 0), en descendant (1) ou à plat (2), à une hauteur moyenne tirée au hasard, avec du bruit. Une image seule ne dit donc pas la classe : il faut l'ordre des images. La cellule suivante crée 1 500 vidéos d'entraînement et 600 de test, de forme `(N, 16, 1, 32, 32)`, montre trois vidéos et définit `fit_28` (Adam, 8 epochs).

**Contrat**
- chaque image, **seule** : `nn.Conv2d(1, 8, 5, stride=2, padding=2)`, ReLU, `nn.AdaptiveMaxPool2d(4)`, aplatie en $8 \times 4 \times 4 = 128$ valeurs (le résumé de la carte que recommande la fiche, au lieu de la carte entière) ;
- les 16 vecteurs d'une vidéo, dans l'ordre : `nn.LSTM(128, 32, batch_first=True)` ; la sortie du dernier pas : `nn.Linear(32, 3)` ;
- `BallCNNLSTM_28()` transforme des vidéos `(N, T, 1, 32, 32)` en logits `(N, 3)` ;
- `params_28` : son nombre de paramètres, calculé **avant** d'écrire la classe (ch. 21 et ✏️ 22.3).

La vérification contrôle `params_28`, les formes et l'indépendance des vidéos d'un mini-batch, entraîne (quelques secondes) et mesure l'accuracy de test (attendu : au moins 90 %), puis celle des mêmes vidéos aux images mélangées (le même ordre pour toutes), et passées à l'envers.

Dans tes notes : pourquoi l'accuracy s'effondre-t-elle quand on mélange les images, alors que chaque image est intacte ? Que devient la classe d'une vidéo passée à l'envers, et pourquoi l'accuracy tombe-t-elle même sous un tiers ? Combien de poids aurait le LSTM s'il recevait la carte entière, $8 \times 16 \times 16$ valeurs par image ?""",
       given=GIVEN_28, todo=TODO_28, check=CHECK_28, solution=SOLUTION_28 + "\n\n\n" + CHECK_28, record=RECORD_28,
       note="Le réseau atteint environ 98 % : la convolution trouve la balle dans chaque image, le LSTM suit sa hauteur d'une "
            "image à l'autre. Mélanger les images garde chaque image intacte mais détruit le mouvement : le réseau ne "
            "voit plus qu'une suite de positions sans ordre et tombe autour de 50 % (il reconnaît encore une partie des "
            "vols à plat). À l'envers, une balle qui monte devient une balle qui descend, et elle va de droite à gauche, "
            "ce que le réseau n'a jamais vu : il confond les deux pentes et même le vol à plat, d'où un quart environ. "
            "Avec la carte entière, le LSTM recevrait 2 048 valeurs par image : $4 \\times (2\\,048 \\times 32 + "
            "32^2 + 64) = 266\\,496$ poids, treize fois plus, pour une information que le max pooling garde."),

    Ex("22.29", "🏆", 4, 120, "Défi taches solaires : battre la persistance de 20 % à six mois",
       "construire la meilleure prévision à six mois des taches solaires, jugée sur une période qu'on ne voit qu'une fois.",
       "Ex 22.23 · Ex 22.21 · ch. 8 et 15 (validation) · fiche « Au-delà du livre »", thread="taches solaires", tracks="C",
       body=r"""Le défi : prévoir la moyenne mensuelle des taches solaires **6 mois** à l'avance mieux que la persistance, avec la méthode de ton choix (un LSTM, un GRU, un modèle de scikit-learn, un mélange…).

**Contrat**
- `make_forecaster_29(train)` reçoit les valeurs mensuelles d'une période (un tableau de `float`) et ajuste ta méthode **sur elles seules** ; elle rend une fonction `forecast(past)` ;
- `forecast(past)` reçoit tous les mois connus jusqu'ici (au moins ceux de `train`, et ceux qui ont suivi) et rend la valeur prévue **6 mois après** la dernière valeur de `past`, un nombre. Elle ne réajuste rien : elle lit `past`.

La vérification calcule le **score public** : `make_forecaster_29(SPOTS[:I86])`, puis une prévision pour chaque mois de 1986 à 2005, comparée à la persistance (le rapport des RMSE ; la persistance vaut 1). Si ce score vaut au plus 0,80, elle révèle le **test caché** : la même chose avec un entraînement jusqu'à fin 2005 et les cibles de 2006 à la fin de la série. Le défi est réussi avec un rapport d'au plus `GOAL_29` = 0,80 sur les deux périodes ; palier 🌟 : au plus `STAR_29` = 0,78 sur le test caché.

Règles d'honneur : ta fonction n'utilise que `train` et `past` (pendant l'évaluation, `SPOTS` et `SUNSPOTS` cachent les mois à venir : les lire donne des NaN) ; le test caché n'est révélé qu'**une fois par session**, et le résultat est figé ; règle tes choix sur la période publique, ou mieux, sur une validation prise dans `train` (ch. 15), car un réglage poussé sur 1986-2005 risque de ne pas tenir sur 2006-2026 (le cycle 24 a été le plus faible depuis un siècle).

Dans tes notes : quelle méthode as-tu retenue, pourquoi, et quels essais ont échoué ? Ton score public et ton score caché sont-ils proches ? Sinon, qu'est-ce que cela dit de la façon dont tu as réglé ta méthode ?""",
       given=GIVEN_29, todo=TODO_29, check=CHECK_29, solution=SOLUTION_29 + "\n\n\n" + CHECK_29,
       note="Le corrigé n'utilise pas de réseau : des moyennes des derniers mois à plusieurs échelles, la tendance sur un "
            "an et le niveau d'il y a 9 à 11 ans (la durée du cycle), sur la racine carrée de la série (qui calme les "
            "pics), passés à une régression ridge et à un gradient boosting dont on moyenne les prévisions. Il obtient "
            "environ 0,74 sur la période publique et 0,79 sur le test caché. Le LSTM de 22.23, utilisé comme prévision, fait 0,77 sur la période "
            "publique mais près de 0,9 sur le test caché : réglé sur 1986-2005, il ne tient pas sur un cycle "
            "bien plus faible que ceux qu'il a vus. C'est la leçon du défi : le score qu'on annonce est celui d'une "
            "période qu'on n'a pas utilisée pour choisir."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 22.Q1 à 22.Q12, 22.R1 à 22.R3, 22.1 à 22.5, 22.8 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 | "
            "★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if kind == "exercise":
        title = "# 22 · Réseaux récurrents (RNN, LSTM, GRU) — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est "
               "pas fait affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la "
               "section « Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch22_rnn/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 22`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 22 · Réseaux récurrents (RNN, LSTM, GRU) — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Découper une série ou un texte en fenêtres, compter les exemples, et valider une prévision sans "
               "fuite de données.\n"
               "- Programmer en NumPy, dans `mylearn.rnn`, les cellules RNN, LSTM et GRU, et les comparer à "
               "`nn.RNN`, `nn.LSTM` et `nn.GRU`.\n"
               "- Mesurer la disparition du gradient, puis entraîner un LSTM qui prévoit les taches solaires et un "
               "autre qui écrit du Holmes caractère par caractère.\n\n"
               "**Rappel express.** Cellule RNN : $h_t = \\tanh(x_t W_{ih} + b_{ih} + h_{t-1} W_{hh} + b_{hh})$. "
               "LSTM (portes $i, f, g, o$) : $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$, "
               "$h_t = o_t \\odot \\tanh(c_t)$. GRU (portes $r, z, n$) : $h_t = (1 - z_t) \\odot n_t + z_t \\odot "
               "h_{t-1}$. Paramètres d'une couche PyTorch : $k\\,(DH + H^2 + 2H)$, $k = 1, 4, 3$. Fenêtres : "
               "$N = \\lfloor (T - w - h)/s \\rfloor + 1$. Température : $p = \\mathrm{softmax}(z/T)$.")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu écrire les équations d'une cellule RNN, LSTM et GRU, et compter leurs paramètres ?\n"
               "2. Sais-tu expliquer pourquoi le gradient d'un RNN s'évanouit ou explose, et ce que change le "
               "chemin additif du LSTM ?\n"
               "3. Sais-tu préparer une série temporelle sans fuite, et régler la température d'un générateur ?\n\n"
               "**Pour aller plus loin** : le billet de C. Olah sur les LSTM et celui d'A. Karpathy sur les RNN, "
               "cités dans la fiche. La suite : le ch. 23, Keras, traduit en PyTorch.")]


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
