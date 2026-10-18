#!/usr/bin/env python
"""Build the two notebooks of chapter 22 from a single source (used by Claude).

    python tools/chapters/build_ch22.py
    python tools/run_all_notebooks.py chapitres/ch22_rnn/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch22_rnn/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
22.1 to 22.5 and of the 🧮 estimate 22.8 (02_exercices.md). The proofs ∂ 22.6 and 22.7, the reflection
(🗣️ 22.9, ⚖️ 22.10, 📄 22.11) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 22.12 to 22.29) come at the second generation session: the libraries
mylearn.sequences and mylearn.rnn, the vanishing gradient measured with autograd, the sunspot forecast against
persistence, the Holmes generator and the CNN-LSTM.
"""

from __future__ import annotations

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
# Parts A to D (exercises 22.12 to 22.29): next generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "22.12 à 22.14", "ta librairie sequences.py : vocabulaire, encodage et fenêtres ; un MLP sur des fenêtres "
                           "mélangées"),
    ("B", "22.15 à 22.19", "ta librairie rnn.py : les cellules RNN, LSTM et GRU, comparées à PyTorch ; la disparition "
                           "du gradient mesurée avec autograd"),
    ("C", "22.20 à 22.24", "l'échantillonnage avec température, une série temporelle piégée, les formes des RNN "
                           "profonds et bidirectionnels, les taches solaires face à la persistance"),
    ("D", "22.25 à 22.29", "le générateur de Holmes, sa loss et sa perplexité, Holmes contre Verne, le CNN-LSTM, le "
                           "défi des taches solaires"),
]


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
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
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
    later = ("" if PARTS else
             "\n\n*Les exercices de code 22.12 à 22.29 (parties A à D, tes librairies `sequences.py` et `rnn.py` et "
             "les RNN PyTorch) seront ajoutés à la prochaine session de génération ; relance alors "
             "`python tools/start_chapter.py 22` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu écrire les équations d'une cellule RNN, LSTM et GRU, et compter leurs paramètres ?\n"
               "2. Sais-tu expliquer pourquoi le gradient d'un RNN s'évanouit ou explose, et ce que change le "
               "chemin additif du LSTM ?\n"
               "3. Sais-tu préparer une série temporelle sans fuite, et régler la température d'un générateur ?\n\n"
               "**Pour aller plus loin** : le billet de C. Olah sur les LSTM et celui d'A. Karpathy sur les RNN, "
               "cités dans la fiche. La suite : le ch. 23, Keras, traduit en PyTorch." + later)]


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
