#!/usr/bin/env python
"""Build the two notebooks of chapter 21 from a single source (used by Claude).

    python tools/chapters/build_ch21.py
    python tools/run_all_notebooks.py chapitres/ch21_cnn/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch21_cnn/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
21.1 to 21.6 and of the 🧮 estimate 21.8 (02_exercices.md). The proof ∂ 21.7, the reflection (🗣️ 21.9,
⚖️ 21.10, 📄 21.11) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 21.12 to 21.32) come at the second generation session: the library mylearn.conv,
hand-made filters, PyTorch CNNs on Fashion-MNIST and CIFAR-10, the filters of a pretrained network, an
adversarial attack, the transposed convolution and the backward pass of a convolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402,F401

CHAPTER = "21"
FOLDER = "chapitres/ch21_cnn"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np


def out_size(size, k, s=1, p=0, d=1):
    """Output length of a convolution or pooling window (the formula of the sheet)."""
    return (size + 2 * p - d * (k - 1) - 1) // s + 1


def correlate(x, w):
    """Cross-correlation without padding, stride 1 (what nn.Conv2d computes), with plain loops."""
    kh, kw = w.shape
    out = np.zeros((x.shape[0] - kh + 1, x.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + kh, j:j + kw] * w)
    return out


def receptive_field(layers):
    """Side of the receptive field of one output, for a stack of (kernel, stride) layers."""
    r, jump = 1, 1
    for k, s in layers:
        r += (k - 1) * jump
        jump *= s
    return r


# Ex 21.1: the yellow detector R + G - B
YELLOW = np.array([1.0, 1.0, -1.0])

# Ex 21.2: a 5 x 5 image and a Sobel kernel
X_212 = np.array([[3, 1, 0, 0, 1], [2, 0, 1, 2, 0], [4, 1, 0, 1, 3], [1, 2, 2, 0, 1], [0, 3, 1, 2, 2]], float)
K_212 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)
Y_212 = correlate(X_212, K_212)
Y_PAD_212 = correlate(np.pad(X_212, 1), K_212)
CONV_212 = correlate(X_212, K_212[::-1, ::-1])                        # the kernel flipped in both directions
MIN_212 = tuple(int(v) for v in np.unravel_index(np.argmin(Y_212), Y_212.shape))

# Ex 21.4: the MNIST convnet of the book
P_CONV1_214 = 32 * (1 * 9 + 1)
P_CONV2_214 = 64 * (32 * 9 + 1)
P_FC1_214 = 12 * 12 * 64 * 128 + 128
P_FC2_214 = 128 * 10 + 10
P_TOTAL_214 = P_CONV1_214 + P_CONV2_214 + P_FC1_214 + P_FC2_214

# Ex 21.6: transposed convolution, stride 2, as stamps x[i, j] * w with their corner at (2i, 2j)
X_216 = np.array([[2.0, -1.0], [1.0, 3.0]])
W_216 = np.array([[1.0, 0.0, 2.0], [-1.0, 3.0, 0.0], [2.0, 1.0, -1.0]])
Y_216 = np.zeros((5, 5))
for _i in range(2):
    for _j in range(2):
        Y_216[2 * _i:2 * _i + 3, 2 * _j:2 * _j + 3] += X_216[_i, _j] * W_216
Y_PAD_216 = Y_216[1:-1, 1:-1]                                           # padding=1 removes one row/column per border

# Ex 21.8: the compute and the activations of VGG16
VGG_BLOCKS = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]
MACS_208, ACTS_208 = [], 0
_c_in, _side = 3, 224
for _block in VGG_BLOCKS:
    for _c_out in _block:
        MACS_208.append(_side * _side * _c_out * _c_in * 9)
        ACTS_208 += _c_out * _side * _side
        _c_in = _c_out
    _side //= 2
FC_208 = 25088 * 4096 + 4096 * 4096 + 4096 * 1000'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
SHAPE = 'a shape as a string, e.g. "(3, 5)"'
WHOLE = "a whole number"

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("21.Q1", "Les deux sens de « profondeur »", [
        ("a", WHOLE, "3",
         r'''mistakes={"la profondeur d'une image, ce sont ses canaux, pas sa taille": 224,
          "une photo en couleur a plus d'un canal": 1}'''),
        ("b", WHOLE, "64",
         r'''mistakes={"la profondeur d'un tenseur, ce sont ses canaux, pas sa hauteur ni sa largeur": 112}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"c'est la profondeur d'un réseau : de quoi est-il fait ?": "A",
          "le nombre de bits par pixel est un autre sens, rare ici": "C",
          "les filtres de VGG16 font 3 × 3": "D"}'''),
        ("d", WHOLE, "3 * 32 * 32",
         r'''mistakes={"trois valeurs PAR PIXEL, pas trois features par image : relis l'encadré « Le livre, corrigé » du §21.2": 3,
          "n'oublie pas les trois canaux": 32 * 32,
          "tu as additionné les dimensions : chaque pixel porte ses propres valeurs": 32 + 32 + 3}'''),
        ("e", TF, "False",
         r'''mistakes={"le convnet MNIST du livre a aussi un pooling, un dropout et des couches denses : est-ce un CNN ?": True}'''),
    ]),
    Paper("21.Q2", "Un seul neurone promené sur toute l'image", [
        ("a", TF, "True",
         r'''mistakes={"relis le §21.2.3 : le neurone emporte-t-il ses poids avec lui ?": False}'''),
        ("b", WHOLE, "25",
         r'''mistakes={"tu as additionné les deux côtés du filtre : il a un poids par case": 10, "le biais n'est pas demandé": 26}'''),
        ("c", TF, "False",
         r'''mistakes={"les poids d'un filtre dépendent-ils de la taille de l'image qu'il parcourt ?": True}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"le noyau, ce sont les poids du filtre": "A",
          "le champ récepteur est tout le carré de pixels lu par le filtre": "C",
          "la carte de caractéristiques est la sortie de la couche": "D"}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"un filtre 3 × 3 a plus de poids qu'un 2 × 2 : ce n'est pas la raison": "A",
          "la parité du côté ne change pas la vitesse d'apprentissage": "C",
          "les bibliothèques acceptent des noyaux pairs (padding \"same\" compris)": "D"}'''),
    ]),
    Paper("21.Q3", "Convolution ou corrélation croisée ?", [
        ("a", TF, "False",
         r'''mistakes={"relis l'encadré 🕰️ du §21.3 : que dit la documentation de PyTorch ?": True}'''),
        ("b", "a whole number", "int(np.dot([1, 2, 3], [1, 0, -1]))",
         r'''mistakes={"c'est la convolution (noyau retourné) : la corrélation croisée ne retourne pas le noyau": 2,
          "il manque un des trois produits de la première fenêtre": -3}'''),
        ("c", "a whole number", "int(np.dot([1, 2, 3], [-1, 0, 1]))",
         r'''mistakes={"c'est la corrélation croisée : la convolution retourne d'abord le noyau": -2}'''),
        ("d", TF, "True",
         r'''mistakes={"retourner (1, 2, 1) donne… (1, 2, 1)": False}'''),
        ("e", WHOLE, "(6 - 3 + 1) ** 2",
         r'''mistakes={"c'est le nombre de positions sur une seule ligne : le filtre descend aussi": 4,
          "une image 6 × 6 n'a pas 36 morceaux 3 × 3 entiers : le filtre doit tenir dans l'image": 36,
          "ça, c'est pour une image 5 × 5": 9}'''),
        ("f", WHOLE, "3 * 3 * 3",
         r'''mistakes={"tu as oublié les canaux : un morceau prend sa fenêtre dans chacun": 9,
          "tu as additionné les canaux au lieu de compter les valeurs de chacun": 9 + 3}'''),
    ]),
    Paper("21.Q4", "Détecteurs de motifs et hiérarchie de filtres", [
        ("a", WHOLE, "3",
         r'''mistakes={"les pixels noirs (0) ne retirent rien : seuls les blancs sous un −1 font baisser le score": 9,
          "le meilleur morceau met du blanc sous tous les +1, pas sous un seul": 1}'''),
        ("b", WHOLE, "3 - 1",
         r'''mistakes={"le pixel du centre est blanc aussi : sous quel poids est-il ?": 3,
          "recompte les pixels blancs de la colonne de gauche": 1}'''),
        ("c", TF, "False",
         r'''mistakes={"relis la fin du §21.3.1 : qui règle les poids des filtres ?": True}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"un filtre 3 × 3 a 9 poids par canal, en haut comme en bas du réseau": "A",
          "le learning rate ne change pas ce que voit un filtre": "C",
          "seule la première couche reçoit les canaux de l'image": "D"}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"c'est le sens du ch. 1 ; ici, le livre emploie le mot autrement": "A",
          "un hyperparamètre se règle avant l'entraînement, ce n'est pas ce que cherche un filtre": "C",
          "un filtre cherche une structure faite de plusieurs pixels": "D"}'''),
    ]),
    Paper("21.Q5", "Padding et stride : que devient la taille ?", [
        ("a", WHOLE, "out_size(12, 5)",
         r'''mistakes={"tu as retiré tout le côté du filtre : à sa dernière position, il tient encore dans l'image": 7,
          "sans padding, la sortie est plus petite que l'entrée": 12}'''),
        ("b", WHOLE, "(7 - 1) // 2",
         r'''mistakes={"c'est la bordure totale : on demande le padding de chaque côté": 6,
          "un filtre 7 × 7 déborde de plus d'un pixel autour de son centre": 1}'''),
        ("c", WHOLE, "out_size(7, 3, s=2)",
         r'''mistakes={"n'oublie pas le + 1 de la formule, et prends le stride vertical pour la hauteur": 2}'''),
        ("d", WHOLE, "out_size(12, 3, s=3)",
         r'''mistakes={"pour la largeur, prends le stride horizontal": 5, "tu as oublié le + 1 de la formule": 3}'''),
        ("e", TF, "True",
         r'''mistakes={"avec un stride égal au côté du filtre, les fenêtres se recouvrent-elles ?": False}'''),
    ]),
    Paper("21.Q6", "Canaux d'entrée, nombre de filtres, canaux de sortie", [
        ("a", WHOLE, "6",
         r'''mistakes={"la profondeur d'un filtre est celle de son ENTRÉE": 4, "un filtre 3 × 3 a une profondeur, en plus": 3}'''),
        ("b", WHOLE, "4", r'''mistakes={"un canal de sortie par filtre": 6}'''),
        ("c", WHOLE, "4 * 6 * 3 * 3",
         r'''mistakes={"tu as oublié les canaux d'entrée : un filtre a la profondeur de son entrée": 4 * 9,
          "tu as compté un seul filtre": 6 * 9}'''),
        ("d", WHOLE, "4 * (6 * 3 * 3 + 1)",
         r'''mistakes={"un biais par filtre, pas par poids": 4 * 6 * 9 * 2, "tu as oublié les biais": 216}'''),
        ("e", WHOLE, "6 * 3 * 3",
         r'''mistakes={"en convolution par canal, chaque filtre ne lit qu'un canal de l'entrée": 6 * 6 * 9,
          "tu as compté un seul filtre : il y en a un par canal": 9}'''),
    ]),
    Paper("21.Q7", "Convolution 1D ou convolution 1×1 ?", [
        ("a", WHOLE, "3 * 50",
         r'''mistakes={"le filtre couvre toute la largeur de chaque mot, pas un nombre par mot": 3,
          "tu as additionné : chaque mot apporte tous ses nombres au filtre": 53}'''),
        ("b", WHOLE, "20 - 3 + 1",
         r'''mistakes={"sans padding, le filtre doit tenir entier dans la phrase": 20, "tu as oublié le + 1 de la formule": 17}'''),
        ("c", WHOLE, "300 * 175",
         r'''mistakes={"tu as additionné les deux nombres de canaux : chaque filtre a un poids par canal d'entrée": 300 + 175,
          "le biais n'est pas demandé": 300 * 175 + 175}'''),
        ("d", TF, "True", r'''mistakes={"un filtre 1 × 1 lit-il les pixels voisins ?": False}'''),
        ("e", WHOLE, "28",
         r'''mistakes={"une convolution 1 × 1 garde la taille de l'image": 175, "la hauteur, pas le nombre de canaux": 300}'''),
    ]),
    Paper("21.Q8", "Couche de convolution et initialisation des filtres", [
        ("a", TF, "True",
         r'''mistakes={"si tout est symétrique au départ, les gradients des filtres sont-ils différents ?": False}'''),
        ("b", WHOLE, "64 * 3 * 3",
         r'''mistakes={"tu as oublié les canaux : le fan-in compte toutes les valeurs reçues": 9,
          "chaque canal apporte toute une fenêtre de valeurs, pas une seule": 64}'''),
        ("c", "a number, 3 decimals", "math.sqrt(2 / 576)",
         r'''decimals=3, mistakes={"l'écart-type, pas la variance": 2 / 576,
          "avec un fan-in de 9 seulement, tu as oublié les 64 canaux": math.sqrt(2 / 9),
          "c'est Glorot avec fan-in = fan-out ; He prend √(2/fan-in)": math.sqrt(1 / 576),
          "c'est l'exemple de la fiche, à 32 canaux ; ici, l'entrée en a 64": math.sqrt(2 / 288)}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"le noyau, ce sont les poids": "A", "le champ récepteur est ce que lit une sortie": "C",
          "le padding est une bordure ajoutée à l'entrée": "D"}'''),
        ("e", TF, "False",
         r'''mistakes={"par défaut, padding=0 : que devient une image de 32 avec un noyau 3 ?": True}'''),
    ]),
    Paper("21.Q9", "Convolution transposée : les mots piégés", [
        ("a", TF, "False",
         r'''mistakes={"relis l'encadré « Le livre, corrigé » du §21.8 : que espace la dilatation ?": True}'''),
        ("b", WHOLE, "(4 - 1) * 2 + 3",
         r'''mistakes={"tu as compté H × s : relis la formule de la convolution transposée (§21.8)": 8,
          "la convolution transposée AGRANDIT son entrée : relis sa formule (§21.8)": 1}'''),
        ("c", WHOLE, "(4 - 1) * 2 - 2 + 3",
         r'''mistakes={"le padding retire p cases de CHAQUE côté": 8, "relis la question : la couche a maintenant padding=1": 9}'''),
        ("d", WHOLE, "3 - 1", r'''mistakes={"tu as pris le stride lui-même : un pixel occupe déjà une des cases de l'écart": 3}'''),
        ("e", TF, "True",
         r'''mistakes={"relis la fin du §21.8 et sa figure : les tampons se recouvrent-ils de façon égale ?": False}'''),
        ("f", WHOLE, "3 * (3 - 1) + 1",
         r'''mistakes={"la dilatation espace les cases du noyau : elle ne multiplie pas son côté": 9,
          "tu as oublié le + 1 : compte les pixels couverts, pas les intervalles entre eux": 6}'''),
    ]),
    Paper("21.Q10", "VGG16 et ce que voient ses filtres", [
        ("a", SHAPE, '"(256, 28, 28)"',
         r'''mistakes={"recompte les poolings jusqu'à la sortie du troisième bloc, le sien compris": "(256, 56, 56)",
          "le format (C, H, W) : les canaux d'abord": "(28, 28, 256)",
          "le troisième bloc a 256 filtres": "(128, 28, 28)"}'''),
        ("b", WHOLE, "5", r'''mistakes={"un pooling par bloc, et VGG16 a cinq blocs": 4}'''),
        ("c", WHOLE, "224 // 2 ** 4",
         r'''mistakes={"le pooling du cinquième bloc vient après ses convolutions, pas avant block5_conv1": 7,
          "as-tu compté le pooling du quatrième bloc ?": 28}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"on ne touche pas aux poids : le réseau n'apprend rien ici": "A",
          "le dropout n'agit pas sur l'image": "C",
          "la méthode part d'une image de bruit, pas des images d'entraînement": "D"}'''),
        ("e", TF, "True",
         r'''mistakes={"relis le §21.9.2 : que cherchent les filtres de block1, puis ceux de block4 ?": False}'''),
    ]),
    Paper("21.Q11", "Exemples adversariaux : vrai ou faux ?", [
        ("a", TF, "False",
         r'''mistakes={"relis le §21.10 : le tigre attaqué ressemble-t-il au tigre d'origine ?": True}'''),
        ("b", TF, "True",
         r'''mistakes={"relis la définition d'une perturbation universelle au §21.10": False}'''),
        ("c", TF, "False",
         r'''mistakes={"VGG16, très bien entraîné, est-il trompé par le tigre attaqué ?": True}'''),
        ("d", "a number, 2 decimals", "0.03 * 28 * 28",
         r'''decimals=2, mistakes={"c'est la variation d'un seul pixel : on demande la somme sur toute l'image": 0.03,
          "la somme des valeurs absolues des variations, pas leur racine": 0.03 * 28}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"une attaque non ciblée cherche n'importe quelle erreur": "A",
          "« universelle » parle d'une perturbation valable pour beaucoup d'images": "C",
          "« physique » parle d'une attaque dans le monde réel (un autocollant)": "D"}'''),
    ]),
    Paper("21.Q12", "🕰️ Après VGG16 : ResNet, ViT et modèles pré-entraînés", [
        ("a", LETTER, '"B"',
         r'''mistakes={"ResNet utilise surtout des filtres 3 × 3": "A", "ResNet s'entraîne par rétropropagation": "C",
          "c'est la tête de VGG16 que ResNet a abandonnée": "D"}'''),
        ("b", WHOLE, "(224 // 16) ** 2",
         r'''mistakes={"c'est le nombre de patchs le long d'un côté : l'image en a dans les deux directions": 14,
          "16 est le côté d'un patch, pas le nombre de patchs par côté": 256}'''),
        ("c", WHOLE, "512",
         r'''mistakes={"tu as gardé les positions : le pooling moyen global en fait la moyenne": 7 * 7,
          "il ne garde pas les 7 × 7 positions de chaque canal": 512 * 49}'''),
        ("d", TF, "True",
         r'''mistakes={"relis l'encadré 🕰️ du §21.9.1 : que télécharge weights=\"DEFAULT\" ?": False}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("21.R1", "Ce que changent model.train() et model.eval() pour le dropout", [
        ("a", TF, "True", r'''mistakes={"en évaluation, le dropout inversé ne coupe plus rien et ne multiplie rien": False}'''),
        ("b", "a number, 4 decimals", "1 / (1 - 0.25)",
         r'''decimals=4, mistakes={"c'est la part gardée : le dropout inversé la compense, il ne l'applique pas une seconde fois": 0.75,
          "le facteur compense la part GARDÉE, pas la part coupée": 4.0}'''),
        ("c", "a whole number (%)", "25",
         r'''fractional="en pourcentage : une proportion de 0,25 vaut 25 %", mistakes={"c'est la part GARDÉE": 75}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"le dropout actif coupe des neurones : peut-il améliorer la mesure ?": "B",
          "le dropout dépend du mode train ou eval": "C", "PyTorch ne vérifie pas le mode pour toi": "D"}'''),
    ]),
    Paper("21.R2", "Le gradient d'un poids utilisé à plusieurs endroits", [
        ("a", WHOLE, "1 + 2 + 3",
         r'''mistakes={"tu as additionné les ∂L/∂y_i seuls : chaque contribution passe aussi par ∂y_i/∂w": 3}'''),
        ("b", WHOLE, "1 * 1 + (-1) * 2 + 2 * 3",
         r'''mistakes={"relis le signe de ∂L/∂y₂": 9,
          "tu as additionné les ∂L/∂y_i sans les pondérer par ∂y_i/∂w": 2}'''),
        ("c", TF, "True", r'''mistakes={"relis la fiche du ch. 18 : `+=` ou `=` dans backward ?": False}'''),
    ]),
    Paper("21.R3", "Le produit scalaire, ce que calcule un filtre en chaque position", [
        ("a", WHOLE, "int(np.dot([1, 2, -1], [3, 0, 2]))",
         r'''mistakes={"n'oublie pas le signe de −1 × 2": 5, "la somme des produits terme à terme": 6}'''),
        ("b", WHOLE, "1 * 0 + 2 * 1 + 3 * 1 + 4 * 0",
         r'''mistakes={"terme à terme, pas un produit matriciel": 10, "les cases hors diagonale comptent aussi : relis la seconde grille": 0}'''),
        ("c", TF, "True",
         r'''mistakes={"w · x = ‖w‖ ‖x‖ cos θ : quand est-il le plus grand ?": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("21.1", "Le détecteur de jaune à la main", [
        ("a", "a number, 2 decimals", "float(YELLOW @ [0.9, 0.85, 0.1])",
         r'''decimals=2, mistakes={"le bleu compte avec le poids −1, pas +1": 1.85}'''),
        ("b", "a number, 2 decimals", "float(YELLOW @ [1.0, 0.5, 0.0])", r'''decimals=2, mistakes={"relis le signe du poids du vert": 0.5}'''),
        ("c", "a number, 2 decimals", "float(YELLOW @ [1.0, 1.0, 1.0])",
         r'''decimals=2, mistakes={"le blanc contient aussi du bleu : as-tu compté sa composante B ?": 2,
          "le bleu a le poids −1 : il fait baisser le score, pas monter": 3}'''),
        ("d", "a number, 2 decimals", "0.0", r'''decimals=2, mistakes={"que vaut chacune des trois composantes d'un pixel noir ?": 1}'''),
        ("e", LETTER, '"A"',
         r'''mistakes={"compare les scores de a) à d) : lequel est le plus grand ?": "B",
          "le blanc a du bleu, que le neurone retire": "C", "le noir a un score nul": "D"}'''),
        ("f", "a number, 2 decimals", "2.0",
         r'''decimals=2, mistakes={"le bleu compte en négatif : il ne peut pas augmenter le score": 3,
          "le bleu retire au score : quelle valeur de B le rend le plus grand ?": 1}'''),
        ("g", "a whole number", "-1",
         r'''mistakes={"le magenta n'a pas de vert : le vert doit faire baisser le score": 1, "le vert doit compter, en négatif": 0}'''),
        ("h", WHOLE, "4 * 6",
         r'''mistakes={"un score par pixel, pas par valeur (le neurone combine les trois canaux)": 4 * 6 * 3,
          "tu as additionné les deux côtés de l'image : il y a un score par pixel": 10}'''),
        ("i", TF, "True",
         r'''mistakes={"un neurone qui lit les 3 canaux d'un seul pixel, avec les mêmes poids partout : relis le §21.6": False}'''),
    ]),
    Paper("21.2", "Une corrélation 5×5 par 3×3 à la main", [
        ("a", WHOLE, "Y_212.shape[1]",
         r'''mistakes={"tu as retiré tout le côté du filtre : à sa dernière position, il tient encore dans l'image": 2,
          "sans padding, la sortie rétrécit": 5}'''),
        ("b", "a whole number", "int(Y_212[0, 0])",
         r'''mistakes={"c'est la valeur avec le noyau retourné (convolution) : nn.Conv2d ne retourne pas le noyau": int(CONV_212[0, 0]),
          "le noyau a des 2 sur sa ligne du milieu": int(np.sum(X_212[:3, :3] * np.array([[1, 0, -1]] * 3)))}'''),
        ("c", "a whole number", "int(Y_212[1, 1])",
         r'''mistakes={"le morceau dont le coin en haut à gauche est en (1, 1) : lignes 1 à 3, colonnes 1 à 3": int(Y_212[0, 0])}'''),
        ("d", "a whole number", "int(Y_212[2, 1])",
         r'''mistakes={"ligne 2, colonne 1 : vérifie que tu n'as pas pris (1, 2)": int(Y_212[1, 2]),
          "c'est la valeur avec le noyau retourné": int(CONV_212[2, 1])}'''),
        ("e", "a whole number", "int(Y_212.min())",
         r'''mistakes={"la plus PETITE valeur, pas la plus grande": int(Y_212.max())}'''),
        ("f", 'a position as a string, e.g. "(0, 4)"', f'"{(1, 2)}"',
         r'''mistakes={"ligne puis colonne : tu les as inversées": "(2, 1)",
          "c'est la position de la plus grande valeur": "(0, 0)"}'''),
        ("g", "a whole number", "int(Y_PAD_212[0, 0])",
         r'''mistakes={"avec le padding, la position (0, 0) est centrée sur le pixel X[0, 0], entouré de zéros": int(Y_212[0, 0])}'''),
        ("h", "a whole number", "int(Y_PAD_212[4, 0])",
         r'''mistakes={"la position (4, 0) : la dernière ligne, la première colonne": int(Y_PAD_212[0, 4])}'''),
        ("i", "a whole number", "int(CONV_212[0, 0])",
         r'''mistakes={"la convolution retourne le noyau : ce n'est pas la valeur de b)": int(Y_212[0, 0])}'''),
        ("j", TF, "True",
         r'''mistakes={"retourne K dans les deux directions : que vaut le noyau retourné, comparé à K ?": False}'''),
    ]),
    Paper("21.3", "Tailles de sortie : padding, stride et la hiérarchie 600 → 300 → 100", [
        ("a", WHOLE, "out_size(600, 5, s=2, p=2)",
         r'''mistakes={"tu as oublié le + 1 de la formule": 299}'''),
        ("b", WHOLE, "out_size(300, 3, s=3, p=2)",
         r'''mistakes={"c'est ce qu'annonce le livre : applique la formule": 100,
          "la partie entière ⌊·⌋ arrondit vers le bas, jamais vers le haut": 102}'''),
        ("c", WHOLE, "0",
         r'''mistakes={"as-tu essayé un padding plus petit ?": 1,
          "relis la question : on demande le plus PETIT padding qui donne exactement 100": 2}'''),
        ("d", WHOLE, "1",
         r'''mistakes={"calcule la largeur avec ce padding : vaut-elle encore 100 ?": 2,
          "relis la question : on demande le plus GRAND padding qui donne encore 100": 0}'''),
        ("e", WHOLE, "8",
         r'''mistakes={"c'est ce qu'écrit le livre ; la profondeur d'un filtre est celle de son entrée": 4,
          "le second étage ne lit pas l'image : il lit la sortie du premier": 1}'''),
        ("f", WHOLE, "out_size(28, 3, d=2)",
         r'''mistakes={"tu as oublié la dilatation : le noyau dilaté couvre plus de pixels que son côté": 26,
          "la dilatation espace les cases du noyau : son étendue n'est pas d × k": 23}'''),
        ("g", WHOLE, "2",
         r'''mistakes={"le noyau dilaté déborde davantage qu'un noyau 3 × 3 ordinaire : relis f)": 1,
          "c'est la bordure totale : on demande le padding de chaque côté": 4}'''),
        ("h", WHOLE, "(100 - 1) * 3 + 3",
         r'''mistakes={"tu as compté H × s : relis le premier terme de la formule": 303,
          "c'est la formule d'une convolution ordinaire : une convolution transposée agrandit": 33}'''),
        ("i", WHOLE, "(7 - 1) * 2 - 2 * 1 + 4",
         r'''mistakes={"le padding se retire des DEUX côtés": 15, "tu as oublié le padding : la convolution transposée le retire": 16}'''),
        ("j", WHOLE, "out_size(14, 4, s=2, p=1)",
         r'''mistakes={"tu as oublié le + 1 de la formule": 6, "il manque le padding et le + 1": 5}'''),
    ]),
    Paper("21.4", "Compter les paramètres : convolution contre couche dense", [
        ("a", WHOLE, "P_CONV1_214",
         r'''mistakes={"tu as oublié les biais : un par filtre": 32 * 9, "tu as compté un seul filtre": 9 + 1}'''),
        ("b", WHOLE, "P_CONV2_214",
         r'''mistakes={"tu as oublié les canaux d'entrée : un filtre a la profondeur de son entrée": 64 * 10,
          "tu as oublié les biais": 64 * 32 * 9}'''),
        ("c", WHOLE, "P_FC1_214",
         r'''mistakes={"tu as oublié les biais": 9216 * 128,
          "l'entrée de la couche dense est tout le tenseur aplati, pas ses seuls canaux": 64 * 128 + 128}'''),
        ("d", WHOLE, "P_FC2_214", r'''mistakes={"tu as oublié les biais": 1280}'''),
        ("e", WHOLE, "P_TOTAL_214",
         r'''mistakes={"il manque une couche dans ta somme": P_TOTAL_214 - P_FC2_214,
          "tu as oublié les biais d'une ou de plusieurs couches": P_TOTAL_214 - 32 - 64 - 128 - 10}'''),
        ("f", "a number (%), 1 decimal", "100 * P_FC1_214 / P_TOTAL_214",
         r'''decimals=1, mistakes={"la part de la première couche dense, pas celle des convolutions": 100 * (P_CONV1_214 + P_CONV2_214) / P_TOTAL_214}'''),
        ("g", WHOLE, "784 * 32 * 26 * 26 + 32 * 26 * 26",
         r'''mistakes={"tu as oublié les biais : un par sortie": 784 * 32 * 26 * 26,
          "une couche dense relie CHAQUE entrée à chaque sortie": 32 * 26 * 26}'''),
        ("h", WHOLE, "26 * 26 * 32 * 9",
         r'''mistakes={"tu as compté un seul filtre": 26 * 26 * 9,
          "on compte les positions de SORTIE, pas les pixels de l'entrée": 28 * 28 * 32 * 9}'''),
    ]),
    Paper("21.5", "Champ récepteur effectif d'une pile convolution + pooling", [
        ("a", WHOLE, "receptive_field([(3, 1), (3, 1)])",
         r'''mistakes={"chaque couche ajoute moins que son côté : relis le facteur (k − 1) du rappel": 6, "tu as compté une seule couche": 3}'''),
        ("b", WHOLE, "receptive_field([(3, 1)] * 3)",
         r'''mistakes={"les champs ne se multiplient pas d'une couche à l'autre : chaque couche ajoute son débord au précédent": 9}'''),
        ("c", WHOLE, "3 * 9 * 64 * 64",
         r'''mistakes={"tu as compté une seule couche": 9 * 64 * 64, "le biais n'est pas demandé": 3 * (9 * 64 + 1) * 64}'''),
        ("d", WHOLE, "7 * 7 * 64 * 64",
         r'''mistakes={"relis la question : le noyau unique doit avoir le champ récepteur de TROIS convolutions": 5 * 5 * 64 * 64}'''),
        ("e", WHOLE, "receptive_field([(3, 1), (3, 3), (3, 1)])",
         r'''mistakes={"c'est le compte du livre : il oublie le débord des filtres du niveau 1": 9,
          "après le pooling, deux cases voisines ne sont plus des pixels voisins : relis le produit des strides": 7}'''),
        ("f", WHOLE, "receptive_field([(3, 1), (3, 3), (3, 1), (3, 3), (3, 1)])",
         r'''mistakes={"le second pooling multiplie encore l'espacement des cases par son stride": 23,
          "les champs ne se multiplient pas d'un niveau à l'autre : applique la formule du rappel, couche par couche": 27}'''),
        ("g", WHOLE, "receptive_field([(5, 2), (3, 3)])",
         r'''mistakes={"c'est l'approximation du livre (6 × 6) : applique la formule du rappel": 6,
          "le débord du second noyau est multiplié par le stride qui le précède": 7}'''),
        ("h", WHOLE, "receptive_field([(3, 1), (3, 1), (2, 2)])",
         r'''mistakes={"le pooling compte comme une couche : il ajoute lui aussi son débord": 5,
          "pour le pooling aussi, le débord est k − 1, pas k": 7}'''),
    ]),
    Paper("21.6", "Convolution transposée à la main : insérer des zéros puis corréler", [
        ("a", WHOLE, "Y_216.shape[1]",
         r'''mistakes={"tu as compté H × s : le dernier tampon déborde de tout le noyau": 4,
          "les tampons se chevauchent : la sortie est plus petite que H × k": 6}'''),
        ("b", WHOLE, "2 - 1", r'''mistakes={"tu as pris le stride lui-même : un pixel occupe déjà une des cases de l'écart": 2}'''),
        ("c", WHOLE, "3 - 1 - 0",
         r'''mistakes={"relis la valeur de p : cette convolution transposée n'a pas de padding": 1,
          "tu as pris le côté du noyau : relis la formule de l'énoncé": 3}'''),
        ("d", "a whole number", "int(Y_216[0, 0])",
         r'''mistakes={"le tampon déposé est x[0, 0] × w, pas w seul": 1}'''),
        ("e", "a whole number", "int(Y_216[1, 1])",
         r'''mistakes={"combien de tampons couvrent vraiment la case (1, 1) ?": int(X_216.sum() * W_216[1, 1])}'''),
        ("f", "a whole number", "int(Y_216[0, 2])",
         r'''mistakes={"la case (0, 2) reçoit plusieurs tampons : as-tu compté celui de x[0, 1] ?": 4,
          "la case (0, 2) reçoit plusieurs tampons : as-tu compté celui de x[0, 0] ?": -1}'''),
        ("g", "a whole number", "int(Y_216[2, 2])",
         r'''mistakes={"la case (2, 2) reçoit les QUATRE tampons, celui de x[1, 1] compris": int(X_216[0, 0] * W_216[2, 2] + X_216[0, 1] * W_216[2, 0] + X_216[1, 0] * W_216[0, 2])}'''),
        ("h", "a whole number", "int(Y_216.sum())",
         r'''mistakes={"tu as oublié les poids du noyau : chaque tampon est x[i, j] × w": int(X_216.sum()),
          "tu as oublié les valeurs de x : chaque tampon est pondéré par son pixel": int(W_216.sum())}'''),
        ("i", WHOLE, "Y_PAD_216.shape[1]", r'''mistakes={"− 2p : on retire une colonne de CHAQUE côté": 4}'''),
        ("j", "a whole number", "int(Y_PAD_216[1, 1])",
         r'''mistakes={"le centre de la sortie 3 × 3 est la case (2, 2) de la sortie sans padding": int(Y_216[1, 1])}'''),
    ]),
    Paper("21.8", "Fermi : paramètres, mémoire et calcul de VGG16", [
        ("a", "a number of millions, 1 decimal", "MACS_208[0] / 1e6",
         r'''decimals=1, mistakes={"tu as oublié les canaux de l'image d'entrée": 224 * 224 * 64 * 9 / 1e6,
          "en millions": MACS_208[0] / 1e9}'''),
        ("b", "a number of billions, 2 decimals", "MACS_208[1] / 1e9",
         r'''decimals=2, mistakes={"tu as oublié les canaux d'entrée de cette couche": 224 * 224 * 64 * 9 / 1e9,
          "en milliards": MACS_208[1] / 1e6}'''),
        ("c", "a number of billions, 1 decimal", "sum(MACS_208) / 1e9",
         r'''decimals=1, mistakes={"chaque pooling divise par 2 le côté de l'image des blocs suivants": sum(c_in * c_out * 224 * 224 * 9 for c_in, c_out in [(3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 256), (256, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512)]) / 1e9,
          "chaque filtre a C_in canaux de 3 × 3 poids (et le côté est divisé par 2 à chaque pooling)": sum(c * 224 * 224 * 9 for c in [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]) / 1e9}'''),
        ("d", "a number (%), 1 decimal", "100 * FC_208 / (FC_208 + sum(MACS_208))",
         r'''decimals=1, mistakes={"la part des couches denses dans le CALCUL, pas dans les paramètres": 89.4}'''),
        ("e", "a number of MB, 1 decimal", "4 * ACTS_208 / 1e6",
         r'''decimals=1, mistakes={"tu as compté des valeurs, pas des octets": ACTS_208 / 1e6, "en Mo de 10⁶ octets": 4 * ACTS_208 / 2 ** 20}'''),
        ("f", "a number of GB, 2 decimals", "64 * 4 * ACTS_208 / 1e9",
         r'''decimals=2, mistakes={"c'est pour une seule image : relis la taille du mini-batch": 4 * ACTS_208 / 1e9,
          "Go de 10⁹ octets, pas Gio": 64 * 4 * ACTS_208 / 2 ** 30}'''),
        ("g", TF, "True",
         r'''mistakes={"compare ta réponse de d) à la part des paramètres des couches denses (✏️ 20.6)": False}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 21.1 à 21.6 et 🧮 21.8)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; une "
               "forme ou une position : entre guillemets (`\"(3, 5)\"`). Arrondis comme l'énoncé le demande, et "
               "seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. Les questions « dans ta "
               "copie », la preuve ∂ 21.7, la réflexion (🗣️ 21.9, ⚖️ 21.10, 📄 21.11) et l'entretien se corrigent "
               "avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Shared tools of parts A to D
# ---------------------------------------------------------------------------
TOOLS = r'''# Tools for the notebook exercises (parts A to D)
import copy
import math
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from wb.nbtools import (StopCheck, as_numpy, checked, filled, pair, refuses, returned, run_pytest_group, same,
                        shape_text, verdict, written)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # a computer without PyTorch (Mac Intel): this chapter is done on Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : ce chapitre se fait sur Colab (00_setup/INSTALL_LOCAL.md). Les "
          "cellules qui utilisent PyTorch affichent ⏳ au lieu de s'exécuter ; les tests de ta librairie conv.py "
          "tournent sans lui : python -m pytest tests/test_ch21_conv.py -q")

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
TEST_FILE = "tests/test_ch21_conv.py"     # the tests of your library conv.py


def n_params(model):
    """The number of parameters (learned numbers) of a torch module."""
    return sum(p.numel() for p in model.parameters())


def _book_convnet_params():
    """Parameters of the MNIST convnet of the book (sheet §21.9), counted by PyTorch on a model built here."""
    features = nn.Sequential(nn.Conv2d(1, 32, 3), nn.Conv2d(32, 64, 3), nn.MaxPool2d(2), nn.Flatten())
    with torch.no_grad():
        width = features(torch.zeros(1, 1, 28, 28)).shape[1]
    return n_params(features) + n_params(nn.Linear(width, 128)) + n_params(nn.Linear(128, 10))


def accuracy(model, X, y, batch_size=1000):
    """Accuracy of a classifier on tensors X (N, C, H, W) and y (N,), in evaluation mode and without autograd."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            correct += (model(X[i:i + batch_size].to(DEVICE)).argmax(1).cpu() == y[i:i + batch_size]).sum().item()
    return correct / len(X)


print("tools ready · torch", torch.__version__ if torch is not None else "absent", "· device", DEVICE)'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="21")   # reload your saved file\n'


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre `mon_travail/mylearn/conv.py` (créé par "
            "`python tools/start_chapter.py 21`), lis la docstring, remplace chaque `raise NotImplementedError(...)` de "
            "l'exercice par ton code et **enregistre**. NumPy et la bibliothèque standard seulement : PyTorch est "
            "l'**oracle** des tests (sans PyTorch, ils comparent à une transcription NumPy des formules, et aucun n'est "
            "ignoré). La première cellule de vérification recharge ta librairie et compare quelques cas à "
            "`torch.nn.functional` ; la seconde lance les tests de l'exercice. "
            "`python -m pytest tests/test_ch21_conv.py -q` les lance tous dans un terminal.")


COMMON_CONTRACT = ("**Contrat commun** (les docstrings, plus ces précisions que vérifient les tests) : tes fonctions "
                   "acceptent des listes (d'entiers aussi) comme des tableaux, et ne modifient **jamais** leurs entrées ; "
                   "les tableaux qu'elles rendent sont de **nouveaux** tableaux `float64`, jamais une entrée ni une vue "
                   "d'une entrée (même pour un padding de 0 ou un noyau 1 × 1) ; un argument « int ou tuple de 2 ints » "
                   "accepte les deux formes, le premier nombre valant pour les lignes (H) et le second pour les colonnes "
                   "(W) ; une taille de noyau, un stride ou une dilatation < 1, un padding négatif et une fenêtre qui ne "
                   "tient pas lèvent `ValueError`.")


def ref_check(check: str) -> str:
    """The solutions version of a check cell: the tests run on the reference."""
    return check.replace('run_pytest_group(TEST_FILE, "', 'run_pytest_group(TEST_FILE, impl="ref", keyword="')


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "{" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items() if letter != answer) + "}"


def tests_cell(ex_id: str, probe: str, keyword: str, what: str = "tes fonctions") -> str:
    """The second check cell of a 🔨: the tests of the exercise (no PyTorch needed)."""
    return (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_FILE, "{keyword}")\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent {what}.")')


# ---------------------------------------------------------------------------
# Part A: the library conv.py (21.12 to 21.18)
# ---------------------------------------------------------------------------
CHECK_12 = r'''with checked("21.12"):
    C = mylearn.conv
    cases_12 = [(7, 3, 1, 0, 1), (7, 3, 2, 0, 1), (10, 4, 3, 1, 1), (9, 3, 1, 0, 2), (6, 2, 2, 1, 1), (5, 1, 1, 0, 1)]
    want_12 = [5, 3, 3, 5, 4, 5]                          # the output sizes of F.conv2d for these cases
    got_12 = [C.conv_output_size(n, k, stride=s, padding=p, dilation=d) for n, k, s, p, d in cases_12]
    verdict("21.12", got_12 == want_12 and all(type(v) is int for v in got_12),
            "conv_output_size : les six tailles de PyTorch, en int Python.",
            f"conv_output_size rend {got_12} pour (taille, k, s, p, d) = {cases_12} : attendu {want_12}, des int Python.")
    refuses("21.12", "conv_output_size(2, 5) (la fenêtre ne tient pas)", lambda: C.conv_output_size(2, 5))
    refuses("21.12", "conv_output_size(0, 1, padding=1) (une taille nulle)", lambda: C.conv_output_size(0, 1, padding=1))
    refuses("21.12", "conv_output_size(8, 3, stride=0)", lambda: C.conv_output_size(8, 3, stride=0))
    x_12 = np.arange(12.0).reshape(1, 2, 2, 3)
    arg_12 = x_12.copy()                                  # pad2d receives a copy; the expected values come from x_12
    out_12 = C.pad2d(arg_12, (1, 2))
    if returned("21.12", "pad2d", out_12):
        if not np.array_equal(arg_12, x_12):
            verdict("21.12", False, "pad2d laisse son entrée intacte.",
                    "pad2d a modifié le tableau x reçu : ne le modifie pas, écris dans un nouveau tableau.")
        same("21.12", "pad2d(x, (1, 2)) contre np.pad (1 ligne en haut et en bas, 2 colonnes à gauche et à droite)",
             out_12, np.pad(x_12, ((0, 0), (0, 0), (1, 1), (2, 2))))
        same("21.12", "pad2d(x, 1, value=-inf)", C.pad2d(x_12.copy(), 1, value=-np.inf),
             np.pad(x_12, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf))
        copy_12 = C.pad2d(x_12, 0)
        verdict("21.12", isinstance(copy_12, np.ndarray) and copy_12 is not x_12 and not np.shares_memory(copy_12, x_12)
                and copy_12.dtype == np.float64,
                "pad2d(x, 0) rend un nouveau tableau float64.",
                "pad2d(x, 0) doit rendre un NOUVEAU tableau float64, ni x ni une vue de x.")
    refuses("21.12", "pad2d sur un tableau 3-D", lambda: C.pad2d(np.ones((2, 3, 3)), 1))'''

GIVEN_13 = r'''X_13, _ = wb.datasets.load_cifar10("test")          # 10 000 photos 32 x 32, uint8 (N, H, W, C)
PHOTOS_13 = X_13[[1078, 9590, 1303, 4285]].astype(np.float64) / 255     # the photos A, B, C, D, values in [0, 1]
fig, axes = plt.subplots(1, 4, figsize=(8, 2.4))
for ax, photo, name in zip(axes, PHOTOS_13, "ABCD"):
    ax.imshow(photo)
    ax.set_title(name)
    ax.axis("off")
plt.show()'''

TODO_13 = r'''weight_13 = ...     # the yellow detector R + G - B as the weight of a 1 x 1 convolution: an array of shape (1, 3, 1, 1)
highest_13 = ...    # a) the photo with the highest MEAN score: "A", "B", "C" or "D"
negative_13 = ...   # b) is the mean score of photo B negative? True or False
peak_13 = ...       # c) the photo that holds the highest score of the four (one pixel): "A", "B", "C" or "D"'''

SOLUTION_13 = r'''weight_13 = np.array([1.0, 1.0, -1.0]).reshape(1, 3, 1, 1)    # 1 output channel, 3 input channels, 1 x 1
highest_13, negative_13, peak_13 = "C", True, "D"'''

CHECK_13 = r'''if not filled(weight_13):
    print("⏳ Ex 21.13 : écris weight_13.")
else:
    w_13 = as_numpy(weight_13)
    verdict("21.13", w_13 is not None and w_13.shape == (1, 3, 1, 1) and np.array_equal(w_13.ravel(), [1, 1, -1]),
            "weight_13 : le neurone R + G - B, au format (C_out, C_in, 1, 1) d'une convolution 1 x 1.",
            "weight_13 : attendu les poids (+1, +1, -1) de R, G et B, rangés en (C_out, C_in, k_h, k_w) = (1, 3, 1, 1) ; reçu "
            + ("rien de lisible" if w_13 is None else f"la forme {w_13.shape} et les valeurs {w_13.ravel().tolist()}") + ".")'''

EXPERIMENT_13 = r'''x_13 = torch.tensor(PHOTOS_13).permute(0, 3, 1, 2)                  # (4, 32, 32, 3) -> (4, 3, 32, 32), NCHW
yellow_13 = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64).reshape(1, 3, 1, 1)
scores_13 = F.conv2d(x_13, yellow_13)[:, 0]                       # (4, 32, 32): one score per pixel
fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
for k, name in enumerate("ABCD"):
    axes[0, k].imshow(PHOTOS_13[k])
    axes[0, k].set_title(name)
    shown = axes[1, k].imshow(scores_13[k], cmap="magma", vmin=-1, vmax=2)
    axes[1, k].set_title(f"moyenne {scores_13[k].mean():.2f} · max {scores_13[k].max():.2f}")
for ax in axes.ravel():
    ax.axis("off")
fig.colorbar(shown, ax=axes[1].tolist(), fraction=0.02)
plt.show()
wb.check("21.13a", highest_13)
wb.check("21.13b", negative_13)
wb.check("21.13c", peak_13)'''

RECORD_13 = r'''wb.record("21.13a", "C", mistakes={"la voiture paraît la plus jaune, mais calcule R + G - B pour l'herbe : beaucoup de vert, peu de bleu": "D",
          "le blanc a aussi du bleu, que le neurone retire : son score est-il le plus haut ?": "A", "un ciel bleu : beaucoup de B, retranché": "B"})
wb.record("21.13b", True, mistakes={"un ciel bleu : B grand, R et G petits ; quel signe pour R + G - B ?": False})
wb.record("21.13c", "D", mistakes={"quelle couleur donne le plus grand score R + G - B (✏️ 21.1) ? Quelle photo en a des pixels ?": "C",
          "le bleu du blanc est retranché : un pixel blanc peut-il faire le pic ?": "A", "le ciel bleu reste bas partout": "B"})'''

CHECK_14 = r'''with checked("21.14"):
    C = mylearn.conv
    x_14 = np.arange(2 * 3 * 5 * 6, dtype=float).reshape(2, 3, 5, 6)
    for kw_14 in [dict(kernel_size=3), dict(kernel_size=(2, 3), stride=2, padding=1), dict(kernel_size=2, dilation=(2, 1))]:
        want_14 = F.unfold(torch.tensor(x_14), **kw_14)
        cols_14 = C.im2col(x_14, **kw_14)
        if not returned("21.14", "im2col", cols_14):
            break
        same("21.14", f"im2col(x, {', '.join(f'{k}={v}' for k, v in kw_14.items())}) contre F.unfold", cols_14, want_14)
    w_14 = np.random.default_rng(14).normal(size=(4, 3, 3, 3))
    conv_14 = np.asarray(w_14.reshape(4, -1) @ C.im2col(x_14, 3)).reshape(2, 4, 3, 4)
    same("21.14", "w.reshape(C_out, -1) @ im2col(x, 3), remis en (N, C_out, H_out, W_out), contre F.conv2d", conv_14,
         F.conv2d(torch.tensor(x_14), torch.tensor(w_14)))
    refuses("21.14", "im2col avec un noyau plus grand que l'image", lambda: C.im2col(np.ones((1, 1, 3, 3)), 4))'''

CHECK_15 = r'''import warnings

with checked("21.15"):
    C = mylearn.conv
    rng_15 = np.random.default_rng(15)
    x_15 = rng_15.normal(size=(2, 4, 7, 6))
    cases_15 = [("noyau 3 × 2 asymétrique, biais", (5, 4, 3, 2), dict(), True),
                ("stride (2, 1) et padding (1, 2)", (3, 4, 3, 3), dict(stride=(2, 1), padding=(1, 2)), False),
                ("dilatation 2", (3, 4, 3, 3), dict(dilation=2), True),
                ("groups=2", (6, 2, 3, 3), dict(groups=2, padding=1), True),
                ("convolution par canal (groups=4)", (4, 1, 3, 3), dict(groups=4), False),
                ('padding="same", noyau pair 2 × 4', (3, 4, 2, 4), dict(padding="same"), True)]
    for label_15, shape_15, kw_15, bias_15 in cases_15:
        w_15 = rng_15.normal(size=shape_15)
        b_15 = rng_15.normal(size=shape_15[0]) if bias_15 else None
        args_15 = (x_15.copy(), w_15.copy(), None if b_15 is None else b_15.copy())   # copies: the expected values use x_15
        out_15 = C.conv2d(*args_15, **kw_15)
        if not returned("21.15", "conv2d", out_15):
            break
        if not (np.array_equal(args_15[0], x_15) and np.array_equal(args_15[1], w_15)
                and (b_15 is None or np.array_equal(args_15[2], b_15))):
            verdict("21.15", False, "conv2d laisse ses entrées intactes.",
                    "conv2d a modifié x, weight ou bias : ne modifie pas ses entrées, écris dans un nouveau tableau.")
            break
        with warnings.catch_warnings():               # PyTorch warns that an even kernel with "same" copies the input
            warnings.simplefilter("ignore", UserWarning)
            want_15 = F.conv2d(torch.tensor(x_15), torch.tensor(w_15), None if b_15 is None else torch.tensor(b_15), **kw_15)
        same("21.15", f"conv2d, {label_15}, contre F.conv2d", out_15, want_15)
    refuses("21.15", "un poids dont shape[1] n'est pas C_in / groups", lambda: C.conv2d(x_15, np.ones((2, 3, 3, 3))))
    refuses("21.15", "un biais de forme (1,) pour 2 filtres", lambda: C.conv2d(x_15, np.ones((2, 4, 3, 3)), np.ones(1)))
    refuses("21.15", 'padding="same" avec un stride de 2', lambda: C.conv2d(x_15, np.ones((2, 4, 3, 3)), padding="same", stride=2))
    refuses("21.15", "groups=3 avec 4 canaux d'entrée", lambda: C.conv2d(x_15, np.ones((3, 1, 3, 3)), groups=3))'''

GIVEN_16 = r'''def conv_ok(x, weight, bias=None, stride=1, padding=0):
    """A correct conv2d (cross-correlation, groups=1), computed by PyTorch: the version your tests must accept."""
    out = F.conv2d(torch.as_tensor(np.asarray(x, dtype=float)), torch.as_tensor(np.asarray(weight, dtype=float)),
                   None if bias is None else torch.as_tensor(np.asarray(bias, dtype=float)), stride=stride, padding=padding)
    return out.numpy()


def conv_bug_1(x, weight, bias=None, stride=1, padding=0):
    """Bug: the kernel is flipped (a mathematician's convolution)."""
    w = torch.flip(torch.as_tensor(np.asarray(weight, dtype=float)), dims=(2, 3))
    out = F.conv2d(torch.as_tensor(np.asarray(x, dtype=float)), w,
                   None if bias is None else torch.as_tensor(np.asarray(bias, dtype=float)), stride=stride, padding=padding)
    return out.numpy()


def conv_bug_2(x, weight, bias=None, stride=1, padding=0):
    """Bug: the bias is ignored."""
    out = F.conv2d(torch.as_tensor(np.asarray(x, dtype=float)), torch.as_tensor(np.asarray(weight, dtype=float)),
                   stride=stride, padding=padding)
    return out.numpy()


def conv_bug_3(x, weight, bias=None, stride=1, padding=0):
    """Bug: the stride is applied to the rows only."""
    out = F.conv2d(torch.as_tensor(np.asarray(x, dtype=float)), torch.as_tensor(np.asarray(weight, dtype=float)),
                   None if bias is None else torch.as_tensor(np.asarray(bias, dtype=float)), stride=(stride, 1),
                   padding=padding)
    return out.numpy()


def conv_bug_4(x, weight, bias=None, stride=1, padding=0):
    """Bug: the padding is added at the top and on the left only."""
    xt = F.pad(torch.as_tensor(np.asarray(x, dtype=float)), (padding, 0, padding, 0))
    out = F.conv2d(xt, torch.as_tensor(np.asarray(weight, dtype=float)),
                   None if bias is None else torch.as_tensor(np.asarray(bias, dtype=float)), stride=stride)
    return out.numpy()


def conv_bug_5(x, weight, bias=None, stride=1, padding=0):
    """Bug: the last row is lost when the stride does not divide the input exactly."""
    out = conv_ok(x, weight, bias, stride, padding)
    h = np.asarray(x).shape[2] + 2 * padding - np.asarray(weight).shape[2]
    return out[:, :, :-1] if h % stride else out


BUGS_16 = [conv_bug_1, conv_bug_2, conv_bug_3, conv_bug_4, conv_bug_5]
PREAMBLE_16 = "import torch\nimport torch.nn.functional as F"     # the imports the test file needs
print("conv_ok and 5 buggy versions ready")'''

TODO_16 = r'''import pytest


# Write your test functions here (their names start with test_). They test a function
# conv(x, weight, bias=None, stride=1, padding=0) whose contract is the one of conv_ok (read it).
# At least ONE of your tests must be parametrized with @pytest.mark.parametrize, for example:
#
# @pytest.mark.parametrize("stride, padding", [(1, 0), (2, 1)])
# def test_something(stride, padding):
#     x = np.random.default_rng(0).normal(size=(1, 1, 5, 5))
#     ...
#     assert ...

my_tests_16 = ...   # the list of your test functions'''

SOLUTION_16 = r'''import pytest


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("size", [(7, 7), (8, 5)])
def test_conv_matches_torch(stride, padding, size):
    rng = np.random.default_rng(stride * 100 + padding * 10 + size[1])
    x = rng.normal(size=(2, 3, *size))
    w = rng.normal(size=(4, 3, 3, 2))                      # an asymmetric, rectangular kernel
    b = rng.normal(size=4)
    want = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=stride, padding=padding).numpy()
    got = np.asarray(conv(x, w, b, stride=stride, padding=padding))
    assert got.shape == want.shape
    assert np.allclose(got, want)


def test_conv_without_bias_is_the_dot_product_of_one_window():
    x = np.arange(9.0).reshape(1, 1, 3, 3)
    w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    assert np.allclose(conv(x, w), [[[[0 - 4, 1 - 5], [3 - 7, 4 - 8]]]])


my_tests_16 = [test_conv_matches_torch, test_conv_without_bias_is_the_dot_product_of_one_window]'''

CHECK_16 = r'''def parametrized_16(tests):
    """True when at least one test function carries a pytest.mark.parametrize mark."""
    return any(mark.name == "parametrize" for t in tests for mark in getattr(t, "pytestmark", []))


if my_tests_16 is ...:
    print("⏳ Ex 21.16 : tests pas encore écrits.")
elif not isinstance(my_tests_16, (list, tuple)) or not my_tests_16 or not all(callable(t) for t in my_tests_16):
    print("❌ Ex 21.16 : my_tests_16 doit être une liste non vide de fonctions de test (leurs noms, sans parenthèses).")
else:
    good_16 = wb.run_pytest(my_tests_16, subject=conv_ok, name="conv", preamble=PREAMBLE_16)
    verdict("21.16", parametrized_16(my_tests_16), "au moins un de tes tests est paramétré.",
            "aucun de tes tests n'est paramétré : utilise @pytest.mark.parametrize pour essayer plusieurs strides et paddings.")
    verdict("21.16", good_16.ok, f"tes tests passent sur conv_ok, la version juste ({good_16.passed} cas).",
            "tes tests doivent tous passer sur conv_ok, la version juste (le détail est au-dessus).")
    if good_16.ok:
        for bug_16 in BUGS_16:
            result_16 = wb.run_pytest(my_tests_16, subject=bug_16, name="conv", preamble=PREAMBLE_16, quiet=True)
            verdict("21.16", result_16.failed + result_16.errors > 0,
                    f"{bug_16.__name__} est attrapé ({result_16.failed + result_16.errors} cas en échec).",
                    f"tes tests passent sur {bug_16.__name__}, qui est faux : lis sa docstring, puis ajoute un cas qui le fait échouer.")'''

GIVEN_17 = r'''IMG_17 = (np.random.default_rng(2169).random((24, 24)) < 0.4).astype(float)    # a noisy binary image: 1 white, 0 black
K_17 = -np.ones((3, 3))
K_17[:, 1] = 1.0                       # the filter: a short vertical white stripe (+1) surrounded by black (-1)
print(K_17)'''

TODO_17 = r'''scores_17 = ...      # the scores of K_17 at every position of IMG_17: a (22, 22) array (your conv2d, no padding)
perfect_17 = ...     # a) the number of positions with a perfect score
worst_17 = ...       # b) the number of positions with the lowest possible score of this filter
twos_17 = ...        # c) the number of positions with a score of 2


def show_17(img, kernel, scores):
    """Reproduce figure 21.14 in five panels: the filter, the image, the scores, the perfect matches, the image
    with a box around each match (plt.Rectangle). End with plt.show()."""
    raise NotImplementedError("show_17")'''

SOLUTION_17 = r'''scores_17 = mylearn.conv.conv2d(IMG_17[None, None], K_17[None, None])[0, 0]
best_17 = K_17[K_17 > 0].sum()                    # 3: every +1 on white, every -1 on black
perfect_17 = int(np.sum(scores_17 == best_17))
worst_17 = int(np.sum(scores_17 == -(K_17 < 0).sum()))   # -6: every -1 on white, every +1 on black
twos_17 = int(np.sum(scores_17 == 2))


def show_17(img, kernel, scores):
    """Reproduce figure 21.14 in five panels: the filter, the image, the scores, the perfect matches, the image
    with a box around each match (plt.Rectangle). End with plt.show()."""
    best = kernel[kernel > 0].sum()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    axes[0].imshow(kernel, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0].set_title("(a) filtre")
    axes[1].imshow(img, cmap="gray")
    axes[1].set_title("(b) image")
    shown = axes[2].imshow(scores, cmap="viridis")
    fig.colorbar(shown, ax=axes[2], fraction=0.046)
    axes[2].set_title("(c) scores")
    axes[3].imshow(scores == best, cmap="gray")
    axes[3].set_title("(d) score parfait")
    axes[4].imshow(img, cmap="gray")
    for i, j in np.argwhere(scores == best):          # the window of output (i, j) starts at pixel (i, j)
        axes[4].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 3, 3, fill=False, edgecolor="red", linewidth=2))
    axes[4].set_title("(e) correspondances")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()'''

CHECK_17 = r'''with checked("21.17"):
    if filled(scores_17):
        arr_17 = as_numpy(scores_17)
        verdict("21.17", arr_17 is not None and arr_17.shape == (22, 22), "scores_17 a la forme (22, 22).",
                f"scores_17 : attendu un tableau (22, 22), une corrélation sans padding (24 - 3 + 1) ; reçu "
                f"{None if arr_17 is None else arr_17.shape}.")
    wb.check("21.17a", perfect_17)
    wb.check("21.17b", worst_17)
    wb.check("21.17c", twos_17)
    if filled(scores_17):
        show_17(IMG_17, K_17, np.asarray(scores_17))'''

RECORD_17 = r'''wb.record("21.17a", perfect_17, mistakes={"un score parfait vaut exactement la somme des +1 du filtre : compte les positions où il est atteint, pas celles qui en approchent": int(np.sum(scores_17 >= 2))})
wb.record("21.17b", worst_17, mistakes={"le score le plus bas possible : chaque -1 sur du blanc ET chaque +1 sur du noir ; quelle valeur cela donne-t-il ?": int(np.sum(scores_17 == scores_17.min() + 1))})
wb.record("21.17c", twos_17, mistakes={"compte les positions de score exactement 2, pas au moins 2": int(np.sum(scores_17 >= 2))})'''

CHECK_18 = r'''with checked("21.18"):
    C = mylearn.conv
    x_18 = np.random.default_rng(18).normal(size=(2, 3, 7, 8))
    for kw_18 in [dict(kernel_size=2), dict(kernel_size=3, stride=2), dict(kernel_size=(3, 2), stride=(1, 2), padding=(1, 1)),
                  dict(kernel_size=3, stride=1, padding=1)]:
        text_18 = ", ".join(f"{k}={v}" for k, v in kw_18.items())
        out_max_18 = C.max_pool2d(x_18, **kw_18)
        if not returned("21.18", "max_pool2d", out_max_18):
            break
        same("21.18", f"max_pool2d({text_18}) contre F.max_pool2d", out_max_18, F.max_pool2d(torch.tensor(x_18), **kw_18))
        out_avg_18 = C.avg_pool2d(x_18, **kw_18)
        if not returned("21.18", "avg_pool2d", out_avg_18):
            break
        same("21.18", f"avg_pool2d({text_18}) contre F.avg_pool2d", out_avg_18, F.avg_pool2d(torch.tensor(x_18), **kw_18))
    neg_18 = -np.ones((1, 1, 2, 2))
    same("21.18", "max_pool2d de valeurs toutes négatives, padding 1 (le padding ne gagne jamais)",
         C.max_pool2d(neg_18, 2, stride=1, padding=1), -np.ones((1, 1, 3, 3)))
    refuses("21.18", "max_pool2d avec un padding de 2 pour une fenêtre de 3", lambda: C.max_pool2d(x_18, 3, padding=2))'''

# ---------------------------------------------------------------------------
# Part B: hand-made filters, 1D and 1 x 1 convolutions, the hierarchical face detector (21.19 to 21.21)
# ---------------------------------------------------------------------------
GIVEN_19 = r'''X_19, _ = wb.datasets.load_cifar10("test")
PHOTO_19 = torch.tensor(X_19[[8]].astype(np.float64) / 255).permute(0, 3, 1, 2)   # one photo (1, 3, 32, 32), in [0, 1]
CONST_19 = torch.full((1, 3, 6, 6), 0.4, dtype=torch.float64)                       # a uniform grey image
EDGE_19 = torch.zeros(1, 3, 6, 6, dtype=torch.float64)
EDGE_19[..., 3:] = 1.0                       # black on the left, white on the right: a vertical edge between columns 2 and 3
LUMA_19 = (0.299, 0.587, 0.114)              # the weights of R, G and B in the luminance (the grey of a colour)
print("PHOTO_19", tuple(PHOTO_19.shape), "· CONST_19", tuple(CONST_19.shape), "· EDGE_19", tuple(EDGE_19.shape))'''

TODO_19 = r'''def blur_19(x):
    """Box blur 3 x 3 of each channel of x (N, 3, H, W), float64: one F.conv2d with groups=3, after a replicate padding."""
    raise NotImplementedError("blur_19")


def sobel_19(x):
    """Horizontal Sobel gradient of the luminance of x (N, 3, H, W): one F.conv2d with a (1, 3, 3, 3) weight, after a
    replicate padding; returns (N, 1, H, W), positive where the image gets brighter to the right."""
    raise NotImplementedError("sobel_19")


def sharpen_19(x):
    """Sharpen each channel of x (N, 3, H, W) with the kernel [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] (groups=3, replicate
    padding), then clamp the result to [0, 1]."""
    raise NotImplementedError("sharpen_19")'''

SOLUTION_19 = r'''def blur_19(x):
    """Box blur 3 x 3 of each channel of x (N, 3, H, W), float64: one F.conv2d with groups=3, after a replicate padding."""
    weight = torch.full((3, 1, 3, 3), 1.0 / 9, dtype=x.dtype)            # (C_out, C_in / groups, 3, 3) = (3, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight, groups=3)


def sobel_19(x):
    """Horizontal Sobel gradient of the luminance of x (N, 3, H, W): one F.conv2d with a (1, 3, 3, 3) weight, after a
    replicate padding; returns (N, 1, H, W), positive where the image gets brighter to the right."""
    sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=x.dtype)
    weight = torch.stack([c * sobel for c in LUMA_19])[None]             # (1, 3, 3, 3): grey THEN Sobel, in one layer
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight)


def sharpen_19(x):
    """Sharpen each channel of x (N, 3, H, W) with the kernel [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] (groups=3, replicate
    padding), then clamp the result to [0, 1]."""
    kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]], dtype=x.dtype)
    weight = kernel.expand(3, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), weight, groups=3).clamp(0.0, 1.0)'''

CHECK_19 = r'''outs_19 = {}
for name_19, fn_19, channels_19 in [("blur_19", blur_19, 3), ("sobel_19", sobel_19, 1), ("sharpen_19", sharpen_19, 3)]:
    with checked("21.19"):
        out_19 = fn_19(PHOTO_19.clone())
        if returned("21.19", name_19, out_19):
            if not isinstance(out_19, torch.Tensor) or tuple(out_19.shape) != (1, channels_19, 32, 32):
                print(f"❌ Ex 21.19 : {name_19} doit rendre un tenseur (1, {channels_19}, 32, 32) pour la photo (1, 3, 32, 32) ; "
                      f"reçu {type(out_19).__name__} de forme {tuple(np.shape(out_19))}.")
            else:
                outs_19[name_19] = out_19
if "blur_19" in outs_19:
    with checked("21.19"):
        same("21.19", "blur_19 d'une image uniforme (les bords ne noircissent pas)", blur_19(CONST_19), CONST_19)
        padded_19 = F.pad(PHOTO_19, (1, 1, 1, 1), mode="replicate")
        same("21.19", "blur_19 contre la moyenne de chaque fenêtre 3 x 3 (F.avg_pool2d)", outs_19["blur_19"],
             F.avg_pool2d(padded_19, 3, stride=1))
if "sobel_19" in outs_19:
    with checked("21.19"):
        same("21.19", "sobel_19 d'une image uniforme", sobel_19(CONST_19), torch.zeros(1, 1, 6, 6))
        same("21.19", "sobel_19 du bord vertical (4 de part et d'autre du bord, 0 ailleurs)", sobel_19(EDGE_19),
             torch.tensor([[[[0.0, 0.0, 4.0, 4.0, 0.0, 0.0]] * 6]]))
        gray_19 = sum(c * PHOTO_19[:, k:k + 1] for k, c in enumerate(LUMA_19))
        kernel_19 = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], dtype=torch.float64)
        same("21.19", "sobel_19 contre deux étapes (la luminance, puis le Sobel sur une image grise)", outs_19["sobel_19"],
             F.conv2d(F.pad(gray_19, (1, 1, 1, 1), mode="replicate"), kernel_19))
if "sharpen_19" in outs_19:
    with checked("21.19"):
        same("21.19", "sharpen_19 d'une image uniforme", sharpen_19(CONST_19), CONST_19)
        sharp_19 = outs_19["sharpen_19"]
        verdict("21.19", float(sharp_19.min()) >= 0.0 and float(sharp_19.max()) <= 1.0, "sharpen_19 reste dans [0, 1].",
                f"sharpen_19 va de {float(sharp_19.min()):.3f} à {float(sharp_19.max()):.3f} : ramène-le dans [0, 1] (clamp).")
if len(outs_19) == 3:
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    panels_19 = [(PHOTO_19, "photo"), (outs_19["blur_19"], "flou"), (outs_19["sobel_19"].abs(), "|Sobel|"),
                 (outs_19["sharpen_19"], "netteté")]
    for ax, (img_19, title_19) in zip(axes, panels_19):
        arr_19 = img_19[0].permute(1, 2, 0).numpy()
        ax.imshow(arr_19[..., 0] if arr_19.shape[-1] == 1 else arr_19, cmap="gray")
        ax.set_title(title_19)
        ax.axis("off")
    plt.show()'''

GIVEN_20 = r'''x1_20 = torch.zeros(8, 64, 30)      # 8 sentences of 30 words, each word a vector of 64 numbers: (N, C, L) for Conv1d
x2_20 = torch.zeros(8, 1, 30, 64)   # the same sentences as 1-channel images of 30 rows (words) and 64 columns
x3_20 = torch.zeros(8, 64, 30, 30)  # 8 feature maps of 64 channels, 30 x 30
print("x1_20", tuple(x1_20.shape), "· x2_20", tuple(x2_20.shape), "· x3_20", tuple(x3_20.shape))'''

TODO_20 = r'''shape1_20 = ...    # a) nn.Conv1d(64, 16, kernel_size=5)(x1_20).shape, as a string, e.g. "(1, 2)"
params1_20 = ...   # b) the number of parameters of this Conv1d
shape2_20 = ...    # c) nn.Conv2d(1, 16, kernel_size=(5, 64))(x2_20).shape
equal_20 = ...     # d) does this Conv2d have exactly as many parameters as the Conv1d of a)? True or False
params3_20 = ...   # e) the number of parameters of nn.Conv2d(64, 16, kernel_size=1)
shape3_20 = ...    # f) nn.Conv2d(64, 16, kernel_size=1)(x3_20).shape'''

EXPERIMENT_20 = r'''torch.manual_seed(20)
conv1_20, conv2_20, conv3_20 = nn.Conv1d(64, 16, 5), nn.Conv2d(1, 16, (5, 64)), nn.Conv2d(64, 16, 1)
for layer_20, x_20 in [(conv1_20, x1_20), (conv2_20, x2_20), (conv3_20, x3_20)]:
    print(f"{str(layer_20):45} {str(tuple(x_20.shape)):18} -> {str(tuple(layer_20(x_20).shape)):18} {n_params(layer_20)} parameters")
with torch.no_grad():                     # the same weights for the full-width Conv2d: w2[o, 0, u, c] = w1[o, c, u]
    conv2_20.weight.copy_(conv1_20.weight.permute(0, 2, 1).unsqueeze(1))
    conv2_20.bias.copy_(conv1_20.bias)
    words_20 = torch.randn(8, 64, 30)
    gap_20 = (conv1_20(words_20) - conv2_20(words_20.permute(0, 2, 1).unsqueeze(1))[..., 0]).abs().max().item()
print(f"same weights: largest gap between the Conv1d and the full-width Conv2d = {gap_20:.1e}")
for letter_20, value_20 in zip("abcdef", [shape1_20, params1_20, shape2_20, equal_20, params3_20, shape3_20]):
    wb.check(f"21.20{letter_20}", shape_text(value_20) if letter_20 in "acf" else value_20)'''

RECORD_20 = r'''wb.record("21.20a", "(8, 16, 26)", mistakes={"tu as retiré tout le noyau : à sa dernière position, il tient encore dans la séquence": "(8, 16, 25)",
          "une Conv1d glisse le long de L (le dernier axe) et remplace les 64 canaux par 16 filtres": "(8, 64, 26)",
          "la sortie garde l'axe des exemples": "(16, 26)"})
wb.record("21.20b", 16 * (64 * 5 + 1), mistakes={"ajoute un biais par filtre": 16 * 64 * 5,
          "tu as oublié les canaux : chaque filtre lit tous les canaux de l'entrée": 16 * (5 + 1)})
wb.record("21.20c", "(8, 16, 26, 1)", mistakes={"le noyau couvre les 64 colonnes : il ne reste qu'une position en largeur": "(8, 16, 26, 64)",
          "la sortie garde l'axe des colonnes, de taille 1": "(8, 16, 26)", "sans padding, le noyau fait aussi rétrécir les lignes": "(8, 16, 30, 1)"})
wb.record("21.20d", True, mistakes={"compte, dans les deux couches, les poids d'un filtre et le nombre de filtres : que lisent-ils ?": False})
wb.record("21.20e", 16 * (64 + 1), mistakes={"ajoute un biais par filtre": 16 * 64, "un noyau 1 x 1 lit tous les canaux d'un pixel, pas un seul": 16 + 16})
wb.record("21.20f", "(8, 16, 30, 30)", mistakes={"une convolution 1 x 1 garde la taille de l'image": "(8, 16, 29, 29)",
          "les canaux passent de 64 à 16": "(8, 64, 30, 30)"})'''

GIVEN_21 = r'''def ink(rows):
    """A 3 x 3 filter of the first level from three strings: '#' -> +1 (ink wanted), '.' -> -1 (blank wanted)."""
    return np.array([[1.0 if c == "#" else -1.0 for c in row] for row in rows])


def wants(spec, channels):
    """A 3 x 3 filter of a higher level: +1 where the input channel must be on, 0 (don't care) everywhere else."""
    w = np.zeros((channels, 3, 3))
    for channel, (i, j) in spec:
        w[channel, i, j] = 1.0
    return w


H, V, B, L, R = range(5)                       # the five filters of level 1
L1_21 = np.stack([ink(["...", "###", "..."]), ink([".#.", ".#.", ".#."]), ink(["###", "###", "###"]),
                  ink(["...", ".##", "..."]), ink(["...", "##.", "..."])])[:, None]     # (5, 1, 3, 3)
E, N, M = range(3)                             # level 2: eye, nose, mouth
L2_21 = np.stack([wants([(H, (0, 0)), (H, (0, 1)), (H, (0, 2)), (B, (1, 1))], 5),          # eyebrow over a pupil
                  wants([(V, (0, 1)), (V, (1, 1)), (H, (2, 1))], 5),                         # a vertical line on a base
                  wants([(L, (1, 0)), (H, (1, 1)), (R, (1, 2)), (H, (2, 1))], 5)])          # a long line over a short one
L3_21 = np.stack([wants([(E, (0, 0)), (E, (0, 2)), (N, (1, 1)), (M, (2, 1))], 3),          # a face seen from the front
                  wants([(E, (0, 1)), (N, (1, 2)), (M, (2, 2))], 3)])                        # a face in profile
FACE_21 = np.zeros((27, 27))
for r_21, c0_21, c1_21 in [(1, 1, 7), (1, 19, 25), (16, 12, 14), (22, 9, 17), (25, 12, 14), (19, 19, 25), (8, 10, 11)]:
    FACE_21[r_21, c0_21:c1_21 + 1] = 1.0       # horizontal strokes
FACE_21[3:6, 3:6] = FACE_21[3:6, 21:24] = FACE_21[12:15, 3:6] = 1.0   # solid blocks
FACE_21[9:15, 13] = FACE_21[10:16, 22] = 1.0   # vertical strokes
SHIFTED_21 = np.zeros((27, 27))
SHIFTED_21[:, 1:] = FACE_21[:, :26]            # the same drawing, one pixel to the right
NAMES_21 = ["H", "V", "B", "L", "R"], ["oeil", "nez", "bouche"], ["face", "profil"]
plt.figure(figsize=(3, 3))
plt.imshow(FACE_21, cmap="gray_r")
plt.title("FACE_21 (27 x 27)")
plt.axis("off")
plt.show()'''

TODO_21 = r'''def detect_21(image):
    """The three levels of filters on a (27, 27) image, with YOUR mylearn.conv.conv2d and max_pool2d.

    Level 1 and 2: conv2d with padding=1, a match where the score equals the number of +1 of the filter (a 0/1 float
    map), then max_pool2d(…, 3). Level 3: conv2d without padding, the raw scores.
    Returns (level1, level2, scores3): (5, 9, 9), (3, 3, 3) and (2,) arrays.
    """
    raise NotImplementedError("detect_21")


def show_21(image, level1, level2, scores3):
    """The figure of the detector: the image, the 5 maps of level 1, the 3 maps of level 2 and the 2 final scores
    (a bar chart). End with plt.show()."""
    raise NotImplementedError("show_21")


n1_21 = ...       # a) the number of 1 in level1 (the 5 maps together), for FACE_21
n2_21 = ...       # b) the number of 1 in level2, for FACE_21
face_21 = ...     # c) for FACE_21: "A" (a face from the front), "B" (a profile) or "C" (no face)
shifted_21 = ...  # d) the same question for SHIFTED_21'''

SOLUTION_21 = r'''def detect_21(image):
    """The three levels of filters on a (27, 27) image, with YOUR mylearn.conv.conv2d and max_pool2d.

    Level 1 and 2: conv2d with padding=1, a match where the score equals the number of +1 of the filter (a 0/1 float
    map), then max_pool2d(…, 3). Level 3: conv2d without padding, the raw scores.
    Returns (level1, level2, scores3): (5, 9, 9), (3, 3, 3) and (2,) arrays.
    """
    C = mylearn.conv
    maps = np.asarray(image, dtype=float)[None, None]                  # (1, 1, 27, 27)
    for bank in (L1_21, L2_21):
        scores = C.conv2d(maps, bank, padding=1)
        wanted = (bank > 0).sum(axis=(1, 2, 3))                          # the number of +1 of each filter
        maps = C.max_pool2d((scores == wanted[None, :, None, None]).astype(float), 3)
        if bank is L1_21:
            level1 = maps[0]
    return level1, maps[0], C.conv2d(maps, L3_21)[0, :, 0, 0]


def show_21(image, level1, level2, scores3):
    """The figure of the detector: the image, the 5 maps of level 1, the 3 maps of level 2 and the 2 final scores
    (a bar chart). End with plt.show()."""
    fig, axes = plt.subplots(1, 10, figsize=(17, 2.2))
    panels = [(image, "image")] + list(zip(level1, NAMES_21[0])) + list(zip(level2, NAMES_21[1]))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    wanted = (L3_21 > 0).sum(axis=(1, 2, 3))
    axes[-1].bar(NAMES_21[2], scores3, color=["tab:green" if s == w else "tab:gray" for s, w in zip(scores3, wanted)])
    axes[-1].set_title("niveau 3")
    plt.tight_layout()
    plt.show()


level1_s21, level2_s21, scores3_s21 = detect_21(FACE_21)
n1_21 = int(level1_s21.sum())
n2_21 = int(level2_s21.sum())
WANTED_S21 = (L3_21 > 0).sum(axis=(1, 2, 3))
face_21 = "A" if scores3_s21[0] == WANTED_S21[0] else "B" if scores3_s21[1] == WANTED_S21[1] else "C"
scores_shift_s21 = detect_21(SHIFTED_21)[2]
shifted_21 = "A" if scores_shift_s21[0] == WANTED_S21[0] else "B" if scores_shift_s21[1] == WANTED_S21[1] else "C"'''

CHECK_21 = r'''with checked("21.21"):
    result_21 = detect_21(FACE_21.copy())
    if returned("21.21", "detect_21", result_21):
        level1_21, level2_21, scores3_21 = pair("21.21", "detect_21", result_21, names=("level1", "level2", "scores3"))
        shapes_21 = [np.shape(level1_21), np.shape(level2_21), np.shape(scores3_21)]
        verdict("21.21", shapes_21 == [(5, 9, 9), (3, 3, 3), (2,)], "detect_21 rend des cartes (5, 9, 9), (3, 3, 3) et 2 scores.",
                f"detect_21 rend des formes {shapes_21} : attendu (5, 9, 9), (3, 3, 3) et (2,) (pense au [0] qui retire l'axe des exemples).")
        if shapes_21 == [(5, 9, 9), (3, 3, 3), (2,)]:
            show_21(FACE_21, level1_21, level2_21, scores3_21)
wb.check("21.21a", n1_21)
wb.check("21.21b", n2_21)
wb.check("21.21c", face_21)
wb.check("21.21d", shifted_21)'''

RECORD_21 = r'''wb.record("21.21a", n1_21, mistakes={"compte les cases à 1 APRÈS le max pooling 3 x 3 du niveau 1 (des cartes 9 x 9)":
          int((mylearn.conv.conv2d(FACE_21[None, None], L1_21, padding=1) == (L1_21 > 0).sum(axis=(1, 2, 3))[None, :, None, None]).sum()),
          "une correspondance, c'est un score ÉGAL au nombre de +1 du filtre, pas seulement positif":
          int(mylearn.conv.max_pool2d((mylearn.conv.conv2d(FACE_21[None, None], L1_21, padding=1) > 0).astype(float), 3).sum())})
wb.record("21.21b", n2_21, mistakes={"une correspondance du niveau 2 demande TOUS ses +1 : un score égal au nombre de +1, pas seulement positif":
          int(mylearn.conv.max_pool2d((mylearn.conv.conv2d(level1_s21[None], L2_21, padding=1) > 0).astype(float), 3).sum())})
wb.record("21.21c", face_21, mistakes={"relis les scores du niveau 3 : lequel atteint le nombre de +1 de son filtre ?": "B", "un des deux filtres du niveau 3 trouve son motif : lequel ?": "C"})
wb.record("21.21d", shifted_21, mistakes={"lance detect_21 sur SHIFTED_21 : le décalage d'un pixel change les blocs du pooling": "A", "relis les scores du niveau 3 pour SHIFTED_21": "B"})'''



# ---------------------------------------------------------------------------
# Part C: the convnet of the book on Fashion-MNIST, its curves, its bugs, its initialisation, stride or pooling
# ---------------------------------------------------------------------------
PART_C_GIVEN = r'''# Fashion-MNIST for parts C and D: training and validation images, tensors (N, 1, 28, 28) in [0, 1]
CLASSES = wb.datasets.FASHION_CLASSES
X_FM, Y_FM = wb.datasets.load_fashion_mnist("train", normalize=True, seed=None)     # the 60 000 training images
ORDER_FM = np.random.default_rng(2122).permutation(len(X_FM))
N_TRAIN, N_VAL = (10_000, 2_000) if FAST_MODE else (50_000, 10_000)
X_TRAIN = torch.from_numpy(X_FM[ORDER_FM[:N_TRAIN]]).unsqueeze(1)
Y_TRAIN = torch.from_numpy(Y_FM[ORDER_FM[:N_TRAIN]])
X_VAL = torch.from_numpy(X_FM[ORDER_FM[50_000:50_000 + N_VAL]]).unsqueeze(1)
Y_VAL = torch.from_numpy(Y_FM[ORDER_FM[50_000:50_000 + N_VAL]])
print(f"X_TRAIN {tuple(X_TRAIN.shape)} · X_VAL {tuple(X_VAL.shape)} · the test images stay hidden (21.32)")'''

TODO_22 = r'''def make_convnet_22():
    """The convnet of the book (fiche §21.9), in PyTorch: an nn.Sequential that turns (N, 1, 28, 28) images into
    (N, 10) logits."""
    raise NotImplementedError("make_convnet_22")


def train_22(model, X, y, epochs=2, batch_size=128, lr=1e-3, seed=0):
    """Train `model` in place with Adam(lr) and nn.CrossEntropyLoss, on mini-batches of a new random order at each
    epoch (torch.randperm with torch.Generator().manual_seed(seed), created once). Return the list of the mean training
    loss of each epoch (Python floats)."""
    raise NotImplementedError("train_22")


in_features_22 = ...   # a) the number of inputs of the first dense layer if the images were 32 x 32 (a whole number)
params32_22 = ...      # b) the number of parameters of the whole network for 32 x 32 images'''

SOLUTION_22 = r'''def make_convnet_22():
    """The convnet of the book (fiche §21.9), in PyTorch: an nn.Sequential that turns (N, 1, 28, 28) images into
    (N, 10) logits."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3), nn.ReLU(),
        nn.Conv2d(32, 64, 3), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(64 * 12 * 12, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 10))                       # logits: the softmax is inside nn.CrossEntropyLoss


def train_22(model, X, y, epochs=2, batch_size=128, lr=1e-3, seed=0):
    """Train `model` in place with Adam(lr) and nn.CrossEntropyLoss, on mini-batches of a new random order at each
    epoch (torch.randperm with torch.Generator().manual_seed(seed), created once). Return the list of the mean training
    loss of each epoch (Python floats)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        total = 0.0
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            xb, yb = X[batch].to(DEVICE), y[batch].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)              # weighted by the size of the (last, smaller) mini-batch
        history.append(total / len(X))
    return history


in_features_22 = 64 * 14 * 14                     # 32 -> 30 -> 28 -> pooling -> 14
params32_22 = 32 * (9 + 1) + 64 * (32 * 9 + 1) + in_features_22 * 128 + 128 + 128 * 10 + 10'''

CHECK_22 = r'''model_22 = None
EPOCHS_22 = 3 if FAST_MODE else 12
with checked("21.22"):
    torch.manual_seed(2122)
    net_22 = make_convnet_22()
    if returned("21.22", "make_convnet_22", net_22):
        net_22 = net_22.to(DEVICE)
        out_22 = net_22(X_TRAIN[:5].to(DEVICE))
        verdict("21.22", tuple(out_22.shape) == (5, 10), "le réseau rend (5, 10) pour 5 images : 10 logits par image.",
                f"le réseau rend la forme {tuple(out_22.shape)} pour 5 images : attendu (5, 10).")
        kinds_22 = [type(layer).__name__ for layer in net_22]
        want_kinds_22 = ["Conv2d", "ReLU", "Conv2d", "ReLU", "MaxPool2d", "Dropout", "Flatten", "Linear", "ReLU", "Dropout", "Linear"]
        verdict("21.22", kinds_22 == want_kinds_22, "les onze couches du convnet du livre, dans l'ordre.",
                f"les couches sont {kinds_22} : attendu celles du tableau du §21.9 (la softmax n'est pas une couche : elle est dans la loss).")
        verdict("21.22", n_params(net_22) == _book_convnet_params(), "le nombre de paramètres est celui du convnet du livre.",
                f"le réseau a {n_params(net_22)} paramètres : compare-le, couche par couche, à ton calcul de ✏️ 21.4.")
        start_22 = time.perf_counter()
        history_22 = train_22(net_22, X_TRAIN, Y_TRAIN, epochs=EPOCHS_22, seed=0)
        seconds_22 = time.perf_counter() - start_22
        if returned("21.22", "train_22", history_22):
            ok_hist_22 = (isinstance(history_22, list) and len(history_22) == EPOCHS_22
                          and all(isinstance(v, float) for v in history_22))
            verdict("21.22", ok_hist_22, f"train_22 rend {EPOCHS_22} loss moyennes, des float Python.",
                    f"train_22 doit rendre une liste de {EPOCHS_22} float Python (une loss moyenne par epoch) ; reçu {history_22!r:.120}.")
            if ok_hist_22:
                print("training loss per epoch:", [round(v, 4) for v in history_22], f"· {seconds_22:.0f} s")
                verdict("21.22", history_22[-1] < history_22[0], "la loss d'entraînement baisse d'une epoch à l'autre.",
                        "la loss d'entraînement ne baisse pas : vérifie zero_grad, backward et step, et le learning rate.")
            acc_22 = accuracy(net_22, X_VAL, Y_VAL)
            target_22 = 0.80 if FAST_MODE else 0.90
            verdict("21.22", acc_22 >= target_22, f"accuracy de validation {acc_22:.3f} (au moins {target_22:.2f} attendu).",
                    f"accuracy de validation {acc_22:.3f} : attendu au moins {target_22:.2f} ; relis le contrat de train_22.")
            model_22 = net_22                          # reused by 21.28 and 21.29
wb.check("21.22a", in_features_22)
wb.check("21.22b", params32_22)'''

RECORD_22 = r'''wb.record("21.22a", in_features_22, mistakes={"le pooling vient après les DEUX convolutions, sans padding chacune": 64 * 15 * 15,
          "n'oublie pas le pooling, qui divise le côté par 2": 64 * 28 * 28, "c'est la taille pour des images 28 x 28 : refais les tailles avec 32": 9216})
wb.record("21.22b", params32_22, mistakes={"ajoute les biais de chaque couche": params32_22 - 32 - 64 - 128 - 10,
          "seule la première couche dense change : les convolutions gardent leurs poids quelle que soit la taille de l'image": 1_199_882})'''

# --- 📈 21.23: two precomputed histories (tools/chapters/history_ch21.py) and a confusion matrix ---
import json as _json

import numpy as np  # noqa: E402

_HISTORY_23 = _json.loads(Path(__file__).with_name("history_ch21.json").read_text())


def _given_23() -> str:
    lines = ["# Two runs of the convnet of the book, 15 epochs on 50 000 images (precomputed: too long for this notebook).",
             "# After each epoch, measured in evaluation mode on 10 000 training images and on the 10 000 validation images.",
             "RUNS_23 = {"]
    for letter, key in (("A", "no_dropout"), ("B", "dropout")):
        lines.append(f'    "{letter}": {{')
        for metric in ("train_loss", "train_acc", "val_loss", "val_acc"):
            lines.append(f'        "{metric}": {[e[metric] for e in _HISTORY_23[key]]},')
        lines.append("    },")
    lines.append("}")
    cm = _HISTORY_23["dropout"][-1]["cm"]
    lines.append("CM_23 = np.array([   # run B after its 15th epoch: row = true class, column = predicted class")
    lines += [f"    {row}," for row in cm]
    lines.append("])")
    lines.append(GIVEN_23_PLOT)
    return "\n".join(lines)


GIVEN_23_PLOT = r'''print("classes:", ", ".join(f"{k} {name}" for k, name in enumerate(CLASSES)))
epochs_23 = np.arange(1, 16)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for (name_23, run_23), color_23 in zip(RUNS_23.items(), ["tab:red", "tab:blue"]):
    axes[0].plot(epochs_23, run_23["train_loss"], ls="--", color=color_23, label=f"{name_23} : entraînement")
    axes[0].plot(epochs_23, run_23["val_loss"], color=color_23, label=f"{name_23} : validation")
    axes[1].plot(epochs_23, run_23["train_acc"], ls="--", color=color_23, label=f"{name_23} : entraînement")
    axes[1].plot(epochs_23, run_23["val_acc"], color=color_23, label=f"{name_23} : validation")
axes[0].set_title("loss (nats)")
axes[1].set_title("accuracy")
for ax in axes[:2]:
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
wb.plot.plot_confusion_matrix(CM_23, CLASSES, ax=axes[2], title="CM_23 : B après 15 epochs (10 000 images de validation)")
plt.tight_layout()
plt.show()'''

TODO_23 = r'''def per_class_23(cm):
    """(recall, precision) of each class for a confusion matrix cm (row = true class, column = predicted class): two
    float arrays of shape (K,). A class with an empty row has a recall of 0, one with an empty column a precision of 0."""
    raise NotImplementedError("per_class_23")


dropout_23 = ...   # a) the run trained WITH its two dropouts: "A" or "B"
stop_23 = ...      # b) for run A, the epoch whose weights an early stopping on the validation loss would keep (the first is 1)
gap_23 = ...       # c) for run A, at its last epoch: training accuracy minus validation accuracy (checked to 3 decimals)
worst_23 = ...     # d) in CM_23, the index (0 to 9) of the class with the lowest recall
instead_23 = ...   # e) the index of the OTHER class most often predicted for the images of class worst_23
share_23 = ...     # f) the share of the errors of CM_23 whose true class OR predicted class is worst_23 (3 decimals)'''

SOLUTION_23 = r'''def per_class_23(cm):
    """(recall, precision) of each class for a confusion matrix cm (row = true class, column = predicted class): two
    float arrays of shape (K,). A class with an empty row has a recall of 0, one with an empty column a precision of 0."""
    cm = np.asarray(cm, dtype=float)
    diag = np.diag(cm)
    rows, cols = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(diag, rows, out=np.zeros_like(diag), where=rows > 0)
    precision = np.divide(diag, cols, out=np.zeros_like(diag), where=cols > 0)
    return recall, precision


recall_23, precision_23 = per_class_23(CM_23)
errors_23 = CM_23 - np.diag(np.diag(CM_23))          # the off-diagonal cells: the errors only
dropout_23 = "B"
stop_23 = int(np.argmin(RUNS_23["A"]["val_loss"])) + 1
gap_23 = RUNS_23["A"]["train_acc"][-1] - RUNS_23["A"]["val_acc"][-1]
worst_23 = int(np.argmin(recall_23))
instead_23 = int(np.argmax(errors_23[worst_23]))
share_23 = (errors_23[worst_23].sum() + errors_23[:, worst_23].sum()) / errors_23.sum()'''

CHECK_23 = r'''with checked("21.23"):
    if written(lambda: per_class_23(np.eye(2))):
        recall_c23, precision_c23 = pair("21.23", "per_class_23", per_class_23(CM_23), names=("recall", "precision"))
        same("21.23", "le recall de chaque classe de CM_23 (la diagonale divisée par la somme de la LIGNE)", recall_c23,
             np.diag(CM_23) / CM_23.sum(axis=1))
        same("21.23", "la precision de chaque classe de CM_23 (la diagonale divisée par la somme de la COLONNE)", precision_c23,
             np.diag(CM_23) / CM_23.sum(axis=0))
        small_23 = np.array([[3, 0, 1], [1, 0, 0], [0, 0, 0]])      # class 1 never predicted, class 2 never present
        recall_s23, precision_s23 = pair("21.23", "per_class_23", per_class_23(small_23), names=("recall", "precision"))
        same("21.23", "per_class_23 d'une matrice dont une ligne et une colonne sont vides (0, sans nan ni avertissement)",
             np.concatenate([recall_s23, precision_s23]), [0.75, 0.0, 0.0, 0.75, 0.0, 0.0])
    else:
        print("⏳ Ex 21.23 : écris per_class_23.")
for letter_23, value_23 in zip("abcdef", [dropout_23, stop_23, gap_23, worst_23, instead_23, share_23]):
    wb.check(f"21.23{letter_23}", value_23)'''

def _mistakes_23(answer, items, decimals=None) -> str:
    """The mistakes dict of a record, as code: a message for each wrong value that differs from the answer and from
    the values already kept (wb.record refuses both)."""
    def key(v):
        return round(float(v), decimals) if decimals is not None else v
    kept, seen = [], {key(answer)}
    for message, value in items:
        value = round(float(value), 4) if decimals is not None else int(value) if not isinstance(value, str) else value
        if key(value) not in seen:
            seen.add(key(value))
            kept.append(f"{message!r}: {value!r}")
    return "{" + ",\n          ".join(kept) + "}"


def _record_23() -> str:
    runs = {name: {m: np.array([e[m] for e in _HISTORY_23[key]]) for m in ("train_loss", "train_acc", "val_loss", "val_acc")}
            for name, key in (("A", "no_dropout"), ("B", "dropout"))}
    A, B = runs["A"], runs["B"]
    cm = np.array(_HISTORY_23["dropout"][-1]["cm"])
    errors = cm - np.diag(np.diag(cm))
    recall, precision = np.diag(cm) / cm.sum(axis=1), np.diag(cm) / cm.sum(axis=0)
    stop = int(np.argmin(A["val_loss"])) + 1
    gap = A["train_acc"][-1] - A["val_acc"][-1]
    worst = int(np.argmin(recall))
    instead = int(np.argmax(errors[worst]))
    share = (errors[worst].sum() + errors[:, worst].sum()) / errors.sum()
    lines = [
        'wb.record("21.23a", "B", mistakes={"regarde l\'écart entre l\'entraînement et la validation : le dropout le réduit, et la loss de validation de A remonte": "A"})',
        'wb.record("21.23b", stop_23, mistakes=' + _mistakes_23(stop, [
            ("les epochs sont comptées à partir de 1", stop - 1),
            ("on garde le minimum de la loss de VALIDATION, pas celui de la loss d'entraînement", np.argmin(A["train_loss"]) + 1),
            ("le minimum de la loss de validation, pas le maximum de l'accuracy de validation", np.argmax(A["val_acc"]) + 1),
            ("c'est l'epoch de B ; b) demande celle de A", np.argmin(B["val_loss"]) + 1)]) + ")",
        'wb.record("21.23c", gap_23, decimals=3, mistakes=' + _mistakes_23(gap, [
            ("c'est l'écart de B ; c) demande celui de A", B["train_acc"][-1] - B["val_acc"][-1]),
            ("l'accuracy d'entraînement MOINS celle de validation : un nombre positif", -gap),
            ("c'est l'écart des LOSSES ; c) demande celui des accuracies", A["val_loss"][-1] - A["train_loss"][-1])], decimals=3) + ")",
        'wb.record("21.23d", worst_23, mistakes=' + _mistakes_23(worst, [
            ("c'est la classe la plus souvent prédite À TORT ; d) demande celle dont les images sont le moins bien reconnues", np.argmax(errors.sum(axis=0))),
            ("le recall se lit sur les LIGNES (la classe vraie), pas sur les colonnes (la precision)", np.argmin(precision))]) + ")",
        'wb.record("21.23e", instead_23, mistakes=' + _mistakes_23(instead, [
            ("une AUTRE classe que worst_23 : la diagonale ne compte pas", worst),
            ("tu as lu la colonne de worst_23 (les images PRÉDITES worst_23) ; e) demande sa ligne (les images de cette classe)", np.argmax(errors[:, worst]))]) + ")",
        'wb.record("21.23f", share_23, decimals=3, mistakes=' + _mistakes_23(share, [
            ("compte aussi les erreurs où worst_23 est la classe PRÉDITE (sa colonne)", errors[worst].sum() / errors.sum()),
            ("compte aussi les erreurs où worst_23 est la classe VRAIE (sa ligne)", errors[:, worst].sum() / errors.sum()),
            ("divise par le nombre d'ERREURS, pas par le nombre d'images", (errors[worst].sum() + errors[:, worst].sum()) / cm.sum())], decimals=3) + ")",
    ]
    return "\n".join(lines)


EX_23 = Ex("21.23", "📈", 2, 15, "Diagnostiquer les courbes et la matrice de confusion du CNN",
       "lire sur deux entraînements complets l'effet du dropout, choisir l'epoch d'un early stopping, et trouver dans une "
       "matrice de confusion la classe que le réseau reconnaît le moins bien.",
       "Ex 21.22 · ch. 3 (matrice de confusion, recall, precision) · Ex 20.25 · fiche §21.9", thread="Fashion-MNIST",
       tracks="R, C",
       body=r"""Deux entraînements complets du convnet de 21.22, trop longs pour ce notebook (un quart d'heure chacun sur un CPU), ont été faits pour toi (`tools/chapters/history_ch21.py`) : 15 epochs d'Adam sur 50 000 images d'entraînement de Fashion-MNIST, l'un **avec** les deux dropouts du livre, l'autre **sans**, tout le reste identique. Après chaque epoch, la loss et l'accuracy ont été mesurées **en mode évaluation** (sans dropout) sur 10 000 des images d'entraînement et sur les 10 000 images de validation : les deux courbes sont donc comparables. Les historiques sont dans `RUNS_23` (des listes de 15 valeurs) ; `CM_23` est la matrice de confusion de l'entraînement B après sa dernière epoch, sur les 10 000 images de validation (ligne = classe vraie, colonne = classe prédite).

Écris `per_class_23(cm)`, qui rend `(recall, precision)` : le recall et la precision de chaque classe (ch. 3), deux tableaux de réels de forme `(K,)` ; une classe dont la ligne est vide a un recall de 0, une classe jamais prédite (colonne vide) une precision de 0, sans `nan` ni avertissement. Puis, en calculant chaque réponse à partir de `RUNS_23` et de `CM_23` :
a) `dropout_23` : l'entraînement fait **avec** les dropouts (`"A"` ou `"B"`) ;
b) `stop_23` : pour A, l'epoch dont un early stopping sur la loss de validation garderait les poids (la première epoch est la n° 1) ;
c) `gap_23` : pour A, à sa dernière epoch, l'accuracy d'entraînement moins l'accuracy de validation (vérifiée à 3 décimales) ;
d) `worst_23` : dans `CM_23`, l'indice (0 à 9) de la classe de plus faible recall ;
e) `instead_23` : l'indice de l'**autre** classe le plus souvent prédite pour les images de la classe `worst_23` ;
f) `share_23` : la part des erreurs de `CM_23` (les cases hors de la diagonale) dont la classe vraie **ou** la classe prédite est `worst_23` (vérifiée à 3 décimales).

Dans tes notes : après l'epoch de b), la loss de validation de A remonte ; son accuracy de validation baisse-t-elle autant ? Qu'est-ce qui peut faire monter une loss sans changer les prédictions ? Regarde les quatre classes Shirt, T-shirt/top, Pullover et Coat dans la matrice : que faudrait-il au réseau pour mieux les séparer ?""",
       given=_given_23(), todo=TODO_23, check=CHECK_23, solution=SOLUTION_23 + "\n" + CHECK_23, record=_record_23(),
       note="A n'a pas de dropout : sa loss d'entraînement plonge, et sa loss de validation passe par un minimum puis "
            "remonte (de 0,228 à l'epoch 7 à 0,390 à la 15ᵉ), alors que son accuracy de validation reste presque plate (0,920 puis 0,921). Une loss de cross-entropy monte quand "
            "le réseau devient **trop sûr de lui** sur les images qu'il rate (une probabilité de 0,99 sur la mauvaise "
            "classe coûte bien plus que 0,6), sans que la classe prédite change. Shirt est la classe la moins bien "
            "reconnue, surtout prise pour un T-shirt/top : ces quatre hauts ne se distinguent que par des détails (col, "
            "boutons, manches) qu'une image 28 × 28 en gris rend mal ; il faudrait plus de filtres et plus de "
            "profondeur, de l'augmentation de données (ch. 24), ou accepter que certaines images sont ambiguës même "
            "pour un humain.")


GIVEN_24 = r'''X_NHWC_24 = X_TRAIN[:8].permute(0, 2, 3, 1).contiguous()    # 8 images in the layout of a Keras notebook: (N, 28, 28, 1)
print("X_NHWC_24:", tuple(X_NHWC_24.shape))'''

TODO_24 = r'''class Net24(nn.Module):
    """The convnet of the book without its dropout (32 then 64 filters 3 x 3, a pooling, a dense layer of 128), translated
    from a Keras notebook: its forward pass receives the images in the Keras layout (N, 28, 28, 1) and must return
    (N, 10) logits. Fix it in place (four bugs)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):                        # x: (N, 28, 28, 1), the layout of the Keras notebook
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)'''

SOLUTION_24 = r'''class Net24(nn.Module):
    """The convnet of the book without its dropout (32 then 64 filters 3 x 3, a pooling, a dense layer of 128), translated
    from a Keras notebook: its forward pass receives the images in the Keras layout (N, 28, 28, 1) and must return
    (N, 10) logits. Fix it in place (four bugs)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)        # bug 2: conv2 reads the 32 channels of conv1
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # bug 3: 28 -> 26 -> 24 -> pooling -> 12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):                        # x: (N, 28, 28, 1), the layout of the Keras notebook
        x = x.permute(0, 3, 1, 2)                # bug 1: PyTorch wants (N, C, H, W)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)        # bug 4: keep the axis of the examples
        x = torch.relu(self.fc1(x))
        return self.fc2(x)'''

CHECK_24 = r'''def diagnose_24(net_class):
    """The four points of 21.24, in the order where the bugs hide one another; a point is checked only when the
    previous ones pass. Returns the number of points passed."""
    torch.manual_seed(24)
    try:
        net = net_class()
    except NotImplementedError:
        raise
    except Exception as exc:
        print(f"❌ Ex 21.24 : Net24() lève {type(exc).__name__} : {str(exc)[:200]}")
        return 0
    seen = {}

    def spy(name):
        def hook(module, inputs):                 # a pre-hook that returns None leaves the inputs unchanged
            seen.setdefault(name, tuple(inputs[0].shape))
        return hook

    hooks = [getattr(net, name).register_forward_pre_hook(spy(name)) for name in ("conv1", "conv2", "fc1", "fc2")
             if isinstance(getattr(net, name, None), nn.Module)]
    error = None
    try:
        out = net(X_NHWC_24[:2].clone())
    except Exception as exc:                     # the bugs still there raise errors: they are diagnosed below
        out, error = None, exc
    for hook in hooks:
        hook.remove()
    points = [
        ("conv1", seen.get("conv1") == (2, 1, 28, 28),
         f"conv1 reçoit un tenseur de forme {seen.get('conv1')} pour 2 images : une couche Conv2d lit l'axe 1 comme "
         "celui des canaux, et attend (N, C, H, W) = (2, 1, 28, 28)"),
        ("conv2", "conv2" in seen and seen["conv2"][1] == net.conv2.in_channels,
         f"conv2 attend {getattr(net.conv2, 'in_channels', '?')} canaux d'entrée et en reçoit "
         f"{seen['conv2'][1] if 'conv2' in seen else '?'} (la sortie de conv1)"),
        ("fc1", "fc1" in seen and len(seen["fc1"]) == 2 and seen["fc1"][0] == 2,
         f"fc1 reçoit un tenseur de forme {seen.get('fc1')} pour 2 images : une couche dense attend une ligne par image, "
         "(2, valeurs par image)"),
    ]
    for k, (name, ok, message) in enumerate(points, start=1):
        if not ok:
            detail = f" (erreur : {type(error).__name__} : {str(error)[:150]})" if error is not None else ""
            print(f"❌ Ex 21.24 : point {k} : {message}{detail}.")
            for later in range(k + 1, 5):
                print(f"⏳ Ex 21.24 : point {later} : corrige d'abord le point {k}.")
            return k - 1
        print(f"✅ Ex 21.24 : point {k} : {name} reçoit ce qu'elle attend.")
    shape_ok = error is None and out is not None and tuple(out.shape) == (2, 10)
    if shape_ok:
        with torch.no_grad():
            alone = net(X_NHWC_24[:1].clone())
            pair_out = net(X_NHWC_24[:2].clone())
        shape_ok = tuple(alone.shape) == (1, 10) and torch.allclose(alone[0], pair_out[0], atol=1e-6)
    if not shape_ok:
        if error is not None:
            print(f"❌ Ex 21.24 : point 4 : fc1 n'accepte pas ce qu'elle reçoit (erreur : {type(error).__name__} : "
                  f"{str(error)[:150]}) : compare son in_features au nombre de valeurs qu'une image donne après le pooling.")
        else:
            print(f"❌ Ex 21.24 : point 4 : le réseau rend la forme {tuple(out.shape)} pour 2 images (attendu (2, 10)), ou "
                  "la sortie d'une image change quand on lui ajoute une voisine.")
        return 3
    print("✅ Ex 21.24 : point 4 : (N, 10) pour N = 1 et 2, et chaque image ne dépend que d'elle-même.")
    return 4


with checked("21.24"):
    passed_24 = diagnose_24(Net24)
    if passed_24 == 4:
        verdict("21.24", n_params(Net24()) == _book_convnet_params(), "le réseau réparé a les paramètres du convnet du livre (sans dropout).",
                f"le réseau réparé a {n_params(Net24())} paramètres : attendu ceux de ✏️ 21.4.")'''

GIVEN_25 = r'''def small_cnn_25():
    """A small CNN for the experiment: 8 filters 3 x 3, a pooling, one dense layer."""
    return nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(8 * 13 * 13, 10))


print(small_cnn_25())'''

TODO_25 = r'''def init_25(model, kind):
    """Initialise IN PLACE the weights and biases of every nn.Conv2d and nn.Linear of `model`:
    "constant": every weight 0.05 and every bias 0 ; "default": the initialisation of PyTorch (reset_parameters) ;
    "he": nn.init.kaiming_normal_(weight, nonlinearity="relu") and biases 0. Any other kind: ValueError."""
    raise NotImplementedError("init_25")


def distinct_filters_25(weight, tol=1e-6):
    """The number of distinct filters of a convolution weight (C_out, C_in, k, k) (a tensor or an array): two filters
    are the same when all their values differ by at most tol."""
    raise NotImplementedError("distinct_filters_25")'''

SOLUTION_25 = r'''def init_25(model, kind):
    """Initialise IN PLACE the weights and biases of every nn.Conv2d and nn.Linear of `model`:
    "constant": every weight 0.05 and every bias 0 ; "default": the initialisation of PyTorch (reset_parameters) ;
    "he": nn.init.kaiming_normal_(weight, nonlinearity="relu") and biases 0. Any other kind: ValueError."""
    if kind not in ("constant", "default", "he"):
        raise ValueError(f"kind must be 'constant', 'default' or 'he', got {kind!r}")
    for layer in model.modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        if kind == "default":
            layer.reset_parameters()
            continue
        with torch.no_grad():
            if kind == "constant":
                layer.weight.fill_(0.05)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            layer.bias.zero_()


def distinct_filters_25(weight, tol=1e-6):
    """The number of distinct filters of a convolution weight (C_out, C_in, k, k) (a tensor or an array): two filters
    are the same when all their values differ by at most tol."""
    filters = as_numpy(weight).reshape(len(weight), -1)
    kept = []
    for f in filters:
        if not any(np.max(np.abs(f - g)) <= tol for g in kept):
            kept.append(f)
    return len(kept)'''

CHECK_25 = r'''results_25 = {}
with checked("21.25"):
    probe_25 = small_cnn_25()
    init_25(probe_25, "constant")
    layers_25 = [m for m in probe_25.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    verdict("21.25", all(torch.allclose(m.weight.detach(), torch.full_like(m.weight, 0.05)) for m in layers_25)
            and all(float(m.bias.detach().abs().max()) == 0.0 for m in layers_25),
            "init_25(model, 'constant') : tous les poids à 0,05, tous les biais à 0.",
            "init_25(model, 'constant') : attendu tous les poids à 0,05 et tous les biais à 0, dans les convolutions ET la couche dense.")
    refuses("21.25", "init_25 avec un nom inconnu", lambda: init_25(small_cnn_25(), "xavier"))
    same_w_25 = torch.ones(4, 1, 3, 3)
    same_w_25[2] += 0.5
    verdict("21.25", distinct_filters_25(same_w_25) == 2 and distinct_filters_25(torch.randn(5, 2, 3, 3)) == 5,
            "distinct_filters_25 compte 2 filtres distincts parmi 4 (trois égaux), et 5 parmi 5 filtres aléatoires.",
            "distinct_filters_25 : attendu 2 pour quatre filtres dont trois égaux, et 5 pour cinq filtres aléatoires.")
    for kind_25 in ("constant", "default", "he"):
        torch.manual_seed(25)
        net_25 = small_cnn_25()
        init_25(net_25, kind_25)
        net_25 = net_25.to(DEVICE)
        train_22(net_25, X_TRAIN[:5000], Y_TRAIN[:5000], epochs=1, seed=0)
        results_25[kind_25] = (distinct_filters_25(net_25[0].weight.detach().cpu()), accuracy(net_25, X_VAL, Y_VAL))
        print(f"{kind_25:9}: {results_25[kind_25][0]} distinct filters out of 8 after one epoch · validation accuracy {results_25[kind_25][1]:.3f}")
    verdict("21.25", results_25["constant"][0] == 1, "les filtres partis de la même valeur sont restés identiques.",
            f"init 'constant' : {results_25['constant'][0]} filtres distincts après une epoch ; attendu 1 (relis init_25 et distinct_filters_25).")
    verdict("21.25", results_25["default"][0] == 8 and results_25["he"][0] == 8, "les départs aléatoires gardent 8 filtres distincts.",
            f"départs aléatoires : {results_25['default'][0]} et {results_25['he'][0]} filtres distincts ; attendu 8.")'''

GIVEN_26 = r'''N_26 = 5_000 if FAST_MODE else 50_000
EPOCHS_26 = 2 if FAST_MODE else 10
X_C26, Y_C26 = wb.datasets.load_cifar10("train", n=N_26, normalize=True, channels_first=True, seed=26)
X_C26, Y_C26 = torch.from_numpy(X_C26), torch.from_numpy(Y_C26)
X_V26, Y_V26 = wb.datasets.load_cifar10("test", n=1_000, normalize=True, channels_first=True, seed=26)
X_V26, Y_V26 = torch.from_numpy(X_V26), torch.from_numpy(Y_V26)
print("CIFAR-10:", tuple(X_C26.shape), "for training,", tuple(X_V26.shape), "for validation")'''

TODO_26 = r'''def make_pool_26():
    """(N, 3, 32, 32) -> (N, 10): two 3 x 3 convolutions (32 then 64 filters, padding 1, stride 1), each followed by
    ReLU and a max pooling 2 x 2, then nn.Flatten and one dense layer."""
    raise NotImplementedError("make_pool_26")


def make_stride_26():
    """(N, 3, 32, 32) -> (N, 10): the same two convolutions with a stride of 2 (padding 1) and no pooling, each followed
    by ReLU, then nn.Flatten and one dense layer."""
    raise NotImplementedError("make_stride_26")'''

SOLUTION_26 = r'''def make_pool_26():
    """(N, 3, 32, 32) -> (N, 10): two 3 x 3 convolutions (32 then 64 filters, padding 1, stride 1), each followed by
    ReLU and a max pooling 2 x 2, then nn.Flatten and one dense layer."""
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 8 * 8, 10))


def make_stride_26():
    """(N, 3, 32, 32) -> (N, 10): the same two convolutions with a stride of 2 (padding 1) and no pooling, each followed
    by ReLU, then nn.Flatten and one dense layer."""
    return nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Flatten(), nn.Linear(64 * 8 * 8, 10))'''

CHECK_26 = r'''def macs_26(model, x):
    """Multiply-adds of the convolutions and dense layers of `model` for one image (counted with forward hooks)."""
    total = [0]

    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            total[0] += output[0].numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            total[0] += module.in_features * module.out_features

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x[:1])
    for hook in hooks:
        hook.remove()
    return total[0]


rows_26 = {}
for name_26, make_26 in [("pooling", make_pool_26), ("stride", make_stride_26)]:
    with checked("21.26"):
        torch.manual_seed(26)
        net_26 = make_26()
        if not returned("21.26", name_26, net_26):
            continue
        net_26 = net_26.to(DEVICE)
        out_26 = net_26(X_C26[:4].to(DEVICE))
        verdict("21.26", tuple(out_26.shape) == (4, 10), f"{name_26} : 10 logits par image.",
                f"{name_26} : le réseau rend {tuple(out_26.shape)} pour 4 images, attendu (4, 10).")
        start_26 = time.perf_counter()
        train_22(net_26, X_C26, Y_C26, epochs=EPOCHS_26, seed=0)
        rows_26[name_26] = (n_params(net_26), macs_26(net_26, X_C26.to(DEVICE)), time.perf_counter() - start_26,
                            accuracy(net_26, X_V26, Y_V26))
if rows_26:
    print(f"{'network':9} {'parameters':>11} {'mult-adds/image':>16} {'training (s)':>13} {'val accuracy':>13}")
    for name_26, (p_26, m_26, t_26, a_26) in rows_26.items():
        print(f"{name_26:9} {p_26:11d} {m_26:16,d} {t_26:13.1f} {a_26:13.3f}".replace(",", " "))
if len(rows_26) == 2:
    verdict("21.26", rows_26["pooling"][0] == rows_26["stride"][0], "les deux réseaux ont le même nombre de paramètres.",
            f"les deux réseaux ont {rows_26['pooling'][0]} et {rows_26['stride'][0]} paramètres : avec le même contrat, ils devraient être égaux.")
    verdict("21.26", (rows_26["pooling"][1], rows_26["stride"][1]) == (5_644_288, 1_441_792),
            f"multiplications-additions par image : {rows_26['pooling'][1]:,} contre {rows_26['stride'][1]:,}".replace(",", " ") + ", 3,9 fois moins avec le stride.",
            "les multiplications-additions par image ne sont pas celles du contrat : vérifie les strides, le padding et les poolings.")'''

# ---------------------------------------------------------------------------
# Part D: the filters of a pretrained network, gradient ascent, FGSM, transposed convolution, backward pass, challenge
# ---------------------------------------------------------------------------
GIVEN_27 = r'''RESNET_27 = WEIGHTS_27 = None
try:
    from torchvision.models import ResNet18_Weights, resnet18
    WEIGHTS_27 = ResNet18_Weights.DEFAULT                     # IMAGENET1K_V1: 1 000 classes, about 45 MB
    RESNET_27 = resnet18(weights=WEIGHTS_27).eval()
except Exception as exc:                                    # no network, or no torchvision: done on Colab
    print(f"⏳ Ex 21.27 : les poids de ResNet-18 ne se téléchargent pas ici ({type(exc).__name__}) : cet exercice se fait sur Colab.")
X_27, Y_27 = wb.datasets.load_cifar10("test", n=4, seed=27, channels_first=True)     # 4 photos, uint8 (4, 3, 32, 32)
PHOTOS_27 = torch.from_numpy(X_27)
if RESNET_27 is not None:
    INPUT_27 = WEIGHTS_27.transforms()(PHOTOS_27)            # resized to 224 x 224 and normalised like ImageNet
    print("ResNet-18 ready · input", tuple(INPUT_27.shape), "· CIFAR-10 classes of the photos:",
          [wb.datasets.CIFAR10_CLASSES[k] for k in Y_27])'''

TODO_27 = r'''def filters_27(model):
    """The 64 filters of model.conv1 as an array (64, 7, 7, 3) to display as images: the channels last, and each
    filter rescaled to [0, 1] by its own minimum and maximum."""
    raise NotImplementedError("filters_27")


def maps_27(model, x, layer):
    """The output of `layer` (a module of `model`, e.g. model.layer1) when the model runs on x, in evaluation mode and
    without autograd, captured by a forward hook that is removed before returning (even after an error)."""
    raise NotImplementedError("maps_27")'''

SOLUTION_27 = r'''def filters_27(model):
    """The 64 filters of model.conv1 as an array (64, 7, 7, 3) to display as images: the channels last, and each
    filter rescaled to [0, 1] by its own minimum and maximum."""
    w = model.conv1.weight.detach().cpu().double().permute(0, 2, 3, 1).numpy()    # (64, 3, 7, 7) -> (64, 7, 7, 3)
    low = w.min(axis=(1, 2, 3), keepdims=True)
    high = w.max(axis=(1, 2, 3), keepdims=True)
    return (w - low) / (high - low)


def maps_27(model, x, layer):
    """The output of `layer` (a module of `model`, e.g. model.layer1) when the model runs on x, in evaluation mode and
    without autograd, captured by a forward hook that is removed before returning (even after an error)."""
    captured = {}
    hook = layer.register_forward_hook(lambda module, inputs, output: captured.setdefault("out", output.detach()))
    try:
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        hook.remove()
    return captured["out"]'''

CHECK_27 = r'''if RESNET_27 is None:
    print("⏳ Ex 21.27 : pas de ResNet-18 pré-entraîné ici (voir la cellule précédente).")
else:
    with checked("21.27"):
        f_27 = filters_27(RESNET_27)
        if returned("21.27", "filters_27", f_27):
            f_27 = as_numpy(f_27)
            ok_27 = f_27 is not None and f_27.shape == (64, 7, 7, 3)
            verdict("21.27", ok_27, "filters_27 rend 64 filtres 7 x 7 en couleur, canaux à la fin.",
                    f"filters_27 : attendu un tableau (64, 7, 7, 3), reçu {None if f_27 is None else f_27.shape}.")
            if ok_27:
                verdict("21.27", np.allclose(f_27.min(axis=(1, 2, 3)), 0) and np.allclose(f_27.max(axis=(1, 2, 3)), 1),
                        "chaque filtre va de 0 à 1 (son propre minimum et son propre maximum).",
                        "chaque filtre doit aller exactement de 0 à 1 : ramène-le par SON minimum et SON maximum, pas ceux des 64.")
                w_27 = RESNET_27.conv1.weight.detach().double()
                verdict("21.27", np.allclose(np.argsort(f_27[5].ravel()), np.argsort(w_27[5].permute(1, 2, 0).numpy().ravel())),
                        "l'ordre des valeurs d'un filtre est conservé (un simple changement d'échelle).",
                        "les valeurs d'un filtre ne sont plus dans le même ordre : as-tu permuté les axes autrement que (64, 7, 7, 3) ?")
                fig, axes = plt.subplots(4, 16, figsize=(12, 3.2))
                for ax, filt in zip(axes.ravel(), f_27):
                    ax.imshow(filt)
                    ax.axis("off")
                fig.suptitle("Les 64 filtres 7 x 7 de conv1 (ResNet-18 pré-entraîné sur ImageNet)")
                plt.show()
    with checked("21.27"):
        x_27 = INPUT_27[:1]
        m1_27 = maps_27(RESNET_27, x_27, RESNET_27.layer1)
        if returned("21.27", "maps_27", m1_27):
            verdict("21.27", tuple(m1_27.shape) == (1, 64, 56, 56), "maps_27(…, layer1) : 64 cartes de 56 x 56.",
                    f"maps_27(…, layer1) : attendu la forme (1, 64, 56, 56), reçu {tuple(np.shape(m1_27))}.")
            with torch.no_grad():
                r = RESNET_27
                want_27 = r.layer1(r.maxpool(r.relu(r.bn1(r.conv1(x_27)))))
            same("21.27", "maps_27(…, layer1) contre les couches appelées une à une", m1_27, want_27, rtol=1e-5, atol=1e-5)
            verdict("21.27", len(RESNET_27.layer1._forward_hooks) == 0, "le hook est retiré après l'appel.",
                    "un hook reste attaché à layer1 : retire-le (hook.remove()), sinon chaque appel en ajoute un.")
            m4_27 = maps_27(RESNET_27, x_27, RESNET_27.layer4)
            verdict("21.27", tuple(m4_27.shape) == (1, 512, 7, 7), "maps_27(…, layer4) : 512 cartes de 7 x 7.",
                    f"maps_27(…, layer4) : attendu (1, 512, 7, 7), reçu {tuple(np.shape(m4_27))}.")
            m3_27 = maps_27(RESNET_27, x_27, RESNET_27.layer3)
            fig, axes = plt.subplots(2, 9, figsize=(13, 3.2))
            axes[0, 0].imshow(PHOTOS_27[0].permute(1, 2, 0))
            axes[0, 0].set_title("photo")
            axes[1, 0].axis("off")
            for k in range(8):
                axes[0, k + 1].imshow(m1_27[0, k], cmap="magma")
                axes[0, k + 1].set_title(f"layer1 · {k}", fontsize=8)
                axes[1, k + 1].imshow(m3_27[0, k], cmap="magma")
                axes[1, k + 1].set_title(f"layer3 · {k}", fontsize=8)
            for ax in axes.ravel():
                ax.set_xticks([])
                ax.set_yticks([])
            plt.show()
    with torch.no_grad():
        probs_27 = RESNET_27(INPUT_27).softmax(dim=1)
    for k, (p_27, c_27) in enumerate(zip(*probs_27.max(dim=1))):
        print(f"photo {k} ({wb.datasets.CIFAR10_CLASSES[Y_27[k]]}): ImageNet says {WEIGHTS_27.meta['categories'][c_27]!r} ({p_27:.0%})")'''

TODO_28 = r'''def ascent_28(model, upto, channel, steps=60, lr=0.1, seed=0):
    """The (1, 1, 28, 28) image, in [0, 1], that makes the channel `channel` of model[:upto + 1] respond as much as
    possible (its mean over the positions), found by gradient ascent on the PIXELS.

    Start: 0.4 + 0.2 * torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(seed)), in float32. Each step:
    the loss -mean(model[:upto + 1](x)[0, channel]), backward, a step of torch.optim.Adam([x], lr), then x clamped
    to [0, 1]. The model is in evaluation mode; its weights do not change and receive no gradient. Returns a detached
    tensor.
    """
    raise NotImplementedError("ascent_28")'''

SOLUTION_28 = r'''def ascent_28(model, upto, channel, steps=60, lr=0.1, seed=0):
    """The (1, 1, 28, 28) image, in [0, 1], that makes the channel `channel` of model[:upto + 1] respond as much as
    possible (its mean over the positions), found by gradient ascent on the PIXELS.

    Start: 0.4 + 0.2 * torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(seed)), in float32. Each step:
    the loss -mean(model[:upto + 1](x)[0, channel]), backward, a step of torch.optim.Adam([x], lr), then x clamped
    to [0, 1]. The model is in evaluation mode; its weights do not change and receive no gradient. Returns a detached
    tensor.
    """
    part = model[:upto + 1]
    model.eval()
    frozen = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)                              # the weights are fixed: only the image is optimised
    x = 0.4 + 0.2 * torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(seed))
    x = x.to(DEVICE).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    try:
        for _ in range(steps):
            optimizer.zero_grad()
            loss = -part(x)[0, channel].mean()               # ascent: minimise the opposite
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x.clamp_(0.0, 1.0)                           # stay a valid image
    finally:
        for p, flag in zip(model.parameters(), frozen):
            p.requires_grad_(flag)
    return x.detach()'''

CHECK_28 = r'''if model_22 is None:
    print("⏳ Ex 21.28 : il faut d'abord le modèle entraîné de 21.22 (model_22).")
else:
    with checked("21.28"):
        before_28 = copy.deepcopy(model_22.state_dict())
        grads_28 = [None if p.grad is None else p.grad.clone() for p in model_22.parameters()]
        img_28 = ascent_28(model_22, 2, 0)
        if returned("21.28", "ascent_28", img_28):
            ok_28 = isinstance(img_28, torch.Tensor) and tuple(img_28.shape) == (1, 1, 28, 28)
            verdict("21.28", ok_28, "ascent_28 rend une image (1, 1, 28, 28).",
                    f"ascent_28 doit rendre un tenseur (1, 1, 28, 28), reçu {type(img_28).__name__} {tuple(np.shape(img_28))}.")
            if ok_28:
                verdict("21.28", float(img_28.min()) >= 0 and float(img_28.max()) <= 1 and not img_28.requires_grad,
                        "l'image reste dans [0, 1] et ne garde pas son graphe (detach).",
                        f"l'image va de {float(img_28.min()):.3f} à {float(img_28.max()):.3f}"
                        f"{' et garde son graphe' if img_28.requires_grad else ''} : clamp à chaque pas, puis detach.")
                same_w_28 = all(torch.equal(before_28[k], v) for k, v in model_22.state_dict().items())
                same_g_28 = all((g is None and p.grad is None) or (g is not None and p.grad is not None and torch.equal(g, p.grad))
                                for g, p in zip(grads_28, model_22.parameters()))
                verdict("21.28", same_w_28 and same_g_28, "les poids du modèle et leurs gradients n'ont pas bougé.",
                        "les poids ou leurs gradients ont changé : seule l'image est optimisée (requires_grad_(False) le "
                        "temps du calcul, un optimiseur qui ne contient que x).")
                start_28 = 0.4 + 0.2 * torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(0))
                with torch.no_grad():
                    gain_28 = (model_22[:3](img_28.to(DEVICE))[0, 0].mean() - model_22[:3](start_28.to(DEVICE))[0, 0].mean()).item()
                    real_28 = model_22[:3](X_VAL[:500].to(DEVICE))[:, 0].mean(dim=(1, 2)).max().item()
                    found_28 = model_22[:3](img_28.to(DEVICE))[0, 0].mean().item()
                print(f"channel 0 of the second convolution: start {found_28 - gain_28:.3f} -> found {found_28:.3f} "
                      f"(best of 500 real images: {real_28:.3f})")
                verdict("21.28", found_28 > real_28, "l'image trouvée excite ce filtre plus qu'aucune des 500 images réelles.",
                        "l'image trouvée n'excite pas ce filtre plus que des images réelles : la montée de gradient a-t-elle "
                        "le bon signe (-moyenne) et le bon nombre de pas ?")
                patterns_28 = [ascent_28(model_22, 2, c) for c in range(8)] + [ascent_28(model_22, 10, c) for c in (1, 5, 7, 8)]
                fig, axes = plt.subplots(1, 12, figsize=(14, 1.8))
                titles_28 = [f"conv2 · {c}" for c in range(8)] + [CLASSES[c] for c in (1, 5, 7, 8)]
                for ax, img, title in zip(axes, patterns_28, titles_28):
                    ax.imshow(img[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                plt.show()'''

TODO_29 = r'''def fgsm_29(model, x, y, eps):
    """FGSM: x + eps * sign(gradient of the cross-entropy loss with respect to x), clamped to [0, 1].

    x: (N, 1, 28, 28) images in [0, 1], y: (N,) labels. The model is in evaluation mode; x, the weights of the model
    and their gradients are left as they were. Returns a detached tensor of the shape of x.
    """
    raise NotImplementedError("fgsm_29")


def attack_curve_29(model, X, y, eps_list, batch_size=500):
    """The accuracy of `model` on the FGSM images of (X, y), for each eps of eps_list: a list of Python floats."""
    raise NotImplementedError("attack_curve_29")'''

SOLUTION_29 = r'''def fgsm_29(model, x, y, eps):
    """FGSM: x + eps * sign(gradient of the cross-entropy loss with respect to x), clamped to [0, 1].

    x: (N, 1, 28, 28) images in [0, 1], y: (N,) labels. The model is in evaluation mode; x, the weights of the model
    and their gradients are left as they were. Returns a detached tensor of the shape of x.
    """
    model.eval()
    x = x.clone().detach().to(DEVICE).requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x), y.to(DEVICE))
    (grad,) = torch.autograd.grad(loss, x)                   # only the gradient with respect to x: the weights' .grad stay
    return (x + eps * grad.sign()).clamp(0.0, 1.0).detach()


def attack_curve_29(model, X, y, eps_list, batch_size=500):
    """The accuracy of `model` on the FGSM images of (X, y), for each eps of eps_list: a list of Python floats."""
    curve = []
    for eps in eps_list:
        correct = 0
        for i in range(0, len(X), batch_size):
            xb, yb = X[i:i + batch_size], y[i:i + batch_size]
            adv = fgsm_29(model, xb, yb, eps)
            with torch.no_grad():
                correct += (model(adv).argmax(dim=1).cpu() == yb).sum().item()
        curve.append(correct / len(X))
    return curve'''

CHECK_29 = r'''EPS_29 = [0.0, 0.02, 0.05, 0.1, 0.2]
if model_22 is None:
    print("⏳ Ex 21.29 : il faut d'abord le modèle entraîné de 21.22 (model_22).")
else:
    with checked("21.29"):
        x_29, y_29 = X_VAL[:200].clone(), Y_VAL[:200].clone()
        before_x_29 = x_29.clone()
        grads_29 = [None if p.grad is None else p.grad.clone() for p in model_22.parameters()]
        adv_29 = fgsm_29(model_22, x_29, y_29, 0.05)
        if returned("21.29", "fgsm_29", adv_29):
            ok_29 = isinstance(adv_29, torch.Tensor) and tuple(adv_29.shape) == tuple(x_29.shape)
            verdict("21.29", ok_29, "fgsm_29 rend des images de la forme de x.",
                    f"fgsm_29 doit rendre un tenseur {tuple(x_29.shape)}, reçu {type(adv_29).__name__} {tuple(np.shape(adv_29))}.")
            if ok_29:
                adv_29 = adv_29.cpu()
                x_ref_29 = x_29.clone().to(DEVICE).requires_grad_(True)
                (g_29,) = torch.autograd.grad(nn.functional.cross_entropy(model_22.eval()(x_ref_29), y_29.to(DEVICE)), x_ref_29)
                want_29 = (x_ref_29 + 0.05 * g_29.sign()).clamp(0, 1).detach().cpu()
                same("21.29", "fgsm_29(…, eps=0.05) contre la formule de la fiche", adv_29, want_29, rtol=0, atol=1e-6)
                verdict("21.29", torch.equal(x_29, before_x_29), "x n'est pas modifié.", "fgsm_29 a modifié x : travaille sur une copie.")
                same_g_29 = all((g is None and p.grad is None) or (g is not None and p.grad is not None and torch.equal(g, p.grad))
                                for g, p in zip(grads_29, model_22.parameters()))
                verdict("21.29", same_g_29, "les gradients des poids n'ont pas bougé.",
                        "les gradients des poids ont changé : seul le gradient par rapport à x sert (torch.autograd.grad).")
                same("21.29", "fgsm_29(…, eps=0) rend x", fgsm_29(model_22, x_29, y_29, 0.0).cpu(), x_29)
    with checked("21.29"):
        curve_29 = attack_curve_29(model_22, X_VAL[:1000], Y_VAL[:1000], EPS_29)
        if returned("21.29", "attack_curve_29", curve_29):
            ok_c29 = isinstance(curve_29, list) and len(curve_29) == len(EPS_29) and all(isinstance(v, float) for v in curve_29)
            verdict("21.29", ok_c29, f"attack_curve_29 rend {len(EPS_29)} accuracies.",
                    f"attack_curve_29 doit rendre une liste de {len(EPS_29)} float Python, reçu {curve_29!r:.120}.")
            if ok_c29:
                gen_29 = torch.Generator().manual_seed(29)
                noise_29 = []
                for eps in EPS_29:                 # the same budget, a random sign instead of the sign of the gradient
                    signs = torch.randint(0, 2, X_VAL[:1000].shape, generator=gen_29) * 2.0 - 1.0
                    noise_29.append(accuracy(model_22, (X_VAL[:1000] + eps * signs).clamp(0, 1), Y_VAL[:1000]))
                print("eps          :", EPS_29)
                print("FGSM         :", [round(v, 3) for v in curve_29])
                print("random signs :", [round(v, 3) for v in noise_29])
                verdict("21.29", curve_29[3] < noise_29[3] - 0.3,
                        f"à eps = 0,1, FGSM fait tomber l'accuracy à {curve_29[3]:.2f}, du bruit de même taille seulement à {noise_29[3]:.2f}.",
                        f"à eps = 0,1, FGSM donne {curve_29[3]:.2f} et du bruit de même taille {noise_29[3]:.2f} : l'attaque devrait faire bien pire.")
                plt.figure(figsize=(5, 3.2))
                plt.plot(EPS_29, curve_29, "o-", label="FGSM")
                plt.plot(EPS_29, noise_29, "s--", label="bruit de même taille")
                plt.xlabel("ε")
                plt.ylabel("accuracy de validation")
                plt.legend()
                plt.show()
                adv_show_29 = fgsm_29(model_22, X_VAL[:6], Y_VAL[:6], 0.1).cpu()
                with torch.no_grad():
                    p_clean_29 = model_22(X_VAL[:6].to(DEVICE)).argmax(1).cpu()
                    p_adv_29 = model_22(adv_show_29.to(DEVICE)).argmax(1).cpu()
                fig, axes = plt.subplots(2, 6, figsize=(10, 3.8))
                for k in range(6):
                    axes[0, k].imshow(X_VAL[k, 0], cmap="gray", vmin=0, vmax=1)
                    axes[0, k].set_title(CLASSES[p_clean_29[k]], fontsize=8)
                    axes[1, k].imshow(adv_show_29[k, 0], cmap="gray", vmin=0, vmax=1)
                    axes[1, k].set_title(CLASSES[p_adv_29[k]], fontsize=8, color="red" if p_adv_29[k] != p_clean_29[k] else "black")
                for ax in axes.ravel():
                    ax.axis("off")
                plt.show()'''

CHECK_30 = r'''with checked("21.30"):
    C = mylearn.conv
    rng_30 = np.random.default_rng(30)
    x_30 = rng_30.normal(size=(2, 3, 4, 5))
    for label_30, shape_30, kw_30, bias_30 in [("stride 2", (3, 2, 3, 3), dict(stride=2), True),
                                              ("stride 2, padding 1, noyau 4", (3, 4, 4, 4), dict(stride=2, padding=1), False),
                                              ("stride (2, 1) et padding (1, 0), noyau 3 x 2", (3, 2, 3, 2), dict(stride=(2, 1), padding=(1, 0)), True),
                                              ("stride 1 (la passe arrière d'une convolution)", (3, 2, 2, 2), dict(), False)]:
        w_30 = rng_30.normal(size=shape_30)
        b_30 = rng_30.normal(size=shape_30[1]) if bias_30 else None
        out_30 = C.conv_transpose2d(x_30, w_30, b_30, **kw_30)
        if not returned("21.30", "conv_transpose2d", out_30):
            break
        want_30 = F.conv_transpose2d(torch.tensor(x_30), torch.tensor(w_30), None if b_30 is None else torch.tensor(b_30), **kw_30)
        same("21.30", f"conv_transpose2d, {label_30}, contre F.conv_transpose2d", out_30, want_30)
    refuses("21.30", "un padding de 3 pour un noyau de 3", lambda: C.conv_transpose2d(x_30, np.ones((3, 1, 3, 3)), padding=3))
    refuses("21.30", "un stride de 0", lambda: C.conv_transpose2d(x_30, np.ones((3, 1, 3, 3)), stride=0))
    refuses("21.30", "un poids dont shape[0] n'est pas C_in", lambda: C.conv_transpose2d(x_30, np.ones((2, 1, 3, 3))))'''

CHECK_31 = r'''with checked("21.31"):
    C = mylearn.conv
    rng_31 = np.random.default_rng(31)
    for label_31, x_shape_31, w_shape_31, kw_31 in [("stride 1, sans padding", (2, 3, 6, 5), (4, 3, 3, 3), dict()),
                                                     ("stride 2, padding 1", (2, 2, 7, 6), (3, 2, 3, 3), dict(stride=2, padding=1)),
                                                     ("stride 2 qui laisse des pixels hors de toute fenêtre", (1, 2, 6, 6), (2, 2, 3, 3), dict(stride=2)),
                                                     ("stride (1, 2), padding (2, 0), noyau 2 x 3", (2, 1, 5, 7), (2, 1, 2, 3), dict(stride=(1, 2), padding=(2, 0)))]:
        x_31 = torch.tensor(rng_31.normal(size=x_shape_31), requires_grad=True)
        w_31 = torch.tensor(rng_31.normal(size=w_shape_31), requires_grad=True)
        b_31 = torch.zeros(w_shape_31[0], dtype=torch.float64, requires_grad=True)
        out_31 = F.conv2d(x_31, w_31, b_31, **kw_31)
        dout_31 = torch.tensor(rng_31.normal(size=tuple(out_31.shape)))
        gx_31, gw_31, gb_31 = torch.autograd.grad(out_31, (x_31, w_31, b_31), dout_31)
        result_31 = C.conv2d_backward(dout_31.numpy(), x_31.detach().numpy(), w_31.detach().numpy(), **kw_31)
        if not returned("21.31", "conv2d_backward", result_31):
            break
        dx_31, dw_31, db_31 = pair("21.31", "conv2d_backward", result_31, names=("dx", "dW", "db"))
        same("21.31", f"dx, {label_31}, contre l'autograd de PyTorch", dx_31, gx_31)
        same("21.31", f"dW, {label_31}", dw_31, gw_31)
        same("21.31", f"db, {label_31}", db_31, gb_31)
    refuses("21.31", "un dout de la mauvaise forme", lambda: C.conv2d_backward(np.ones((1, 1, 3, 3)), np.ones((1, 1, 4, 4)), np.ones((1, 1, 3, 3))))'''

GIVEN_32 = r'''X_ALL_32, Y_ALL_32 = (torch.from_numpy(a) for a in wb.datasets.load_fashion_mnist("train", normalize=True, seed=None))
X_ALL_32 = X_ALL_32.unsqueeze(1)                                   # the 60 000 training images, (N, 1, 28, 28)
X_HIDDEN_32, Y_HIDDEN_32 = (torch.from_numpy(a) for a in wb.datasets.load_fashion_mnist("test", normalize=True, seed=None))
X_HIDDEN_32 = X_HIDDEN_32.unsqueeze(1)                             # the hidden test: do not look
MAX_PARAMS_32 = 100_000
TIME_LIMIT_32, GOAL_32, STAR_32 = (150.0, 0.90, 0.92) if FAST_MODE else (600.0, 0.92, 0.93)
print("train:", tuple(X_ALL_32.shape), "· hidden test:", tuple(X_HIDDEN_32.shape),
      f"· goal {GOAL_32:.0%} in {TIME_LIMIT_32:.0f} s at most, fewer than {MAX_PARAMS_32:_} parameters".replace("_", " "))'''

TODO_32 = r'''def train_32(X, y):
    """Train a CNN on the Fashion-MNIST images X (a float32 tensor (60000, 1, 28, 28), values in [0, 1]) and their
    labels y (int64), with PyTorch, and return the trained model: an nn.Module that maps (N, 1, 28, 28) images to
    (N, 10) logits. Fewer than 100 000 parameters; at most TIME_LIMIT_32 seconds of training."""
    raise NotImplementedError("train_32")'''

SOLUTION_32 = r'''def train_32(X, y):
    """Train a CNN on the Fashion-MNIST images X (a float32 tensor (60000, 1, 28, 28), values in [0, 1]) and their
    labels y (int64), with PyTorch, and return the trained model: an nn.Module that maps (N, 1, 28, 28) images to
    (N, 10) logits. Fewer than 100 000 parameters; at most TIME_LIMIT_32 seconds of training."""
    torch.manual_seed(32)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),     # 28 -> 14
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 14 -> 7
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * 7 * 7, 48), nn.ReLU(), nn.Linear(48, 10)).to(DEVICE)
    epochs, batch = (8, 128) if FAST_MODE else (25, 128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=epochs * math.ceil(len(X) / batch))
    generator = torch.Generator().manual_seed(32)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=generator).split(batch):
            loss = nn.functional.cross_entropy(model(X[idx].to(DEVICE)), y[idx].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model.eval()'''

CHECK_32 = r'''with checked("21.32"):
    started_32 = time.perf_counter()
    model_32 = train_32(X_ALL_32, Y_ALL_32)
    seconds_32 = time.perf_counter() - started_32
    if not isinstance(model_32, nn.Module):
        print(f"❌ Ex 21.32 : train_32 doit rendre le modèle entraîné, un nn.Module ; reçu {type(model_32).__name__}.")
    else:
        count_32 = n_params(model_32)
        ok_32 = count_32 < MAX_PARAMS_32 and seconds_32 <= TIME_LIMIT_32
        verdict("21.32", count_32 < MAX_PARAMS_32, f"{count_32:_} paramètres (moins de 100 000).".replace("_", " "),
                f"{count_32:_} paramètres : le défi en permet moins de 100 000.".replace("_", " "))
        verdict("21.32", seconds_32 <= TIME_LIMIT_32, f"entraînement en {seconds_32:.0f} s (au plus {TIME_LIMIT_32:.0f} s).",
                f"entraînement en {seconds_32:.0f} s : le défi en permet au plus {TIME_LIMIT_32:.0f} (moins d'epochs, un réseau plus léger).")
        if ok_32:
            train_acc_32 = accuracy(model_32, X_ALL_32[:10_000], Y_ALL_32[:10_000])
            print(f"accuracy on 10 000 training images: {train_acc_32:.4f}")
            ok_32 = train_acc_32 >= GOAL_32
            if not ok_32:
                print(f"❌ Ex 21.32 : moins de {GOAL_32:.0%} sur des images d'entraînement : le test caché n'est pas révélé ; "
                      "améliore d'abord l'entraînement.")
        if ok_32:
            REVEALED_32 = globals().get("REVEALED_32") or {}
            leaves_32 = [m for m in model_32.modules() if not list(m.children())]
            signature_32 = tuple((type(m).__name__, tuple(p.shape for p in m.parameters(recurse=False))) for m in leaves_32)
            if signature_32 not in REVEALED_32:                    # revealed once per architecture and per session (an honour rule)
                REVEALED_32[signature_32] = (accuracy(model_32, X_HIDDEN_32, Y_HIDDEN_32), round(seconds_32))
            test_32, time_32 = REVEALED_32[signature_32]
            print(f"hidden test: {test_32:.4f} (revealed once per session for this architecture, trained in {time_32} s; "
                  "a second run of the same architecture shows the same frozen result)")
            verdict("21.32", test_32 >= GOAL_32,
                    f"défi réussi : au moins {GOAL_32:.0%} sur les 10 000 images du test caché, avec moins de 100 000 paramètres."
                    + (f" 🌟 Palier atteint : au moins {STAR_32:.0%}." if test_32 >= STAR_32 else ""),
                    f"pas encore : il faut au moins {GOAL_32:.0%} sur le test caché.")'''


PART_A = Part("A", "Ta librairie conv.py : tailles, im2col, convolution et pooling",
              "Tu écris en NumPy, dans ta librairie, la mécanique d'une couche de convolution : les tailles et le padding "
              "(21.12), la vue « œil de mouche » (21.14), la convolution elle-même (21.15) et le pooling (21.18) ; au "
              "passage, un détecteur de jaune sur des photos (21.13), des tests paramétrés (21.16) et la chasse aux "
              "rayures de la figure 21.14 (21.17). La cellule suivante importe PyTorch et les outils des parties A à D. "
              "Ne recopie jamais un nombre à la main : chaque réponse vérifiée est calculée par ton code.",
              given=TOOLS, exercises=[
    Ex("21.12", "🔨", 1, 15, "conv_output_size et pad2d",
       "écrire la formule des tailles de sortie et le padding d'un mini-batch d'images.",
       "0A (tableaux NumPy) · ✏️ 21.3 · fiche §21.3.4, §21.3.5", tracks="R, M, C", mylearn="conv.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `conv_output_size(size, kernel_size, stride=1, padding=0, dilation=1)` et `pad2d(x, padding, value=0.0)`. **Contrat** propre à ces deux fonctions :
- `conv_output_size` rend la formule de la fiche, $\lfloor (H + 2p - d(k - 1) - 1)/s \rfloor + 1$, en **`int` Python** (pas un `numpy.int64` ni un `float`) ; elle refuse un argument qui n'est pas un entier, une taille < 1 (même si le padding donnerait une fenêtre), un noyau, un stride ou une dilatation < 1, un padding négatif, et une fenêtre qui ne tient pas dans l'entrée complétée ;
- `pad2d` complète les deux derniers axes d'un tableau `(N, C, H, W)` : `padding` vaut $p$ (les quatre côtés) ou $(p_h, p_w)$ ($p_h$ lignes en haut **et** en bas, $p_w$ colonnes à gauche **et** à droite), avec la valeur `value` ; elle refuse un tableau qui n'a pas quatre axes.

La première cellule compare six tailles à celles de PyTorch, essaie trois refus et trois paddings ; la seconde lance les tests (`-k "test_conv_output_size_ or test_pad2d_"`).""",
       check=RELOAD + CHECK_12, solution=CHECK_12,
       after=[("check", tests_cell("21.12", "mylearn.conv.conv_output_size(5, 3)", "test_conv_output_size_ or test_pad2d_")),
              ("solution", ref_check(tests_cell("21.12", "mylearn.conv.conv_output_size(5, 3)", "test_conv_output_size_ or test_pad2d_")))]),
    Ex("21.13", "🔮", 1, 10, "Prédire où une image CIFAR-10 « jaunit »",
       "prédire ce que fait le détecteur de jaune de ✏️ 21.1 sur de vraies photos, puis le vérifier par une convolution 1 × 1.",
       "✏️ 21.1 · fiche §21.2.2, §21.6", thread="CIFAR-10", tracks="C", hypothesis=True,
       body=r"""Le neurone de ✏️ 21.1 calcule en chaque pixel le score $R + G - B$ (composantes entre 0 et 1). On le promène sur les quatre photos de CIFAR-10 affichées par la cellule ci-dessous : A (un avion sur un ciel blanc), B (un avion sur un ciel bleu), C (un chevreuil dans l'herbe), D (une voiture jaune-orangé). Écris d'abord `weight_13`, les poids de ce neurone vus comme ceux d'une convolution 1 × 1 de PyTorch, de forme `(C_out, C_in, 1, 1)`. Puis prédis, **avant** d'exécuter l'expérience :
a) `highest_13` : la photo dont le score **moyen** est le plus haut (`"A"`, `"B"`, `"C"` ou `"D"`) ;
b) `negative_13` : le score moyen de la photo B est-il négatif ? (`True` ou `False`) ;
c) `peak_13` : la photo qui contient le pixel au score le plus haut des quatre (une lettre).

L'expérience applique la convolution 1 × 1 aux quatre photos (au format `(N, C, H, W)`), affiche les cartes de scores avec leur moyenne et leur maximum, et vérifie tes prédictions.

Dans tes notes : ce neurone est-il un bon détecteur de jaune ? Que faudrait-il ajouter pour qu'il cesse de répondre à l'herbe verte ou au blanc ?""",
       given=GIVEN_13, todo=TODO_13, check=CHECK_13,
       solution=SOLUTION_13 + "\n" + CHECK_13,
       after=[("code", guarded(EXPERIMENT_13, ["highest_13", "negative_13", "peak_13"],
                               "⏳ Ex 21.13 : écris d'abord tes trois prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé, la couleur que tu voyais ou "
                     "la formule du score ?")],
       record=RECORD_13,
       note="Le score $R + G - B$ ne mesure pas « jaune » mais « beaucoup de rouge et de vert, peu de bleu ». L'herbe, "
            "très verte et peu bleue, monte haut ; le blanc vaut $1 + 1 - 1 = 1$ ; le ciel bleu passe sous 0. Pour un "
            "vrai détecteur de jaune, il faudrait aussi exiger que $R$ et $G$ soient **proches** l'un de l'autre, ce "
            "qu'un neurone linéaire seul ne sait pas faire : il faut une non-linéarité et une seconde couche (✏️ 21.1, "
            "« dans ta copie »)."),

    Ex("21.14", "🔨", 2, 30, "im2col : la vue « œil de mouche »",
       "ranger chaque morceau de l'image dans une colonne, pour faire d'une convolution un seul produit matriciel.",
       "Ex 21.12 · fiche §21.3.2", tracks="M, C", mylearn="conv.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `im2col(x, kernel_size, stride=1, padding=0, dilation=1)`. **Contrat** :
- `x` a la forme `(N, C, H, W)` ; le résultat a la forme `(N, C·k_h·k_w, L)`, avec $L = H_{out} \times W_{out}$ (les tailles de `conv_output_size`, sur chaque axe) : c'est la disposition de `torch.nn.functional.unfold` ;
- la colonne `l` contient le morceau de la position de sortie `(l // W_out, l % W_out)` (ligne par ligne) ; dans une colonne, la valeur du canal `c`, de la ligne `i` et de la colonne `j` du noyau est à l'indice `c·k_h·k_w + i·k_w + j`, l'ordre de `weight.reshape(C_out, -1)` : ainsi `weight.reshape(C_out, -1) @ im2col(x, k)` calcule la convolution ;
- le padding est fait de zéros ; avec une dilatation $d$, le noyau lit un pixel tous les $d$.

La première cellule compare trois cas à `F.unfold`, puis vérifie que le produit matriciel redonne `F.conv2d` ; la seconde lance les tests (`-k "test_im2col_"`).""",
       check=RELOAD + CHECK_14, solution=CHECK_14,
       after=[("check", tests_cell("21.14", "mylearn.conv.im2col(np.ones((1, 1, 2, 2)), 1)", "test_im2col_", "ta fonction")),
              ("solution", ref_check(tests_cell("21.14", "mylearn.conv.im2col(np.ones((1, 1, 2, 2)), 1)", "test_im2col_", "ta fonction")))]),

    Ex("21.15", "🔨", 3, 90, "conv2d : stride, padding, dilatation et groupes",
       "écrire la couche de convolution de PyTorch en NumPy, avec tous ses réglages.",
       "Ex 21.14 · ✏️ 21.2 · fiche §21.3, §21.4, §21.4.1", tracks="M, C", mylearn="conv.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)`, la corrélation croisée de `F.conv2d`. **Contrat** :
- `x` a la forme `(N, C_in, H, W)`, `weight` la forme `(C_out, C_in/groups, k_h, k_w)`, `bias` la forme `(C_out,)` ou vaut `None` ; le résultat a la forme `(N, C_out, H_out, W_out)` ; le noyau n'est **pas** retourné ;
- `groups = g` coupe les canaux d'entrée et de sortie en $g$ groupes consécutifs : le groupe $j$ de sortie ne lit que le groupe $j$ d'entrée (`groups = C_in` : une convolution par canal) ;
- `padding` est un entier, un couple, `"valid"` (pas de padding) ou `"same"` (une sortie de la taille de l'entrée, avec un stride de 1 seulement) ; avec `"same"`, la bordure totale vaut $d(k - 1)$ sur chaque axe, et quand elle est impaire, la ligne et la colonne en trop vont **en bas et à droite**, comme dans PyTorch ;
- elle lève `ValueError` pour : `groups` < 1, ou qui ne divise pas `C_in` ou `C_out` ; un poids qui n'a pas quatre axes, ou dont `shape[1]` n'est pas `C_in/groups` ; un biais d'une autre forme que `(C_out,)`, même diffusable (`(1,)`) ; une autre chaîne que `"same"` et `"valid"` ; `"same"` avec un stride > 1 sur l'un des deux axes.

La première cellule compare six cas à `F.conv2d` (noyau rectangulaire, stride et padding par axe, dilatation, groupes, convolution par canal, `"same"` avec un noyau pair) et essaie quatre refus ; la seconde lance les tests (`-k "test_conv2d_ and not test_conv2d_backward_"`).""",
       check=RELOAD + CHECK_15, solution=CHECK_15,
       after=[("check", tests_cell("21.15", "mylearn.conv.conv2d(np.ones((1, 1, 2, 2)), np.ones((1, 1, 1, 1)))",
                                   "test_conv2d_ and not test_conv2d_backward_", "ta fonction")),
              ("solution", ref_check(tests_cell("21.15", "mylearn.conv.conv2d(np.ones((1, 1, 2, 2)), np.ones((1, 1, 1, 1)))",
                                                "test_conv2d_ and not test_conv2d_backward_", "ta fonction")))]),

    Ex("21.16", "🛠️", 2, 20, "Tests pytest paramétrés pour conv2d",
       "écrire des tests paramétrés qui acceptent une convolution juste et attrapent cinq versions fausses.",
       "Ex 21.15 · 0A (pytest, `parametrize`) · fiche §21.3.4, §21.3.5", tracks="C",
       body=r"""Une convolution a beaucoup de réglages, et ses bugs se cachent dans les cas qu'on n'essaie pas : un stride qui ne tombe pas juste, un padding asymétrique, un biais oublié. La cellule ci-dessous définit `conv_ok(x, weight, bias=None, stride=1, padding=0)`, une convolution juste (elle appelle PyTorch), et cinq versions fausses, `conv_bug_1` à `conv_bug_5` (lis leurs docstrings). Écris des fonctions de test pytest (des fonctions `test_...` qui utilisent `assert`) pour une fonction `conv` de ce contrat, et mets-les dans la liste `my_tests_16`. Règles :
- au moins un test **paramétré** (`@pytest.mark.parametrize`), qui essaie plusieurs strides, paddings et tailles ;
- l'oracle est `F.conv2d` de PyTorch (le fichier de test importe `torch` et `torch.nn.functional as F`, en plus de `math`, `numpy as np` et `pytest`) ;
- tes tests doivent **tous passer** sur `conv_ok`, et **au moins un** doit échouer sur chacune des cinq versions fausses.

La vérification lance vraiment pytest : sur `conv_ok`, puis sur chaque version fausse, renommée `conv`.

Dans tes notes : lequel des cinq bugs a été le plus difficile à attraper, et quel cas a fini par le faire échouer ? Pourquoi un test sur une seule taille d'image (par exemple 6 × 6 avec un stride de 2) ne suffit-il pas ?""",
       given=GIVEN_16, todo=TODO_16, check=CHECK_16, solution=SOLUTION_16 + "\n" + CHECK_16,
       note="`conv_bug_5` ne se trompe que quand le stride ne divise pas $H + 2p - k$ : un test sur une taille où il "
            "divise (6 avec un noyau 2 et un stride de 2) le laisse passer. `conv_bug_3` passe tout test où le stride "
            "vaut 1, `conv_bug_1` tout test à noyau symétrique, `conv_bug_2` tout test sans biais et `conv_bug_4` tout "
            "test sans padding : paramétrer sur plusieurs valeurs de chaque réglage, avec un noyau asymétrique et un "
            "biais, les attrape tous."),

    Ex("21.17", "🎨", 2, 20, "Reproduire la chasse aux rayures verticales (Fig. 21.14)",
       "promener un filtre sur une image bruitée avec ta convolution, puis lire et afficher où il trouve son motif.",
       "Ex 21.15 · fiche §21.3.1", thread="synthétique", tracks="C",
       body=r"""La figure 21.14 du livre promène un filtre 3 × 3 sur une image bruitée en noir (0) et blanc (1) : $+1$ sur sa colonne du milieu, $-1$ ailleurs, il cherche une courte rayure verticale blanche entourée de noir. On refait l'expérience sur l'image `IMG_17` (24 × 24 pixels, tirée au hasard) avec le filtre `K_17`.
- `scores_17` : les scores de `K_17` en chaque position de `IMG_17`, calculés avec **ta** fonction `mylearn.conv.conv2d`, sans padding (pense au format `(N, C, H, W)` : `IMG_17[None, None]`) ;
- a) `perfect_17` : le nombre de positions de score parfait (le motif exact) ;
- b) `worst_17` : le nombre de positions qui ont le score le plus bas possible de ce filtre ;
- c) `twos_17` : le nombre de positions de score 2 ;
- `show_17(img, kernel, scores)` : la figure en cinq panneaux (a) à (e), comme celle du livre : le filtre, l'image, les scores (avec une barre de couleurs), les positions de score parfait, et l'image avec un cadre autour de chaque correspondance (`plt.Rectangle`) ; finis par `plt.show()`.

Dans tes notes : deux correspondances sont voisines, l'une juste sous l'autre ; regarde l'image : quel motif les produit ? Que vaudrait le score d'une position où une rayure parfaite a **un** pixel noir de trop, en haut ?""",
       given=GIVEN_17, todo=TODO_17, check=CHECK_17, solution=SOLUTION_17 + "\n" + CHECK_17, record=RECORD_17,
       note="Les deux correspondances voisines viennent d'une rayure de **quatre** pixels : le filtre ne contraint que "
            "ses trois lignes, si bien qu'une rayure plus longue répond à deux positions successives. Une rayure "
            "parfaite dont un pixel de la colonne du milieu est noir perd 1 (le $+1$ n'est plus sur du blanc) : 2. "
            "C'est pourquoi un score de 2 signale un motif presque complet."),

    Ex("21.18", "🔨", 2, 25, "max_pool2d et avg_pool2d",
       "écrire le max pooling et le pooling moyen aux conventions de PyTorch, padding compris.",
       "Ex 21.14 · fiche §21.3.3 (max pooling), §21.9 (encadré « le pooling dans PyTorch »)", tracks="M, C", mylearn="conv.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `max_pool2d(x, kernel_size, stride=None, padding=0)` et `avg_pool2d(x, kernel_size, stride=None, padding=0)`. **Contrat** :
- `x` a la forme `(N, C, H, W)` ; chaque canal est réduit seul ; la sortie a la forme `(N, C, H_out, W_out)` (les tailles de `conv_output_size`) ; `stride=None` veut dire `stride = kernel_size` (des blocs disjoints) ;
- le padding vaut au plus `kernel_size // 2` sur chaque axe (règle de PyTorch ; au-delà, `ValueError`) ; pour le max, les cases de padding valent $-\infty$ : elles ne sont jamais le maximum ; pour la moyenne, ce sont des zéros, **comptés** dans la moyenne (`count_include_pad=True`, le défaut de PyTorch : chaque fenêtre est divisée par $k_h k_w$) ;
- quand plusieurs cases d'une fenêtre ont la même valeur maximale, seule la valeur compte.

La première cellule compare quatre réglages à `F.max_pool2d` et `F.avg_pool2d`, essaie des valeurs toutes négatives avec padding et un padding trop grand ; la seconde lance les tests (`-k "test_max_pool2d_ or test_avg_pool2d_"`).""",
       check=RELOAD + CHECK_18, solution=CHECK_18,
       after=[("check", tests_cell("21.18", "mylearn.conv.max_pool2d(np.ones((1, 1, 2, 2)), 2)", "test_max_pool2d_ or test_avg_pool2d_")),
              ("solution", ref_check(tests_cell("21.18", "mylearn.conv.max_pool2d(np.ones((1, 1, 2, 2)), 2)", "test_max_pool2d_ or test_avg_pool2d_")))]),

])

PART_B = Part("B", "Des filtres faits main, les convolutions 1D et 1 × 1, et un détecteur de visage",
              "Tu utilises maintenant les convolutions de PyTorch (`F.conv2d`, `nn.Conv1d`, `nn.Conv2d`) avec des filtres "
              "que tu choisis : flou, Sobel et netteté sur une photo (21.19), des convolutions 1D, pleine largeur et 1 × 1 "
              "(21.20) ; puis tu construis avec **ta** librairie le détecteur de visage hiérarchique du livre (21.21).",
              exercises=[
    Ex("21.19", "📦", 2, 20, "Flou, Sobel, netteté : des filtres faits main avec F.conv2d",
       "écrire trois filtres classiques du traitement d'image comme des poids de convolution, au bon format.",
       "ch. 20 (tenseurs) · Ex 21.15 · fiche §21.3.1, §21.4, §21.4.1", thread="CIFAR-10", tracks="C",
       body=r"""Trois filtres que les logiciels d'images appliquent depuis longtemps, écrits comme des couches de convolution. Les images sont des tenseurs `float64` de forme `(N, 3, H, W)`, de valeurs dans $[0, 1]$.

**Contrat**
- `blur_19(x)` : un flou « boîte » 3 × 3 de **chaque** canal (la moyenne des 9 pixels de la fenêtre), en **un** appel à `F.conv2d` avec `groups=3` ; il rend un tenseur de la forme de `x` ;
- `sobel_19(x)` : le gradient horizontal de la **luminance** $0{,}299\,R + 0{,}587\,G + 0{,}114\,B$ (`LUMA_19`) par le noyau de Sobel $\begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}$, positif là où l'image s'éclaire vers la droite, en **un** appel à `F.conv2d` avec un poids de forme `(1, 3, 3, 3)` ; il rend `(N, 1, H, W)` ;
- `sharpen_19(x)` : chaque canal passé par le noyau $\begin{pmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{pmatrix}$ (`groups=3`), puis ramené dans $[0, 1]$ ; il rend la forme de `x` ;
- les trois gardent la taille de l'image en la complétant par **réplication** des bords (`F.pad(x, (1, 1, 1, 1), mode="replicate")`), pas par des zéros, et ne modifient pas `x`.

La vérification contrôle les formes, une image uniforme (inchangée par le flou et la netteté, nulle pour Sobel), un bord vertical, et compare le flou à la moyenne des fenêtres et le Sobel à deux étapes (luminance, puis Sobel) ; puis elle affiche les trois résultats.

Dans tes notes : pourquoi la luminance suivie du Sobel tient-elle en **une seule** convolution ? Que deviendraient les bords de l'image floue avec un padding de zéros ? Que vaut la somme des poids de chacun des trois noyaux, et qu'est-ce que cela dit d'une image uniforme ?""",
       given=GIVEN_19, todo=TODO_19, check=CHECK_19, solution=SOLUTION_19 + "\n\n\n" + CHECK_19,
       note="La luminance est une combinaison linéaire des canaux, et la convolution est linéaire : appliquer le Sobel "
            "à $\\sum_c a_c x_c$ revient à appliquer $a_c \\times$ Sobel à chaque canal et à additionner, ce qu'une "
            "convolution à 3 canaux d'entrée fait en une fois. Avec des zéros autour, la moyenne des fenêtres du bord "
            "mélange l'image et du noir : un cadre sombre. Les sommes des poids : 1 pour le flou et la netteté (une "
            "image uniforme reste la même), 0 pour Sobel (elle donne 0) : c'est la « réponse » d'un filtre à une "
            "image sans motif."),

    Ex("21.20", "🔮", 2, 15, "Conv1d, noyau pleine largeur et 1×1 : prédire formes et paramètres",
       "prédire les formes et les paramètres de convolutions 1D, pleine largeur et 1 × 1, puis le vérifier.",
       "Ex 21.15 · ✏️ 21.4 · fiche §21.5, §21.6", tracks="C", hypothesis=True,
       body=r"""Trois tenseurs : `x1_20` de forme `(8, 64, 30)` (8 phrases de 30 mots, chaque mot un vecteur de 64 nombres, rangés comme `nn.Conv1d` les attend : `(N, C, L)`) ; `x2_20` de forme `(8, 1, 30, 64)` (les mêmes phrases comme des images d'un canal, 30 lignes et 64 colonnes) ; `x3_20` de forme `(8, 64, 30, 30)`. Prédis, **avant** d'exécuter quoi que ce soit (une forme entre guillemets, comme `"(1, 2)"`) :
a) `shape1_20` : la forme de `nn.Conv1d(64, 16, kernel_size=5)(x1_20)` ;
b) `params1_20` : le nombre de paramètres de cette `Conv1d` ;
c) `shape2_20` : la forme de `nn.Conv2d(1, 16, kernel_size=(5, 64))(x2_20)` (un noyau qui couvre toute la largeur) ;
d) `equal_20` : cette `Conv2d` a-t-elle exactement autant de paramètres que la `Conv1d` de a) ? (`True` ou `False`) ;
e) `params3_20` : le nombre de paramètres de `nn.Conv2d(64, 16, kernel_size=1)` ;
f) `shape3_20` : la forme de sa sortie pour `x3_20`.

L'expérience construit les trois couches, affiche leurs formes et leurs nombres de paramètres, copie les poids de la `Conv1d` dans la `Conv2d` pleine largeur pour comparer leurs sorties, et vérifie tes prédictions.""",
       given=GIVEN_20, todo=TODO_20,
       solution=r'''shape1_20, params1_20, shape2_20, equal_20, params3_20, shape3_20 = "(8, 16, 26)", 5136, "(8, 16, 26, 1)", True, 1040, "(8, 16, 30, 30)"''',
       after=[("code", guarded(EXPERIMENT_20, ["shape1_20", "params1_20", "shape2_20", "equal_20", "params3_20", "shape3_20"],
                               "⏳ Ex 21.20 : écris d'abord tes six prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : la `Conv2d` pleine largeur et la `Conv1d` calculent-elles la même chose ? Qu'est-ce "
                     "qui, dans leur rangement des données, les distingue ? Pour une convolution 1 × 1, à quelle couche du "
                     "ch. 20 ressemble le calcul fait en chaque pixel ?")],
       record=RECORD_20,
       note="La `Conv1d` et la `Conv2d` pleine largeur calculent **la même chose** : avec les mêmes poids, rangés "
            "autrement ($w_2[o, 0, u, c] = w_1[o, c, u]$), leurs sorties ne diffèrent que d'arrondis ; seule la place "
            "des 64 nombres change (des canaux pour l'une, des colonnes pour l'autre). Une convolution 1 × 1 applique "
            "en chaque pixel la même couche dense `nn.Linear(64, 16)` aux 64 canaux de ce pixel."),

    Ex("21.21", "🎨", 3, 45, "Le détecteur de visage hiérarchique (Fig. 21.18 à 21.23)",
       "construire avec ta librairie une hiérarchie de trois niveaux de filtres 3 × 3 et de poolings qui trouve un visage.",
       "Ex 21.15 · Ex 21.18 · ✏️ 21.5 · fiche §21.3.3", thread="synthétique", tracks="C",
       body=r"""Le livre trouve un visage dans une image binaire de 27 × 27 pixels avec dix petits filtres 3 × 3 sur trois niveaux. On reprend l'idée avec nos propres dessins et nos propres filtres (cellule ci-dessous, 1 pour l'encre, 0 pour le blanc) :
- **niveau 1**, `L1_21` de forme `(5, 1, 3, 3)` : des valeurs $+1$ (encre voulue) et $-1$ (blanc voulu) qui cherchent un trait horizontal (H), un trait vertical (V), un bloc plein (B), l'extrémité gauche (L) et l'extrémité droite (R) d'un trait horizontal ;
- **niveau 2**, `L2_21` de forme `(3, 5, 3, 3)` : des $+1$ là où une carte du niveau 1 doit être allumée, des 0 (« peu importe ») ailleurs ; ils combinent ces briques en œil, nez et bouche ;
- **niveau 3**, `L3_21` de forme `(2, 3, 3, 3)` : un visage de face et un visage de profil.

Écris `detect_21(image)` avec **tes** fonctions `mylearn.conv.conv2d` et `mylearn.conv.max_pool2d`. Aux niveaux 1 et 2 : une convolution avec `padding=1`, une **correspondance** là où le score est **égal** au nombre de $+1$ du filtre (une carte de 0 et de 1, en réels), puis un max pooling 3 × 3 (une case du résultat vaut 1 si son bloc contient au moins une correspondance). Au niveau 3 : une convolution sans padding, dont tu gardes les deux scores. Puis :
a) `n1_21` : le nombre de cases à 1 dans les cinq cartes du niveau 1, pour `FACE_21` ;
b) `n2_21` : le nombre de cases à 1 dans les trois cartes du niveau 2 ;
c) `face_21` : pour `FACE_21`, `"A"` (un visage de face trouvé), `"B"` (un profil) ou `"C"` (aucun visage) ;
d) `shifted_21` : la même question pour `SHIFTED_21`, le même dessin décalé d'un pixel vers la droite.

Écris aussi `show_21(image, level1, level2, scores3)`, la figure du détecteur : l'image, les cinq cartes du niveau 1, les trois du niveau 2 et les deux scores finaux (des barres) ; finis par `plt.show()`.

Dans tes notes : pourquoi un décalage d'**un** pixel change-t-il la réponse ? Combien de pixels de l'image d'origine « voit » un filtre du niveau 3 (✏️ 21.5 f) ? Qu'est-ce qu'un CNN entraîné fait de mieux que ces dix filtres dessinés à la main ?""",
       given=GIVEN_21, todo=TODO_21, check=CHECK_21, solution=SOLUTION_21 + "\n" + CHECK_21, record=RECORD_21,
       note="Le max pooling 3 × 3 regroupe les pixels par blocs **fixes** : décalé d'un pixel, un trait peut passer "
            "d'un bloc à son voisin, et le motif du niveau 2 n'est plus aligné. La hiérarchie tolère les positions "
            "**à l'intérieur** d'un bloc, pas n'importe quel décalage. Un filtre du niveau 3 dépend de 35 × 35 pixels "
            "(plus que l'image). Un CNN entraîné apprend ses filtres à partir de milliers d'exemples, avec des poids "
            "réels au lieu de $\\pm 1$, des chevauchements (stride 1, poolings de stride 2) et beaucoup plus de filtres "
            "par niveau : il tolère les décalages, les déformations et le bruit que ces règles exactes refusent."),
])


PART_C = Part("C", "Le convnet du livre sur Fashion-MNIST : entraîner, diagnostiquer, réparer, comparer",
              "Tu traduis en PyTorch le convnet MNIST du livre et tu l'entraînes sur Fashion-MNIST (21.22) ; tu lis les "
              "courbes et la matrice de confusion d'un entraînement complet (21.23), tu répares un CNN dont les formes ne "
              "passent pas (21.24), puis tu mesures l'effet de l'initialisation (21.25) et du choix entre stride et "
              "pooling (21.26). La cellule suivante prépare les données des parties C et D : des tenseurs "
              "`(N, 1, 28, 28)` de valeurs dans $[0, 1]$ ; les images de test restent cachées jusqu'au défi (21.32).",
              given=PART_C_GIVEN, exercises=[
    Ex("21.22", "📦", 3, 45, "Le convnet MNIST du livre en PyTorch, sur Fashion-MNIST",
       "traduire en PyTorch le convnet du livre, écrire sa boucle d'entraînement et l'adapter à d'autres tailles d'image.",
       "ch. 20 (la boucle d'entraînement, 20.23) · ✏️ 21.4 · fiche §21.7, §21.9", thread="Fashion-MNIST", tracks="R, M, C",
       body=r"""Le convnet du livre (fiche §21.9 et son encadré 🕰️ « de `mnist_cnn.py` à PyTorch ») : deux convolutions 3 × 3 sans padding (32 puis 64 filtres), un max pooling 2 × 2, un dropout de 0,25, l'aplatissement, une couche dense de 128 neurones, un dropout de 0,5 et une couche dense de 10 sorties, avec des ReLU après les convolutions et la première couche dense. Fashion-MNIST a la même forme que MNIST : 28 × 28 pixels, 10 classes de vêtements.

**Contrat**
- `make_convnet_22()` rend un `nn.Sequential` de **onze** couches, dans cet ordre : `Conv2d`, `ReLU`, `Conv2d`, `ReLU`, `MaxPool2d`, `Dropout`, `Flatten`, `Linear`, `ReLU`, `Dropout`, `Linear` ; il transforme des images `(N, 1, 28, 28)` en `(N, 10)` **logits** (pas de softmax : elle est dans la loss) ;
- `train_22(model, X, y, epochs=2, batch_size=128, lr=1e-3, seed=0)` entraîne `model` **en place** (il est déjà sur `DEVICE`, et chaque mini-batch y est envoyé) avec `torch.optim.Adam(lr)` et `nn.CrossEntropyLoss`, en mode entraînement ; à chaque epoch, un nouvel ordre aléatoire des exemples, tiré par `torch.randperm` avec **un** générateur `torch.Generator().manual_seed(seed)` créé avant la boucle ; il rend la liste des loss moyennes de chaque epoch (la moyenne sur les **exemples**, pondérée par la taille de chaque mini-batch), en `float` Python ;
- a) `in_features_22` : le nombre d'entrées de la première couche dense si les images faisaient 32 × 32 pixels (des photos de CIFAR-10 en gris) ;
- b) `params32_22` : le nombre de paramètres de tout le réseau pour ces images 32 × 32.

La vérification construit le réseau, contrôle sa sortie, ses couches et son nombre de paramètres, l'entraîne 3 epochs sur les 10 000 images d'entraînement (12 epochs sur 50 000 en mode complet ; une trentaine de secondes sur un CPU en mode rapide), puis mesure l'accuracy de validation (au moins 0,80 attendu en mode rapide, 0,90 en mode complet). Le modèle entraîné, `model_22`, resservira en 21.28 et 21.29.

Dans tes notes : le script du livre utilisait Adadelta avec `lr=1.0` ; pourquoi ne faut-il pas recopier ce `lr` avec Adam ? Où se trouvent l'essentiel des paramètres de ce réseau, et que changerait un pooling moyen global avant la couche dense ?""",
       todo=TODO_22, check=CHECK_22, solution=SOLUTION_22 + "\n\n\n" + CHECK_22, record=RECORD_22,
       note="Le `lr` d'Adadelta multiplie un pas déjà adapté à l'échelle des gradients, si bien que 1,0 y est la valeur "
            "par défaut ; avec Adam, `lr` **est** la taille du pas, et 1,0 ferait exploser l'entraînement (le défaut "
            "est $10^{-3}$). 98 % des paramètres sont dans la première couche dense ($9\\,216 \\times 128$) ; un "
            "pooling moyen global ramènerait les 64 cartes à 64 nombres, et cette couche à $64 \\times 128 + 128 = "
            "8\\,320$ paramètres, au prix d'une partie de l'information de position."),

    EX_23,

    Ex("21.24", "🐛", 2, 20, "Le CNN dont les formes ne passent pas",
       "réparer un CNN traduit de Keras dont les formes ne s'enchaînent pas, en lisant ce que reçoit chaque couche.",
       "Ex 21.22 · fiche §21.4.1, §21.7, encadré 🕰️ « le format des images »", thread="Fashion-MNIST", tracks="R, C",
       body=r"""Un collègue a traduit de Keras un petit CNN : le convnet du livre sans ses dropouts (32 puis 64 filtres 3 × 3, un pooling 2 × 2, une couche dense de 128 neurones, 10 sorties). Ses images arrivent au format de son notebook Keras, `(N, 28, 28, 1)` (`X_NHWC_24`), et sa classe `Net24` ne marche pas : **quatre** bugs, tous des erreurs de forme. Corrige-la **sur place**, dans la cellule, sans changer l'architecture (les nombres de filtres et de neurones restent ceux-là) ni le format des images reçues par `forward`.

La vérification fait passer deux images dans le réseau en lisant ce que reçoit chaque couche (des hooks `register_forward_pre_hook`, ch. 20), et donne ses diagnostics dans l'ordre où les bugs se cachent les uns les autres : tant qu'un point échoue, les suivants attendent (⏳). Elle s'exécute dès le premier « Tout exécuter » : les diagnostics du code fautif sont ton point de départ.

Dans tes notes : pour chaque bug, quel message d'erreur de PyTorch le révèle, et lequel ne lève **aucune** erreur avec un mini-batch d'une seule image ? Comment aurais-tu trouvé les tailles sans les calculer (fais passer un tenseur et lis `x.shape` couche par couche, ou `nn.LazyLinear`) ?""",
       given=GIVEN_24, todo=TODO_24, check=CHECK_24, solution=SOLUTION_24 + "\n\n\n" + CHECK_24,
       note="Les quatre bugs : les images Keras sont en `(N, H, W, C)` alors qu'une `Conv2d` attend `(N, C, H, W)` "
            "(`permute(0, 3, 1, 2)`) ; `conv2` doit lire les 32 canaux que produit `conv1` ; après deux convolutions "
            "sans padding (28 → 26 → 24) et le pooling, une image donne $64 \\times 12 \\times 12 = 9\\,216$ valeurs, "
            "pas $64 \\times 13 \\times 13$ ; `torch.flatten(x)` aplatit tout le mini-batch en un seul vecteur, "
            "`torch.flatten(x, 1)` (ou `nn.Flatten()`) garde l'axe des exemples. Ce dernier bug ne lève aucune erreur "
            "avec une seule image : la sortie a seulement la forme `(10,)` au lieu de `(1, 10)`."),

    Ex("21.25", "🔬", 2, 25, "Briser la symétrie : filtres identiques, aléatoires ou He",
       "mesurer ce que deviennent des filtres partis de la même valeur, comparés à des départs aléatoires.",
       "Ex 21.22 · ch. 16 (initialisation de He) · fiche §21.7.1", thread="Fashion-MNIST", tracks="C",
       body=r"""Le §21.7.1 affirme que des filtres partis de la même valeur restent identiques. On le mesure sur un petit CNN (`small_cnn_25`, 8 filtres).

**Contrat**
- `init_25(model, kind)` initialise **en place** les poids et les biais de chaque `nn.Conv2d` et `nn.Linear` du modèle : `"constant"`, tous les poids à 0,05 et tous les biais à 0 ; `"default"`, l'initialisation de PyTorch (la méthode `reset_parameters()` de chaque couche) ; `"he"`, `nn.init.kaiming_normal_(poids, nonlinearity="relu")` et les biais à 0 ; un autre nom lève `ValueError` ;
- `distinct_filters_25(weight, tol=1e-6)` rend le nombre de filtres **distincts** d'un poids de convolution `(C_out, C_in, k, k)` (un tenseur ou un tableau) : deux filtres sont les mêmes quand toutes leurs valeurs diffèrent d'au plus `tol`.

La vérification essaie tes deux fonctions, puis, pour chaque initialisation, entraîne le petit CNN une epoch sur 5 000 images avec **ton** `train_22` (21.22), compte ses filtres distincts et mesure son accuracy de validation.

Dans tes notes : pourquoi les filtres partis de 0,05 reçoivent-ils exactement les mêmes gradients à chaque pas (pense au ch. 16) ? Un seul filtre distinct suffit-il à faire mieux que le hasard ? Entre `"default"` et `"he"`, l'écart d'accuracy est-il plus grand que ce que changerait une autre graine ?""",
       given=GIVEN_25, todo=TODO_25, check=CHECK_25, solution=SOLUTION_25 + "\n\n\n" + CHECK_25,
       note="Avec des poids tous égaux, les huit cartes de la convolution sont identiques, la couche dense les voit avec "
            "les mêmes poids, donc les gradients des huit filtres sont égaux, et Adam leur applique le même pas : ils "
            "restent égaux pour toujours. Le réseau garde un seul filtre utile et apprend quand même quelque chose (bien "
            "plus que 10 % de hasard), mais nettement moins que les départs aléatoires. Entre l'initialisation par "
            "défaut et celle de He, l'écart après une epoch sur un réseau aussi peu profond est de l'ordre de ce que "
            "change une graine : He compte surtout pour les réseaux profonds sans normalisation."),

    Ex("21.26", "🔬", 3, 40, "Stride ou pooling : accuracy et temps sur CIFAR-10",
       "comparer deux façons de réduire l'image, par pooling ou par stride, en paramètres, en calcul, en temps et en accuracy.",
       "Ex 21.22 · 🧮 21.8 · fiche §21.4.2, §21.9.1 (encadré 🕰️ « après VGG16 »)", thread="CIFAR-10", tracks="C",
       body=r"""Deux façons de diviser la taille d'une image par 2 entre deux couches : un max pooling 2 × 2 après une convolution de stride 1, ou une convolution de stride 2. On compare deux petits réseaux sur des photos de CIFAR-10 (`(N, 3, 32, 32)`, valeurs dans $[0, 1]$).

**Contrat**
- `make_pool_26()` : deux convolutions 3 × 3 de 32 puis 64 filtres (padding 1, stride 1), chacune suivie d'une ReLU et d'un max pooling 2 × 2, puis `nn.Flatten()` et une couche dense vers 10 sorties ;
- `make_stride_26()` : les deux mêmes convolutions avec un stride de 2 (padding 1) et sans pooling, chacune suivie d'une ReLU, puis `nn.Flatten()` et une couche dense vers 10 sorties ;
- les deux rendent `(N, 10)` logits pour des images `(N, 3, 32, 32)`.

La vérification construit les deux réseaux, compte leurs paramètres et leurs multiplications-additions par image (par des hooks sur les couches), les entraîne avec **ton** `train_22` (2 epochs sur 5 000 images en mode rapide, 10 epochs sur 50 000 en mode complet) et mesure le temps d'entraînement et l'accuracy sur 1 000 photos de test. 🚀 En mode complet, un GPU Colab est conseillé (une vingtaine de minutes sur un CPU).

Dans tes notes : d'où vient l'écart de calcul, alors que les paramètres sont les mêmes ? L'écart d'accuracy est-il plus grand que l'écart de temps ? Pourquoi les architectures récentes (l'encadré 🕰️ du §21.9.1) préfèrent-elles le stride ?""",
       given=GIVEN_26, todo=TODO_26, check=CHECK_26, solution=SOLUTION_26 + "\n\n\n" + CHECK_26,
       note="Les poids sont les mêmes (le pooling n'en a aucun), mais le réseau à pooling calcule ses convolutions sur "
            "des images **quatre fois plus grandes** (32 × 32 puis 16 × 16, contre 16 × 16 puis 8 × 8) : environ "
            "quatre fois plus de multiplications-additions, et un entraînement plusieurs fois plus long. En échange, il "
            "regarde chaque position avant de réduire, et gagne souvent quelques points d'accuracy sur un réseau aussi "
            "petit. Les architectures récentes (ResNet, ConvNeXt) réduisent par des convolutions de stride 2 : le "
            "réseau apprend lui-même comment réduire, pour un calcul bien moindre, et la profondeur compense."),
])


PART_D = Part("D", "Regarder les filtres, tromper le réseau, et finir ta librairie",
              "Tu regardes ce que voient les filtres d'un ResNet-18 pré-entraîné (21.27) et de ton convnet (21.28), tu "
              "trompes ton convnet par une attaque FGSM (21.29), puis tu finis ta librairie : la convolution transposée "
              "(21.30) et la rétropropagation d'une convolution (21.31). Le défi (21.32) clôt le chapitre.",
              exercises=[
    Ex("21.27", "📦", 2, 30, "Filtres et cartes d'activation d'un ResNet-18 pré-entraîné",
       "afficher les filtres de la première couche d'un réseau pré-entraîné et lire ses cartes d'activation par un hook.",
       "Ex 21.22 · ch. 20 (hooks, modèles pré-entraînés) · fiche §21.9.1 (encadré 🕰️), §21.9.3", thread="CIFAR-10",
       tracks="R, M, C",
       body=r"""`torchvision.models.resnet18(weights="DEFAULT")` télécharge (une fois, environ 45 Mo) un ResNet-18 entraîné sur ImageNet, et `WEIGHTS_27.transforms()` le prétraitement qui va avec : les photos sont agrandies à 224 × 224 et normalisées comme les images d'ImageNet (`INPUT_27`, quatre photos de CIFAR-10). Sans réseau, la cellule de données affiche ⏳ et l'exercice se fait sur Colab.

**Contrat**
- `filters_27(model)` rend les 64 filtres de `model.conv1` (un poids `(64, 3, 7, 7)`) sous forme d'un tableau NumPy `(64, 7, 7, 3)`, les canaux à la fin pour `plt.imshow`, chaque filtre ramené dans $[0, 1]$ par **son** minimum et **son** maximum ;
- `maps_27(model, x, layer)` rend la sortie de la couche `layer` (un module du modèle, comme `model.layer1`) quand le modèle s'applique à `x`, en mode évaluation et sans autograd, capturée par un hook `register_forward_hook` **retiré** avant de rendre le résultat, même si le modèle lève une erreur.

La vérification contrôle les formes, l'échelle de chaque filtre, compare les cartes de `layer1` aux couches appelées une à une et vérifie que le hook est retiré ; elle affiche les 64 filtres, huit cartes de `layer1` et de `layer3` pour une photo, puis ce qu'ImageNet « voit » sur les quatre photos de CIFAR-10.

Dans tes notes : quels motifs reconnais-tu parmi les filtres de `conv1` (bords, rayures, taches de couleur) ? Pourquoi les cartes de `layer3` sont-elles plus petites et plus difficiles à relier à la photo ? Pourquoi les prédictions d'ImageNet sur ces photos sont-elles souvent fausses ou bizarres ?""",
       given=GIVEN_27, todo=TODO_27, check=CHECK_27, solution=SOLUTION_27 + "\n\n\n" + CHECK_27,
       note="Les filtres de `conv1` sont ceux qu'on trouve à la première couche de presque tous les CNN entraînés sur des "
            "photos : des bords orientés, des rayures et des taches de couleur. `layer3` vient après quatre réductions "
            "par 2 (224 → 112 → 56 → 28 → 14) : ses cartes sont petites, et chaque case résume un grand champ récepteur, d'où des réponses "
            "à des motifs plus complexes. Les photos de CIFAR-10 font 32 × 32 pixels, agrandies sept fois : bien plus "
            "floues que les photos d'ImageNet, et leurs classes (« camion », « cheval ») ne sont pas celles d'ImageNet "
            "(1 000 classes, dont plus d'une centaine de races de chiens) : le réseau répond par la classe ImageNet la plus proche, souvent "
            "fausse. C'est pourquoi on fait un *fine-tuning* plutôt que de prendre le réseau tel quel (bonus B1)."),

    Ex("21.28", "📦", 3, 45, "Ce qu'un filtre cherche : montée de gradient sur l'entrée",
       "fabriquer, par montée de gradient sur les pixels, les images qui excitent le plus des filtres et des classes de ton convnet.",
       "Ex 21.22 · Ex 21.27 · ch. 5 (montée de gradient) · fiche §21.9.2 (et son encadré 🧮)", thread="Fashion-MNIST",
       tracks="M, C",
       body=r"""La méthode du §21.9.2 sur ton convnet entraîné `model_22` (21.22) : on part d'une image presque grise et on la modifie pas à pas pour faire répondre le plus possible un filtre donné, les poids du réseau restant fixés.

**Contrat**
- `ascent_28(model, upto, channel, steps=60, lr=0.1, seed=0)` rend l'image `(1, 1, 28, 28)`, de valeurs dans $[0, 1]$, qui fait répondre le plus le canal `channel` de `model[:upto + 1]` (les couches 0 à `upto` du `nn.Sequential`), mesuré par la **moyenne** de ce canal sur ses positions ;
- départ : `0.4 + 0.2 * torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(seed))` (en `float32`) ; à chaque pas : la loss $-$moyenne, la passe arrière, un pas de `torch.optim.Adam([x], lr)`, puis l'image ramenée dans $[0, 1]$ ;
- le modèle est en mode évaluation ; ses poids ne changent pas et ne reçoivent **aucun** gradient (leur `.grad` reste ce qu'il était) ; le résultat est détaché du graphe.

La vérification lance ta fonction sur le filtre 0 de la deuxième convolution (`upto=2`, avant sa ReLU), contrôle l'image, les poids et leurs gradients, compare la réponse du filtre à celle de 500 vraies images, puis affiche les motifs des huit premiers filtres et les images qui maximisent quatre classes (`upto=10`, les logits).

Dans tes notes : les motifs des filtres ressemblent-ils à des vêtements ? Et les images des classes ? Pourquoi une image qui fait répondre la classe « Trouser » plus fort que n'importe quel vrai pantalon ne ressemble-t-elle pas forcément à un pantalon ?""",
       todo=TODO_28, check=CHECK_28, solution=SOLUTION_28 + "\n\n\n" + CHECK_28,
       note="Certains canaux de la deuxième convolution donnent des textures répétées (rayures, grilles, grains) : "
            "chaque position du filtre est poussée vers son motif ; d'autres donnent une image presque uniforme, noire "
            "ou blanche : ils répondent au fond ou à l'encre, partout. Aucun ne ressemble à un vêtement. Les images "
            "des classes gardent au mieux quelques traces de forme, mais restent étranges : la montée de "
            "gradient trouve ce qui fait monter le **score**, pas ce qui ressemble aux exemples ; sans régularisation "
            "(flou, décalages aléatoires, prior appris), elle exploite des détails que le réseau trouve décisifs et "
            "qu'aucune vraie image n'a. C'est la même faiblesse que celle des exemples adversariaux (21.29)."),

    Ex("21.29", "📦", 3, 40, "Attaque FGSM contre ton CNN Fashion-MNIST",
       "fabriquer des exemples adversariaux par FGSM et mesurer l'accuracy de ton convnet en fonction de ε.",
       "Ex 21.22 · fiche §21.10, section « Au-delà du livre : l'attaque FGSM »", thread="Fashion-MNIST", tracks="R, C",
       body=r"""L'attaque FGSM de la fiche contre ton convnet entraîné `model_22` (21.22) : un pas de taille $\varepsilon$ dans le sens du **signe** du gradient de la loss par rapport aux **pixels**.

**Contrat**
- `fgsm_29(model, x, y, eps)` rend $\mathrm{clamp}\big(\mathbf{x} + \varepsilon\,\mathrm{sign}(\nabla_{\mathbf{x}} L), 0, 1\big)$ pour des images `x` `(N, 1, 28, 28)` dans $[0, 1]$, leurs labels `y` et la cross-entropy $L$ moyenne du mini-batch ; le modèle est en mode évaluation ; `x`, les poids du modèle et **leurs gradients** restent ce qu'ils étaient ; le résultat est détaché, de la forme de `x` ;
- `attack_curve_29(model, X, y, eps_list, batch_size=500)` rend la liste des accuracies du modèle sur les images attaquées, une par valeur de `eps_list` (des `float` Python).

La vérification compare ta fonction à la formule sur 200 images de validation, contrôle `x`, les gradients des poids et le cas $\varepsilon = 0$, puis trace l'accuracy sur 1 000 images en fonction de $\varepsilon$ (de 0 à 0,2), à côté d'un bruit de même taille aux signes tirés au hasard ; elle montre six images avant et après l'attaque.

Dans tes notes : à $\varepsilon = 0{,}1$, les images attaquées te paraissent-elles encore faciles à reconnaître ? Pourquoi le bruit aléatoire de même taille fait-il si peu, quand l'attaque fait tant (pense au modèle linéaire de la fiche) ? Quelles parades aideraient ici, et à quel prix ?""",
       todo=TODO_29, check=CHECK_29, solution=SOLUTION_29 + "\n\n\n" + CHECK_29,
       note="À $\\varepsilon = 0{,}1$, un humain reconnaît encore tous les vêtements, alors que l'accuracy du réseau "
            "s'effondre. Le bruit aléatoire change le score d'un neurone de façon désordonnée : les contributions des "
            "pixels se compensent, et leur somme grandit comme $\\sqrt{n}$ ; l'attaque aligne chaque pixel sur le signe "
            "du gradient, et l'effet grandit comme $n$ (le $\\varepsilon\\sum_j \\lvert w_j\\rvert$ de la fiche). Parades : "
            "l'entraînement adversarial (entraîner sur des exemples attaqués, PGD), au prix d'un entraînement plusieurs "
            "fois plus long et d'une accuracy un peu plus basse sur les images propres ; les défenses qui masquent "
            "le gradient se contournent."),

    Ex("21.30", "🔨", 3, 50, "conv_transpose2d par insertion de zéros",
       "écrire la convolution transposée par la recette de la fiche (insérer des zéros, compléter, retourner le noyau).",
       "Ex 21.15 · ✏️ 21.6 · fiche §21.8", tracks="M, C", mylearn="conv.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `conv_transpose2d(x, weight, bias=None, stride=1, padding=0)`, la convolution transposée de `F.conv_transpose2d`. **Contrat** :
- `x` a la forme `(N, C_in, H, W)` ; `weight` a la forme `(C_in, C_out, k_h, k_w)`, la disposition des convolutions **transposées** de PyTorch (les canaux d'entrée d'abord) ; `bias` a la forme `(C_out,)` ou vaut `None` ;
- la sortie a la forme `(N, C_out, (H - 1) s_h - 2 p_h + k_h, (W - 1) s_w - 2 p_w + k_w)` ; `padding` retire $p$ lignes (ou colonnes) de **chaque** bord de la sortie, avec $0 \le p \le k - 1$ sur chaque axe ;
- elle lève `ValueError` pour : un poids qui n'a pas quatre axes ou dont `shape[0]` n'est pas `C_in` ; un biais d'une autre forme que `(C_out,)`, même diffusable ; un stride < 1 ; un padding hors de $[0, k - 1]$ sur l'un des axes (même si la sortie existerait) ; une sortie vide.

La première cellule compare quatre cas à `F.conv_transpose2d` (dont un stride par axe et un padding) et essaie trois refus ; la seconde lance les tests (`-k "test_conv_transpose2d_"`).""",
       check=RELOAD + CHECK_30, solution=CHECK_30,
       after=[("check", tests_cell("21.30", "mylearn.conv.conv_transpose2d(np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)))",
                                   "test_conv_transpose2d_", "ta fonction")),
              ("solution", ref_check(tests_cell("21.30", "mylearn.conv.conv_transpose2d(np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)))",
                                                "test_conv_transpose2d_", "ta fonction")))]),

    Ex("21.31", "🔨", 4, 100, "conv2d_backward : rétropropager à travers une convolution",
       "écrire les gradients d'une convolution par rapport à son entrée, à ses poids et à son biais.",
       "Ex 21.15 · ∂ 21.7 · ch. 18 (passe arrière d'une couche) · fiche, section « Au-delà du livre : la rétropropagation d'une convolution »",
       tracks="M, C", mylearn="conv.py",
       body=mylearn_howto() + "\n\n" + COMMON_CONTRACT + r"""

Écris `conv2d_backward(dout, x, weight, stride=1, padding=0)`, les gradients de `conv2d` (sans groupes ni dilatation) que calcule l'autograd de PyTorch. **Contrat** :
- `dout` a la forme de la sortie de la passe avant `(N, C_out, H_out, W_out)` ; `x` la forme `(N, C_in, H, W)` ; `weight` la forme `(C_out, C_in, k_h, k_w)` ; `stride` et `padding` sont ceux de la passe avant (des entiers ou des couples) ;
- elle rend le triplet `(dx, dW, db)`, de formes `(N, C_in, H, W)`, `(C_out, C_in, k_h, k_w)` et `(C_out,)` : $\partial L/\partial W$ et $\partial L/\partial b$ **somment** sur toutes les positions et toutes les images (∂ 21.7 question 7) ;
- un pixel de `x` qu'aucune fenêtre ne lit (un stride qui ne tombe pas juste) reçoit un gradient **nul** ;
- un `dout` d'une autre forme que la sortie de la passe avant lève `ValueError`, même s'il a le même nombre de valeurs ou serait diffusable.

La première cellule compare `dx`, `dW` et `db` à l'autograd de PyTorch dans quatre cas (dont un stride qui laisse des pixels hors de toute fenêtre) et essaie un `dout` de mauvaise forme ; la seconde lance les tests (`-k "test_conv2d_backward_"`).""",
       check=RELOAD + CHECK_31, solution=CHECK_31,
       after=[("check", tests_cell("21.31", "mylearn.conv.conv2d_backward(np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)))",
                                   "test_conv2d_backward_", "ta fonction")),
              ("solution", ref_check(tests_cell("21.31", "mylearn.conv.conv2d_backward(np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)), np.ones((1, 1, 1, 1)))",
                                                "test_conv2d_backward_", "ta fonction")))]),

    Ex("21.32", "🏆", 4, 120, "Défi Fashion-MNIST : au moins 92 % avec moins de 100 000 paramètres",
       "atteindre 92 % sur le test de Fashion-MNIST avec un CNN de moins de 100 000 paramètres, dans un temps d'entraînement limité.",
       "Ex 21.22 · Ex 21.23 · ch. 19 (plannings), ch. 20 (batchnorm, dropout) · fiche §21.9", thread="Fashion-MNIST",
       tracks="C",
       body=r"""Les 60 000 images d'entraînement de Fashion-MNIST (`X_ALL_32`, un tenseur `(60000, 1, 28, 28)` de valeurs dans $[0, 1]$, et leurs labels), et un **test caché** de 10 000 images. Le convnet du livre (21.22) dépasse 92 % sur la validation après une douzaine d'epochs, avec 1,2 million de paramètres ; le MLP du défi du ch. 20 plafonnait vers 90 %. Écris `train_32(X, y)`, qui entraîne un CNN avec PyTorch et rend le modèle entraîné : un `nn.Module` qui transforme des images `(N, 1, 28, 28)` en 10 logits. Tu choisis l'architecture, la régularisation, l'optimiseur, le planning, la taille des mini-batches.

Règles : **moins de 100 000 paramètres** ; un temps d'entraînement d'au plus `TIME_LIMIT_32` secondes sur ta machine (150 s en mode rapide, 10 minutes en mode complet) ; le défi est réussi avec une accuracy d'au moins `GOAL_32` sur le test caché : **92 % en mode complet**, 90 % en mode rapide, où le temps est plus court ; palier 🌟 : 93 % en mode complet, 92 % en mode rapide. 🚀 En mode complet, un GPU Colab est conseillé. Règle d'honneur : `train_32` entraîne vraiment le réseau (pas de poids préparés à l'avance), ne lit jamais `X_HIDDEN_32`, et n'utilise pas le test caché pour régler ta méthode : garde une partie des images d'entraînement comme validation pendant tes essais. La vérification ne révèle le test qu'une fois par session et par architecture (et seulement si ton réseau atteint l'objectif sur des images d'entraînement), et fige le résultat.

Dans tes notes : ta démarche (ce que tu as essayé, dans quel ordre, avec quelle validation), ce qui a le plus compté (la batchnorm, la largeur des convolutions, la taille de la couche dense, le planning, le nombre d'epochs), et où se trouvent les paramètres de ton réseau.""",
       given=GIVEN_32, todo=TODO_32, check=CHECK_32,
       solution=SOLUTION_32 + "\n\n\n" + CHECK_32.replace('with checked("21.32"):\n', "if True:\n"),
       note="La référence : deux blocs convolution 3 × 3 + batchnorm + ReLU + max pooling (16 puis 32 filtres), un "
            "dropout de 0,3, une couche dense de 48 neurones, 80 698 paramètres (dont 75 312 dans la première couche "
            "dense) ; AdamW et un planning *one-cycle*, 8 epochs en mode rapide, 25 en mode complet. Les chiffres (temps, "
            "test) et les essais de calibrage sont dans `05_solutions.md`."),
])


PARTS: list = [PART_A, PART_B, PART_C, PART_D]
NEXT_SESSION = [
    ("A", "21.12 à 21.18", "ta librairie conv.py : tailles et padding, la vue « œil de mouche » (im2col), la "
                           "convolution et le pooling, des tests paramétrés, la chasse aux rayures"),
    ("B", "21.19 à 21.21", "des filtres faits main avec PyTorch, convolutions 1D et 1 × 1, le détecteur de visage "
                           "hiérarchique"),
    ("C", "21.22 à 21.26", "le convnet MNIST du livre sur Fashion-MNIST, ses courbes, une chasse aux bugs de formes, "
                           "l'initialisation, stride ou pooling sur CIFAR-10"),
    ("D", "21.27 à 21.32", "les filtres d'un ResNet-18 pré-entraîné, la montée de gradient sur l'entrée, l'attaque "
                           "FGSM, la convolution transposée et la rétropropagation dans ta librairie, le défi "
                           "Fashion-MNIST"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 21.Q1 à 21.Q12, 21.R1 à 21.R3, 21.1 à 21.6, 21.8 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 | "
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
        title = "# 21 · Réseaux convolutifs (CNN) — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Tout exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est "
               "pas fait affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la "
               "section « Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch21_cnn/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 21`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 21 · Réseaux convolutifs (CNN) — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Calculer à la main la taille de sortie, le nombre de paramètres et le champ récepteur d'une pile "
               "de convolutions et de poolings.\n"
               "- Programmer en NumPy, dans `mylearn.conv`, la convolution, `im2col`, le pooling, la convolution "
               "transposée et la rétropropagation, et les comparer à `torch.nn.functional`.\n"
               "- Entraîner des CNN PyTorch, regarder leurs filtres et les tromper par une attaque adversariale.\n\n"
               "**Rappel express.** Une couche de convolution calcule une corrélation croisée : "
               "$y_{n,o,i,j} = b_o + \\sum_{c,u,v} W_{o,c,u,v}\\,x_{n,c,\\,s i + u,\\,s j + v}$, poids de forme "
               "$(C_{out}, C_{in}, k_h, k_w)$, $C_{out}(C_{in}k_hk_w + 1)$ paramètres. Taille de sortie : "
               "$\\lfloor (H + 2p - d(k-1) - 1)/s \\rfloor + 1$ ; convolution transposée : $(H - 1)s - 2p + k$. "
               "Champ récepteur : $r_\\ell = r_{\\ell-1} + (k_\\ell - 1)\\prod_{i<\\ell} s_i$. FGSM : "
               "$\\mathbf{x} + \\varepsilon\\,\\mathrm{sign}(\\nabla_{\\mathbf{x}}L)$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 21.12 à 21.32 (parties A à D, ta librairie `conv.py` et les CNN PyTorch) "
             "seront ajoutés à la prochaine session de génération ; relance alors `python tools/start_chapter.py 21` "
             "pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu calculer la forme de sortie et le nombre de paramètres d'une couche de convolution, avec "
               "padding, stride et dilatation ?\n"
               "2. Sais-tu expliquer le partage de poids, le champ récepteur et ce que font une convolution 1 × 1 "
               "et une convolution transposée ?\n"
               "3. Sais-tu dire comment on regarde ce que voit un filtre, et comment on fabrique un exemple "
               "adversarial ?\n\n"
               "**Pour aller plus loin** : les notes de CS231n sur les réseaux convolutifs et le guide de Dumoulin et "
               "Visin, cités dans la fiche. La suite : le ch. 22, les réseaux récurrents." + later)]


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
