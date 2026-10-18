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
          "une photo en couleur a un canal par couleur primaire": 1}'''),
        ("b", WHOLE, "64",
         r'''mistakes={"la profondeur d'un tenseur (C, H, W) est son nombre de canaux, le premier nombre": 112}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"c'est la profondeur d'un réseau : de quoi est-il fait ?": "A",
          "le nombre de bits par pixel est un autre sens, rare ici": "C",
          "les filtres de VGG16 font 3 × 3": "D"}'''),
        ("d", WHOLE, "3 * 32 * 32",
         r'''mistakes={"trois valeurs PAR PIXEL, pas trois features par image : relis l'encadré « Le livre, corrigé » du §21.2": 3,
          "n'oublie pas les trois canaux": 32 * 32,
          "32 × 32 pixels, chacun avec 3 valeurs : un produit, pas une somme": 32 + 32 + 3}'''),
        ("e", TF, "False",
         r'''mistakes={"le convnet MNIST du livre a aussi un pooling, un dropout et des couches denses : est-ce un CNN ?": True}'''),
    ]),
    Paper("21.Q2", "Un seul neurone promené sur toute l'image", [
        ("a", TF, "True",
         r'''mistakes={"relis le §21.2.3 : le neurone emporte-t-il ses poids avec lui ?": False}'''),
        ("b", WHOLE, "25",
         r'''mistakes={"5 × 5 poids, et un seul canal": 10, "le biais n'est pas demandé": 26}'''),
        ("c", TF, "False",
         r'''mistakes={"un filtre 5 × 5 a ses 25 poids, sur une image de 100 × 100 comme de 1 000 × 1 000": True}'''),
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
          "la première fenêtre couvre x₀, x₁, x₂ : 1 × 1 + 2 × 0 + 3 × (−1)": -3}'''),
        ("c", "a whole number", "int(np.dot([1, 2, 3], [-1, 0, 1]))",
         r'''mistakes={"c'est la corrélation croisée : la convolution retourne d'abord le noyau": -2}'''),
        ("d", TF, "True",
         r'''mistakes={"retourner (1, 2, 1) donne… (1, 2, 1)": False}'''),
        ("e", WHOLE, "(6 - 3 + 1) ** 2",
         r'''mistakes={"(H − k + 1) positions dans chaque direction : un produit des deux": 4,
          "une image 6 × 6 n'a pas 36 morceaux 3 × 3 entiers : le filtre doit tenir dans l'image": 36,
          "ça, c'est pour une image 5 × 5": 9}'''),
        ("f", WHOLE, "3 * 3 * 3",
         r'''mistakes={"chaque morceau a k² valeurs PAR CANAL": 9, "3 canaux de 3 × 3 valeurs : un produit": 9 + 3}'''),
    ]),
    Paper("21.Q4", "Détecteurs de motifs et hiérarchie de filtres", [
        ("a", WHOLE, "3",
         r'''mistakes={"les pixels noirs (0) ne retirent rien : seuls les blancs sous un −1 font baisser le score": 9,
          "chaque blanc sous un +1 ajoute 1 : combien de +1 y a-t-il ?": 1}'''),
        ("b", WHOLE, "3 - 1",
         r'''mistakes={"le pixel du centre est sous un −1 : il retire 1": 3, "la colonne de gauche compte 3 pixels blancs sous des +1": 1}'''),
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
         r'''mistakes={"sans padding, on perd k − 1 = 4 pixels, pas k": 7, "sans padding, la sortie est plus petite que l'entrée": 12}'''),
        ("b", WHOLE, "(7 - 1) // 2",
         r'''mistakes={"(k − 1)/2 de chaque côté, pas k − 1": 6, "un filtre 7 × 7 déborde de 3 pixels de chaque côté": 1}'''),
        ("c", WHOLE, "out_size(7, 3, s=2)",
         r'''mistakes={"⌊(7 − 3)/2⌋ + 1 : n'oublie pas le + 1, et prends le stride vertical (2)": 2}'''),
        ("d", WHOLE, "out_size(12, 3, s=3)",
         r'''mistakes={"la largeur avec le stride horizontal, 3": 5, "⌊(12 − 3)/3⌋ + 1": 3}'''),
        ("e", TF, "True",
         r'''mistakes={"avec un stride égal au côté du filtre, les fenêtres se recouvrent-elles ?": False}'''),
    ]),
    Paper("21.Q6", "Canaux d'entrée, nombre de filtres, canaux de sortie", [
        ("a", WHOLE, "6",
         r'''mistakes={"la profondeur d'un filtre est celle de son ENTRÉE": 4, "un filtre 3 × 3 a une profondeur, en plus": 3}'''),
        ("b", WHOLE, "4", r'''mistakes={"un canal de sortie par filtre": 6}'''),
        ("c", WHOLE, "4 * 6 * 3 * 3",
         r'''mistakes={"chaque filtre a 6 canaux de 3 × 3 poids": 4 * 9, "4 filtres de 6 × 3 × 3 poids": 6 * 9}'''),
        ("d", WHOLE, "4 * (6 * 3 * 3 + 1)",
         r'''mistakes={"un biais par filtre, pas par poids": 4 * 6 * 9 * 2, "ajoute les 4 biais": 216}'''),
        ("e", WHOLE, "6 * 3 * 3",
         r'''mistakes={"par canal : chaque filtre n'a qu'UN canal, et il y en a 6": 6 * 6 * 9,
          "un filtre 3 × 3 par canal, 6 canaux": 9}'''),
    ]),
    Paper("21.Q7", "Convolution 1D ou convolution 1×1 ?", [
        ("a", WHOLE, "3 * 50",
         r'''mistakes={"le filtre couvre toute la largeur : 50 nombres par mot": 3, "3 mots de 50 nombres : un produit": 53}'''),
        ("b", WHOLE, "20 - 3 + 1",
         r'''mistakes={"sans padding, le filtre doit tenir entier dans la phrase": 20, "n − k + 1": 17}'''),
        ("c", WHOLE, "300 * 175",
         r'''mistakes={"un poids par canal d'entrée et par filtre : un produit": 300 + 175,
          "le biais n'est pas demandé": 300 * 175 + 175}'''),
        ("d", TF, "True", r'''mistakes={"un filtre 1 × 1 lit-il les pixels voisins ?": False}'''),
        ("e", WHOLE, "28",
         r'''mistakes={"une convolution 1 × 1 garde la taille de l'image": 175, "la hauteur, pas le nombre de canaux": 300}'''),
    ]),
    Paper("21.Q8", "Couche de convolution et initialisation des filtres", [
        ("a", TF, "True",
         r'''mistakes={"si tout est symétrique au départ, les gradients des filtres sont-ils différents ?": False}'''),
        ("b", WHOLE, "64 * 3 * 3",
         r'''mistakes={"le fan-in compte toutes les valeurs reçues : 64 canaux de 3 × 3": 9, "64 canaux × 9 valeurs": 64}'''),
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
         r'''mistakes={"(H − 1) s − 2p + k, avec p = 0": 8, "la convolution transposée AGRANDIT : (H − 1) s + k": 1}'''),
        ("c", WHOLE, "(4 - 1) * 2 - 2 + 3",
         r'''mistakes={"le padding retire p cases de CHAQUE côté : − 2p": 8, "sans padding, ce serait 9": 9}'''),
        ("d", WHOLE, "3 - 1", r'''mistakes={"s − 1 zéros entre deux pixels": 3}'''),
        ("e", TF, "True",
         r'''mistakes={"relis la fin du §21.8 et sa figure : les tampons se recouvrent-ils de façon égale ?": False}'''),
        ("f", WHOLE, "3 * (3 - 1) + 1",
         r'''mistakes={"un noyau k de dilatation d couvre d(k − 1) + 1 pixels": 9, "d(k − 1) + 1 = 3 × 2 + 1": 6}'''),
    ]),
    Paper("21.Q10", "VGG16 et ce que voient ses filtres", [
        ("a", SHAPE, '"(256, 28, 28)"',
         r'''mistakes={"trois poolings ont divisé 224 par 2³": "(256, 56, 56)",
          "le format (C, H, W) : les canaux d'abord": "(28, 28, 256)",
          "le troisième bloc a 256 filtres": "(128, 28, 28)"}'''),
        ("b", WHOLE, "5", r'''mistakes={"un pooling par bloc, et VGG16 a cinq blocs": 4}'''),
        ("c", WHOLE, "224 // 2 ** 4",
         r'''mistakes={"le cinquième bloc est précédé de quatre poolings, pas cinq": 7,
          "quatre poolings ont divisé 224 par 2⁴": 28}'''),
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
         r'''decimals=2, mistakes={"chaque pixel bouge d'exactement ε : il y a 28 × 28 pixels": 0.03,
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
         r'''mistakes={"224/16 = 14 patchs dans CHAQUE direction": 14, "14 × 14, pas 16 × 16": 256}'''),
        ("c", WHOLE, "512",
         r'''mistakes={"le pooling moyen global garde une valeur PAR CANAL": 7 * 7,
          "il ne garde pas les 7 × 7 positions de chaque canal": 512 * 49}'''),
        ("d", TF, "True",
         r'''mistakes={"relis l'encadré 🕰️ du §21.9.1 : que télécharge weights=\"DEFAULT\" ?": False}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("21.R1", "Ce que changent model.train() et model.eval() pour le dropout", [
        ("a", TF, "True", r'''mistakes={"en évaluation, le dropout inversé ne coupe plus rien et ne multiplie rien": False}'''),
        ("b", "a number, 4 decimals", "1 / (1 - 0.25)",
         r'''decimals=4, mistakes={"on divise par 1 − p, on ne multiplie pas": 0.75, "on divise par 1 − p, pas par p": 4.0}'''),
        ("c", "a whole number (%)", "25",
         r'''fractional="en pourcentage : une proportion de 0,25 vaut 25 %", mistakes={"c'est la part GARDÉE": 75}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"le dropout actif coupe des neurones : peut-il améliorer la mesure ?": "B",
          "le dropout dépend du mode train ou eval": "C", "PyTorch ne vérifie pas le mode pour toi": "D"}'''),
    ]),
    Paper("21.R2", "Le gradient d'un poids utilisé à plusieurs endroits", [
        ("a", WHOLE, "1 + 2 + 3",
         r'''mistakes={"chaque y_i dépend de w avec le coefficient x_i : additionne les x_i, pas les ∂L/∂y_i seuls (qui valent 1)": 3}'''),
        ("b", WHOLE, "1 * 1 + (-1) * 2 + 2 * 3",
         r'''mistakes={"chaque contribution est (∂L/∂y_i) × x_i : n'oublie pas le signe −": 9,
          "somme des (∂L/∂y_i) × x_i, pas des ∂L/∂y_i seuls": 2}'''),
        ("c", TF, "True", r'''mistakes={"relis la fiche du ch. 18 : `+=` ou `=` dans backward ?": False}'''),
    ]),
    Paper("21.R3", "Le produit scalaire, ce que calcule un filtre en chaque position", [
        ("a", WHOLE, "int(np.dot([1, 2, -1], [3, 0, 2]))",
         r'''mistakes={"n'oublie pas le signe de −1 × 2": 5, "la somme des produits terme à terme": 6}'''),
        ("b", WHOLE, "1 * 0 + 2 * 1 + 3 * 1 + 4 * 0",
         r'''mistakes={"terme à terme, pas un produit matriciel": 10, "les cases hors diagonale (2 et 3) comptent : la seconde grille y vaut 1": 0}'''),
        ("c", TF, "True",
         r'''mistakes={"w · x = ‖w‖ ‖x‖ cos θ : quand est-il le plus grand ?": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("21.1", "Le détecteur de jaune à la main", [
        ("a", "a number, 2 decimals", "float(YELLOW @ [0.9, 0.85, 0.1])",
         r'''decimals=2, mistakes={"le bleu compte avec le poids −1 : R + G − B, pas R + G + B": 1.85}'''),
        ("b", "a number, 2 decimals", "float(YELLOW @ [1.0, 0.5, 0.0])", r'''decimals=2, mistakes={"R + G − B avec B = 0": 0.5}'''),
        ("c", "a number, 2 decimals", "float(YELLOW @ [1.0, 1.0, 1.0])",
         r'''decimals=2, mistakes={"le bleu du blanc retire 1": 2, "1 + 1 − 1": 3}'''),
        ("d", "a number, 2 decimals", "0.0", r'''decimals=2, mistakes={"un pixel noir a ses trois composantes à 0": 1}'''),
        ("e", LETTER, '"A"',
         r'''mistakes={"compare les scores de a) à d) : lequel est le plus grand ?": "B",
          "le blanc a du bleu, que le neurone retire": "C", "le noir a un score nul": "D"}'''),
        ("f", "a number, 2 decimals", "2.0",
         r'''decimals=2, mistakes={"le maximum est atteint pour R = G = 1 et B = 0": 3, "avec B = 0, pas B = 1": 1}'''),
        ("g", "a whole number", "-1",
         r'''mistakes={"le magenta n'a pas de vert : le vert doit faire baisser le score": 1, "le vert doit compter, en négatif": 0}'''),
        ("h", WHOLE, "4 * 6",
         r'''mistakes={"un score par pixel, pas par valeur (le neurone combine les trois canaux)": 4 * 6 * 3,
          "4 × 6 pixels": 10}'''),
        ("i", TF, "True",
         r'''mistakes={"un neurone qui lit les 3 canaux d'un seul pixel, avec les mêmes poids partout : relis le §21.6": False}'''),
    ]),
    Paper("21.2", "Une corrélation 5×5 par 3×3 à la main", [
        ("a", WHOLE, "Y_212.shape[1]",
         r'''mistakes={"sans padding, un filtre 3 × 3 fait perdre 2 pixels, pas 3": 2, "sans padding, la sortie rétrécit": 5}'''),
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
         r'''mistakes={"⌊(600 + 4 − 5)/2⌋ + 1 : n'oublie pas le + 1": 299}'''),
        ("b", WHOLE, "out_size(300, 3, s=3, p=2)",
         r'''mistakes={"c'est ce qu'annonce le livre : applique la formule": 100,
          "⌊·⌋ arrondit vers le bas : ⌊301/3⌋ = 100, puis + 1": 102}'''),
        ("c", WHOLE, "0",
         r'''mistakes={"essaie p = 0 : ⌊(300 − 3)/3⌋ + 1": 1, "avec p = 2, on trouve b)": 2}'''),
        ("d", WHOLE, "1",
         r'''mistakes={"avec p = 2, on trouve b), pas 100": 2, "p = 1 donne-t-il encore 100 ?": 0}'''),
        ("e", WHOLE, "8",
         r'''mistakes={"c'est ce qu'écrit le livre ; la profondeur d'un filtre est celle de son entrée": 4,
          "l'entrée du second étage est la sortie du premier, qui a 8 filtres": 1}'''),
        ("f", WHOLE, "out_size(28, 3, d=2)",
         r'''mistakes={"le noyau dilaté couvre d(k − 1) + 1 = 5 pixels, pas 3": 26, "il couvre 5 pixels, pas 6": 23}'''),
        ("g", WHOLE, "2",
         r'''mistakes={"le noyau dilaté couvre 5 pixels : il déborde de 2 de chaque côté": 1, "la bordure totale, 4, se partage en deux": 4}'''),
        ("h", WHOLE, "(100 - 1) * 3 + 3",
         r'''mistakes={"(H − 1) s + k, pas H × s + k": 303, "c'est la formule d'une convolution ordinaire : une convolution transposée agrandit": 33}'''),
        ("i", WHOLE, "(7 - 1) * 2 - 2 * 1 + 4",
         r'''mistakes={"− 2p : un de chaque côté": 15, "(H − 1) s − 2p + k = 12 − 2 + 4": 16}'''),
        ("j", WHOLE, "out_size(14, 4, s=2, p=1)",
         r'''mistakes={"vérifie le padding (+ 2p) et le + 1 : ⌊(14 + 2 − 4)/2⌋ + 1": 6, "il manque le padding (+ 2) et le + 1": 5}'''),
    ]),
    Paper("21.4", "Compter les paramètres : convolution contre couche dense", [
        ("a", WHOLE, "P_CONV1_214",
         r'''mistakes={"ajoute un biais par filtre": 32 * 9, "32 filtres de 3 × 3 poids sur un canal, plus un biais chacun": 9 + 1}'''),
        ("b", WHOLE, "P_CONV2_214",
         r'''mistakes={"chaque filtre a 32 canaux de 3 × 3 poids": 64 * 10, "ajoute les 64 biais": 64 * 32 * 9}'''),
        ("c", WHOLE, "P_FC1_214",
         r'''mistakes={"ajoute les 128 biais": 9216 * 128, "l'entrée est l'aplatissement de 64 × 12 × 12 = 9 216 valeurs": 64 * 128 + 128}'''),
        ("d", WHOLE, "P_FC2_214", r'''mistakes={"ajoute les 10 biais": 1280}'''),
        ("e", WHOLE, "P_TOTAL_214",
         r'''mistakes={"n'oublie pas la dernière couche dense : additionne a) à d)": P_TOTAL_214 - P_FC2_214,
          "ajoute les biais de chaque couche": P_TOTAL_214 - 32 - 64 - 128 - 10}'''),
        ("f", "a number (%), 1 decimal", "100 * P_FC1_214 / P_TOTAL_214",
         r'''decimals=1, mistakes={"la part de la première couche dense, pas celle des convolutions": 100 * (P_CONV1_214 + P_CONV2_214) / P_TOTAL_214}'''),
        ("g", WHOLE, "784 * 32 * 26 * 26 + 32 * 26 * 26",
         r'''mistakes={"ajoute un biais par sortie : 32 × 26 × 26": 784 * 32 * 26 * 26,
          "une couche dense relie chaque entrée (784) à chaque sortie (32 × 26 × 26)": 32 * 26 * 26}'''),
        ("h", WHOLE, "26 * 26 * 32 * 9",
         r'''mistakes={"9 multiplications par position et par filtre, sur les 26 × 26 positions": 26 * 26 * 9,
          "le nombre de positions de SORTIE, 26 × 26, pas 28 × 28": 28 * 28 * 32 * 9}'''),
    ]),
    Paper("21.5", "Champ récepteur effectif d'une pile convolution + pooling", [
        ("a", WHOLE, "receptive_field([(3, 1), (3, 1)])", r'''mistakes={"chaque couche ajoute k − 1 = 2 pixels, pas 3": 6, "3 + 2": 3}'''),
        ("b", WHOLE, "receptive_field([(3, 1)] * 3)", r'''mistakes={"1 + 2 + 2 + 2": 9}'''),
        ("c", WHOLE, "3 * 9 * 64 * 64",
         r'''mistakes={"trois couches de 64 × 64 × 3 × 3 poids": 9 * 64 * 64, "le biais n'est pas demandé": 3 * (9 * 64 + 1) * 64}'''),
        ("d", WHOLE, "7 * 7 * 64 * 64",
         r'''mistakes={"le champ récepteur de b) est 7 × 7 : une seule convolution 7 × 7": 5 * 5 * 64 * 64}'''),
        ("e", WHOLE, "receptive_field([(3, 1), (3, 3), (3, 1)])",
         r'''mistakes={"c'est le compte du livre : il oublie le débord des filtres du niveau 1": 9,
          "après le pooling, deux cases voisines sont espacées de 3 pixels : le dernier noyau ajoute (3 − 1) × 3": 7}'''),
        ("f", WHOLE, "receptive_field([(3, 1), (3, 3), (3, 1), (3, 3), (3, 1)])",
         r'''mistakes={"après le second pooling, deux cases voisines sont espacées de 9 pixels": 23,
          "le dernier noyau ajoute (3 − 1) × 9, et le pooling d'avant (3 − 1) × 3": 27}'''),
        ("g", WHOLE, "receptive_field([(5, 2), (3, 3)])",
         r'''mistakes={"c'est l'approximation du livre (6 × 6) ; le second noyau ajoute (3 − 1) × 2 au premier": 6,
          "le premier stride, 2, multiplie le débord du second noyau": 7}'''),
        ("h", WHOLE, "receptive_field([(3, 1), (3, 1), (2, 2)])",
         r'''mistakes={"le pooling 2 × 2 compte comme une couche de noyau 2": 5, "2 + 2 + 1, partant de 1": 7}'''),
    ]),
    Paper("21.6", "Convolution transposée à la main : insérer des zéros puis corréler", [
        ("a", WHOLE, "Y_216.shape[1]",
         r'''mistakes={"(H − 1) s + k = 1 × 2 + 3": 4, "les tampons se chevauchent : (H − 1) s + k = 1 × 2 + 3, pas 2 × 3": 6}'''),
        ("b", WHOLE, "2 - 1", r'''mistakes={"s − 1 zéros entre deux pixels": 2}'''),
        ("c", WHOLE, "3 - 1 - 0", r'''mistakes={"k − 1 − p, avec k = 3 et p = 0": 1, "k − 1 − p = 2": 3}'''),
        ("d", "a whole number", "int(Y_216[0, 0])",
         r'''mistakes={"seul le tampon de x[0, 0] = 2 couvre la case (0, 0)": 1}'''),
        ("e", "a whole number", "int(Y_216[1, 1])",
         r'''mistakes={"seul le tampon de x[0, 0] couvre la case (1, 1) : x[0, 0] × w[1, 1]": int(X_216.sum() * W_216[1, 1])}'''),
        ("f", "a whole number", "int(Y_216[0, 2])",
         r'''mistakes={"la case (0, 2) reçoit DEUX tampons : celui de x[0, 0] et celui de x[0, 1]": 4,
          "le tampon de x[0, 1] commence en (0, 2) : il y dépose x[0, 1] × w[0, 0]": -1}'''),
        ("g", "a whole number", "int(Y_216[2, 2])",
         r'''mistakes={"la case (2, 2) reçoit les QUATRE tampons, celui de x[1, 1] compris": int(X_216[0, 0] * W_216[2, 2] + X_216[0, 1] * W_216[2, 0] + X_216[1, 0] * W_216[0, 2])}'''),
        ("h", "a whole number", "int(Y_216.sum())",
         r'''mistakes={"chaque tampon a pour somme x[i, j] × (somme de w)": int(X_216.sum()),
          "la somme de y est (somme de x) × (somme de w)": int(W_216.sum())}'''),
        ("i", WHOLE, "Y_PAD_216.shape[1]", r'''mistakes={"− 2p : on retire une colonne de CHAQUE côté": 4}'''),
        ("j", "a whole number", "int(Y_PAD_216[1, 1])",
         r'''mistakes={"le centre de la sortie 3 × 3 est la case (2, 2) de la sortie sans padding": int(Y_216[1, 1])}'''),
    ]),
    Paper("21.8", "Fermi : paramètres, mémoire et calcul de VGG16", [
        ("a", "a number of millions, 1 decimal", "MACS_208[0] / 1e6",
         r'''decimals=1, mistakes={"224 × 224 positions, 64 filtres, 3 × 3 × 3 = 27 poids chacun": 224 * 224 * 64 * 9 / 1e6,
          "en millions": MACS_208[0] / 1e9}'''),
        ("b", "a number of billions, 2 decimals", "MACS_208[1] / 1e9",
         r'''decimals=2, mistakes={"chaque filtre a 64 canaux de 3 × 3 poids": 224 * 224 * 64 * 9 / 1e9,
          "en milliards": MACS_208[1] / 1e6}'''),
        ("c", "a number of billions, 1 decimal", "sum(MACS_208) / 1e9",
         r'''decimals=1, mistakes={"chaque pooling divise par 2 le côté de l'image des blocs suivants": sum(c_in * c_out * 224 * 224 * 9 for c_in, c_out in [(3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 256), (256, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512)]) / 1e9,
          "chaque filtre a C_in canaux de 3 × 3 poids (et le côté est divisé par 2 à chaque pooling)": sum(c * 224 * 224 * 9 for c in [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]) / 1e9}'''),
        ("d", "a number (%), 1 decimal", "100 * FC_208 / (FC_208 + sum(MACS_208))",
         r'''decimals=1, mistakes={"la part des couches denses dans le CALCUL, pas dans les paramètres": 89.4}'''),
        ("e", "a number of MB, 1 decimal", "4 * ACTS_208 / 1e6",
         r'''decimals=1, mistakes={"4 octets par valeur float32": ACTS_208 / 1e6, "en Mo de 10⁶ octets": 4 * ACTS_208 / 2 ** 20}'''),
        ("f", "a number of GB, 2 decimals", "64 * 4 * ACTS_208 / 1e9",
         r'''decimals=2, mistakes={"multiplie par les 64 images du mini-batch": 4 * ACTS_208 / 1e9,
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
# Parts A to D (exercises 21.12 to 21.32): next generation session
# ---------------------------------------------------------------------------
PARTS: list = []
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
