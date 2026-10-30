#!/usr/bin/env python
"""Build the two notebooks of chapter 25 from a single source (used by Claude).

    python tools/chapters/build_ch25.py
    python tools/run_all_notebooks.py chapitres/ch25_autoencodeurs/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch25_autoencodeurs/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
25.1 to 25.4 and 25.6 and of the guided derivation ∂ 25.7 (02_exercices.md). The derivations ∂ 25.5 and 25.8, the
estimates 🧮 25.9 and 25.10, the reflection (🗣️ 25.11, ⚖️ 25.12, 📄 25.13) and the interview are corrected with
05_solutions.md.
Parts A to C (exercises 25.14 to 25.39): dense autoencoders (one image, MNIST, the latent space, interpolation,
the bottleneck, linear autoencoder and PCA), convolutional and denoising autoencoders, and the VAE (the learner's
mylearn.vae, a PyTorch VAE, beta, anomalies, a VAE in NumPy, the denoising challenge). The long trainings of 25.34
and the multi-seed calibrations of the predictions and of the challenge come from tools/chapters/history_ch25.py
(history_ch25.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, bug_letters_mistakes, check_unique_keys, code, guarded, md,  # noqa: E402,F401
                         paper_cells, part_cells, setup_cell, write_notebook)

CHAPTER = "25"
FOLDER = "chapitres/ch25_autoencodeurs"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math


def dense(n_in, n_out):
    """Parameters of nn.Linear(n_in, n_out): weights and biases."""
    return (n_in + 1) * n_out


def conv_params(c_in, c_out, k):
    """Parameters of nn.Conv2d(c_in, c_out, k) (or ConvTranspose2d): weights and one bias per output channel."""
    return c_out * (c_in * k * k + 1)


def conv_out(h, k, s=1, p=0):
    """Height after nn.Conv2d(kernel_size=k, stride=s, padding=p)."""
    return (h + 2 * p - k) // s + 1


def tconv_out(h, k, s=1, p=0, op=0):
    """Height after nn.ConvTranspose2d(kernel_size=k, stride=s, padding=p, output_padding=op)."""
    return (h - 1) * s - 2 * p + k + op


def mlp_params(sizes):
    return sum(dense(a, b) for a, b in zip(sizes, sizes[1:]))


def gaussian_kl(mu, var):
    """KL(N(mu, var) || N(0, 1)) in nats."""
    return 0.5 * (mu ** 2 + var - 1 - math.log(var))


def kl_bits(p, q):
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q))


def letter_mix(word_a, word_b):
    """The book's letter blend: the middle letter (the earlier one when there are two)."""
    out = ""
    for a, b in zip(word_a, word_b):
        lo, hi = sorted((ord(a), ord(b)))
        out += chr((lo + hi) // 2)
    return out


ENC_DEEP = [784, 512, 256, 128, 20]
P_ENC_DEEP = mlp_params(ENC_DEEP)
P_DEC_DEEP = mlp_params(ENC_DEEP[::-1])
P_ENC_CONV = conv_params(1, 16, 3) + conv_params(16, 8, 3) + conv_params(8, 3, 3)
P_DEC_CONV = conv_params(3, 16, 3) + conv_params(16, 1, 3)
A_25_2 = dict(cx=2.0, cy=2.0, r=1.0, g=1.0)
B_25_2 = dict(cx=10.0, cy=6.0, r=4.0, g=0.4)
MIX_25_2 = {key: 0.75 * A_25_2[key] + 0.25 * B_25_2[key] for key in A_25_2}
'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "F"'
LETTERS = 'the letters, e.g. "AF"'
SHAPE = 'a shape as a string, e.g. "(3, 5, 1)"'
VECTOR = 'a vector as a string, e.g. "(1, -2)"'
WORD = 'a word as a string, e.g. "ABCD"'
WHOLE = "a whole number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("25.Q1", "Avec ou sans perte : le test de l'inversibilité", [
        ("a", LETTERS, '"AC"',
         r'''mistakes={"arrondir au degré : peut-on retrouver les décimales ?": "ABC",
          "JPEG jette de l'information, même en qualité 90": "ACE",
          "une photo en niveaux de gris : peut-on retrouver ses couleurs ?": "ACF",
          "une lettre sur deux : peut-on retrouver les lettres retirées ?": "ACD",
          "écrire en binaire change la forme d'un nombre : peut-on revenir au décimal ?": "C",
          "un fichier ZIP : retrouve-t-on le fichier exact en le décompressant ?": "A"}'''),
        ("b", TF, "True",
         r'''mistakes={"applique le test de l'inversibilité : existe-t-il une autre façon d'être sans perte ?": False}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"l'erreur de reconstruction est petite mais pas nulle : retrouve-t-on l'image exacte ?": "A"}'''),
        ("d", "a number", "2_400_000 / 300_000",
         r'''decimals=3, mistakes={"on divise la taille d'origine par la taille compressée, pas l'inverse": 0.125,
          "c'est la part des octets supprimés, pas le facteur de compression": 0.875}'''),
        ("e", TF, "False",
         r'''mistakes={"compte les fichiers courts et les fichiers longs : y a-t-il assez de fichiers courts pour tous ?": True}'''),
    ]),
    Paper("25.Q2", "Pourquoi JPEG et MP3 battent un autoencodeur généraliste", [
        ("a", LETTER, '"C"',
         r'''mistakes={"JPEG n'apprend rien : ses règles ont été écrites par des ingénieurs": "A",
          "JPEG jette de l'information : relis le §25.2.2": "B",
          "JPEG ne garde aucun catalogue d'images": "D"}'''),
        ("b", LETTER, '"C"',
         r'''mistakes={"le code a été appris sur un seul type d'images": "A",
          "simple pour toi, pas pour un code appris sur d'autres images": "B"}'''),
        ("c", TF, "False",
         r'''mistakes={"relis la première page du chapitre : qui aurait quelque chose à craindre ?": True}'''),
        ("d", LETTERS, '"AC"',
         r'''mistakes={"il manque un des deux usages : relis le §25.1": "A",
          "un seul usage : le livre en donne deux, relis le §25.1": "C",
          "chiffrer n'est pas comprimer : un autoencodeur ne cache rien": "ABC",
          "dater une image n'est pas un usage d'autoencodeur": "ACD"}'''),
        ("e", DEC2, "128_000 * 180 / 8 / 1e6",
         r'''decimals=2, mistakes={"8 bits par octet : as-tu divisé par 8 ?": 23.04,
          "3 minutes, c'est 180 secondes": 0.048,
          "un million d'octets, c'est 10⁶ octets, pas 2²⁰": 2.75}'''),
    ]),
    Paper("25.Q3", "Mélange de contenu ou mélange de paramètres ?", [
        ("a", LETTER, '"A"', r'''mistakes={"on mélange les pixels, pas ce qui décrit les objets": "B"}'''),
        ("b", WORD, 'letter_mix("LUNE", "SOIR")',
         r'''mistakes={"quand il y a deux lettres au milieu, la règle prend la plus proche du début de l'alphabet": "PRLL",
          "la lettre du milieu, pas la plus petite des deux lettres": "LOIE"}'''),
        ("c", TF, "False", r'''mistakes={"relis ce que donne le mélange des deux noms de fruits du livre": True}'''),
        ("d", LETTER, '"C"',
         r'''mistakes={"un code court peut être mal rangé : que faut-il entre des objets voisins ?": "A",
          "binaire ou non, ce qui compte est la place des codes les uns par rapport aux autres": "B",
          "« sans perte » ne dit rien de la place des codes entre eux": "D"}'''),
    ]),
    Paper("25.Q4", "Goulot, code, variables latentes : vrai ou faux", [
        ("a", TF, "False", r'''mistakes={"la cible d'un autoencodeur, d'où vient-elle ?": True}'''),
        ("b", TF, "True", r'''mistakes={"que cherche à reproduire un autoencodeur ?": False}'''),
        ("c", TF, "True", r'''mistakes={"où se trouve le goulot, et que contient-il ?": False}'''),
        ("d", LETTER, '"C"',
         r'''mistakes={"il n'y a aucun label à fournir": "A",
          "c'est le terme du livre, corrigé dans la fiche": "B",
          "il n'y a ni agent ni récompense": "D"}'''),
        ("e", TF, "True",
         r'''mistakes={"si le code est aussi grand que l'entrée, que peut faire le réseau le plus simple ?": False}'''),
    ]),
    Paper("25.Q5", "Le réseau qui ne savait redessiner qu'une image", [
        ("a", WHOLE, "2 * 4096 * 10",
         r'''mistakes={"deux couches : l'encodeur et le décodeur ont chacune leurs poids": 4096 * 10,
          "une image de 64 × 64 a 64 × 64 pixels, pas 64 + 64": 2 * 128 * 10,
          "relis la question : sans les biais": 2 * 4096 * 10 + 4096 + 10}'''),
        ("b", WHOLE, "64 * 64",
         r'''mistakes={"un biais par neurone de la couche de sortie : combien en a-t-elle ?": 10}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"les biais de la sortie ne dépendent pas de l'entrée": "A",
          "rien n'est tiré au hasard dans un réseau entraîné": "B",
          "le réseau accepte toute image de la bonne taille": "D"}'''),
        ("d", TF, "False", r'''mistakes={"relis l'histoire du tigre : que montre l'image noire ?": True}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"sur ses images d'entraînement, un réseau qui mémorise semble excellent": "A",
          "le nombre d'epochs ne dit pas ce que le réseau a appris": "B",
          "la taille du goulot ne dit pas si le réseau triche": "D"}'''),
    ]),
    Paper("25.Q6", "Profondeur ou nombre de latents : qui améliore quoi ?", [
        ("a", LETTER, '"C"',
         r'''mistakes={"moins de latents, c'est moins d'information dans le code": "A",
          "relis les figures 25.18 à 25.20": "B"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"plus de couches, c'est plus de capacité pour calculer et décoder le code": "A",
          "relis la figure 25.22": "C"}'''),
        ("c", LETTERS, '"AC"',
         r'''mistakes={"il manque une des deux conditions du livre": "A",
          "une condition manque": "C",
          "un autoencodeur n'a pas de labels": "ABC",
          "avec un learning rate nul, rien n'est appris": "ACD"}'''),
        ("d", TF, "False", r'''mistakes={"le livre parle d'une habitude, pas d'une obligation": True}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"2 latents donnent des reconstructions très floues": "A",
          "la loss reste celle de la reconstruction": "C",
          "le décodeur sert toujours": "D"}'''),
    ]),
    Paper("25.Q7", "Du bruit dans l'espace latent d'un autoencodeur classique", [
        ("a", LETTER, '"B"', r'''mistakes={"compare la taille du bruit à celle des latents": "A"}'''),
        ("b", LETTER, '"A"', r'''mistakes={"compare la taille de ce bruit à celle des latents": "B"}'''),
        ("c", LETTER, '"A"', r'''mistakes={"chaque chiffre dépend plus ou moins de chaque latent": "B"}'''),
        ("d", TF, "False", r'''mistakes={"relis ce que donnent des latents tirés au hasard": True}'''),
        ("e", LETTER, '"A"', r'''mistakes={"relis la figure 25.28": "B"}'''),
    ]),
    Paper("25.Q8", "Lire une carte latente à deux dimensions", [
        ("a", LETTER, '"A"',
         r'''mistakes={"la carte ne dit rien de la fréquence des chiffres": "B",
          "les points sont placés par l'encodeur, pas par le dessin": "C",
          "des codes voisins pour des chiffres différents, est-ce bon signe ?": "D"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"aucun exemple n'a de code dans cette zone": "A",
          "le décodeur rend toujours une image, pas forcément noire": "C"}'''),
        ("c", LETTER, '"B"', r'''mistakes={"on mélange les codes, pas les pixels": "A"}'''),
        ("d", TF, "False", r'''mistakes={"relis le §25.5.3 : que donnent certains mélanges ?": True}'''),
        ("e", VECTOR, '"(5, 3)"',
         r'''mistakes={"le milieu est la moyenne des deux points : as-tu divisé par 2 ?": "(10, 6)",
          "c'est la différence des deux codes, pas leur milieu": "(6, -6)",
          "moyenne coordonnée par coordonnée : la première avec la première": "(4, 4)"}'''),
    ]),
    Paper("25.Q9", "Autoencodeur linéaire et ACP", [
        ("a", LETTER, '"B"',
         r'''mistakes={"rien ne force les latents à être orthogonaux ni rangés par variance": "A",
          "pas n'importe lequel : celui qui minimise l'erreur de reconstruction": "C",
          "garde-t-on les directions de plus grande ou de plus petite variance ?": "D"}'''),
        ("b", TF, "False", r'''mistakes={"relis l'encadré « Le livre, corrigé » du §25.6": True}'''),
        ("c", TF, "True", r'''mistakes={"l'ACP commence par centrer les données : qui le fait dans l'autoencodeur ?": False}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"aucun modèle ne garantit une erreur nulle": "B",
          "un autoencodeur apprend sur des données": "C",
          "des axes orthogonaux classés par variance, c'est ce que fournit l'ACP": "D"}'''),
        ("e", DEC1, "0.6 * 5 + 0.8 * 0",
         r'''decimals=1, mistakes={"le produit scalaire associe chaque coordonnée de u à la même coordonnée de x": 4.0,
          "c'est la norme de x, pas sa coordonnée sur u": 5.0,
          "le code est un seul nombre, la coordonnée sur la droite ; 1,8 est une coordonnée de la projection": 1.8}'''),
    ]),
    Paper("25.Q10", "Autoencodeurs convolutifs et débruiteurs", [
        ("a", LETTER, '"B"',
         r'''mistakes={"le réseau doit apprendre à rendre l'image propre": "A",
          "si la cible est l'entrée bruitée, que va apprendre le réseau ?": "C",
          "sans bruit à l'entrée, il n'y a rien à débruiter": "D"}'''),
        ("b", TF, "False", r'''mistakes={"relis ce que le livre dit du code de son débruiteur": True}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"écrêter ne change pas la durée de l'entraînement": "B",
          "écrêter ne retire pas le bruit": "C",
          "la loss ne devient pas nulle": "D"}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"des couches denses perdent la structure de l'image": "A",
          "le dropout ne change pas la taille d'une image": "C",
          "la batchnorm ne change pas la taille d'une image": "D"}'''),
        ("e", TF, "False", r'''mistakes={"le réseau a-t-il vu des photos pendant son entraînement ?": True}'''),
    ]),
    Paper("25.Q11", "Une entrée jamais vue : un tigre chez les chiffres", [
        ("a", LETTER, '"C"',
         r'''mistakes={"le réseau ne sait reconstruire que des chiffres": "A",
          "le réseau rend toujours quelque chose de visible": "B",
          "le résultat dépend de l'entrée": "D"}'''),
        ("b", TF, "True", r'''mistakes={"un code appris sur des chiffres représente-t-il bien autre chose ?": False}'''),
        ("c", TF, "False", r'''mistakes={"une image presque vide est-elle difficile à reconstruire ?": True}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"la photo n'avait pas de cadre": "B",
          "le padding ne colore pas la sortie du décodeur": "C",
          "c'est systématique, pas un hasard": "D"}'''),
        ("e", LETTERS, '"BC"',
         r'''mistakes={"une des réponses manque": "B",
          "il manque une réponse": "C",
          "un chiffre du test MNIST vient de la même distribution que l'entraînement": "ABC"}'''),
    ]),
    Paper("25.Q12", "Les trois exigences d'un VAE et l'astuce de reparamétrisation", [
        ("a", LETTERS, '"ABC"',
         r'''mistakes={"des latents entiers n'aident pas à tirer des codes": "ABCD",
          "le livre veut des codes proches pour des entrées proches, pas un code par classe": "ABCE",
          "il manque une exigence : relis le §25.9.1": "AB",
          "il manque une des trois exigences": "BC",
          "il manque une exigence": "AC"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"un VAE ne rend pas un seul nombre par latent": "A",
          "il n'y a pas de label": "C",
          "relis la description de l'encodeur au §25.9.2": "D"}'''),
        ("c", TF, "False", r'''mistakes={"tiré tel quel, z est-il une fonction de μ et de σ ?": True}'''),
        ("d", DEC2, "1.5 + 0.5 * (-2)",
         r'''decimals=2, mistakes={"σ multiplie ε : z = μ + σ ε": -0.5,
          "ε vaut −2, pas 2": 2.5,
          "σ = 0,5 est l'écart-type : ne l'élève pas au carré": 1.0}'''),
        ("e", TF, "False", r'''mistakes={"le tirage de ε change-t-il d'un passage à l'autre ?": True}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("25.R1", "Quelles augmentations garder pour des chiffres manuscrits ?", [
        ("a", LETTERS, '"ABE"',
         r'''mistakes={"un chiffre retourné gauche-droite reste-t-il le même chiffre ?": "ABCE",
          "un 6 tourné de 180°, est-ce encore un 6 ?": "ABDE",
          "il manque une transformation plausible": "AB",
          "il en manque une": "AE",
          "une transformation plausible manque": "BE"}'''),
        ("b", TF, "False", r'''mistakes={"le test doit représenter les vraies données : faut-il le transformer ?": True}'''),
        ("c", TF, "True", r'''mistakes={"chaque epoch voit-elle les mêmes images bruitées ?": False}'''),
    ]),
    Paper("25.R2", "Résumer une séquence entière dans l'état final d'un LSTM", [
        ("a", SHAPE, '"(1, 16, 32)"',
         r'''mistakes={"h_n garde l'axe des couches en premier, même avec batch_first=True": "(16, 1, 32)",
          "h_n ne contient que le dernier pas de temps : c'est output qui les a tous": "(16, 50, 32)",
          "il manque un axe à h_n": "(16, 32)"}'''),
        ("b", DEC1, "50 * 8 / 32",
         r'''decimals=1, mistakes={"compte toutes les valeurs de la séquence : 50 pas de 8 features": 1.6,
          "le code a 32 valeurs, pas 8": 50.0}'''),
        ("c", TF, "True", r'''mistakes={"un encodeur et un décodeur peuvent être des réseaux récurrents": False}'''),
    ]),
    Paper("25.R3", "La divergence KL, ce que mesure le terme de régularisation du VAE", [
        ("a", DEC2, "kl_bits([0.5, 0.5], [0.25, 0.75])",
         r'''decimals=2, mistakes={"en bits, le logarithme est en base 2": 0.14,
          "KL(P‖Q) moyenne log(P/Q) sous P, pas sous Q": 0.19}'''),
        ("b", TF, "False", r'''mistakes={"calcule KL(Q‖P) avec les lois de a) et compare": True}'''),
        ("c", TF, "True", r'''mistakes={"chaque terme vaut p log(p / p) : que vaut log 1 ?": False}'''),
        ("d", TF, "True", r'''mistakes={"relis l'inégalité de Gibbs (ch. 6)": False}'''),
        ("e", DEC2, "kl_bits([0.5, 0.5], [0.25, 0.75]) * math.log(2)",
         r'''decimals=2, mistakes={"pour passer des bits aux nats, on multiplie par ln 2": 0.30,
          "relis l'unité demandée": 0.21}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("25.1", "Compter les paramètres d'un autoencodeur à goulot", [
        ("a", WHOLE, "dense(784, 20)",
         r'''mistakes={"un biais par neurone, en plus des poids": 784 * 20,
          "relis la question : l'encodeur a un biais par latent": dense(20, 784)}'''),
        ("b", WHOLE, "dense(20, 784)",
         r'''mistakes={"relis la question : on demande le décodeur": dense(784, 20),
          "sans oublier un biais par pixel de sortie": 784 * 20}'''),
        ("c", WHOLE, "dense(784, 20) + dense(20, 784)",
         r'''mistakes={"les biais comptent": 2 * 784 * 20}'''),
        ("d", WHOLE, "P_ENC_DEEP",
         r'''mistakes={"les biais comptent": P_ENC_DEEP - (512 + 256 + 128 + 20)}'''),
        ("e", WHOLE, "P_ENC_DEEP + P_DEC_DEEP",
         r'''mistakes={"le décodeur n'a pas les mêmes biais que l'encodeur : compte-le à part": 2 * P_ENC_DEEP,
          "les biais comptent": P_ENC_DEEP + P_DEC_DEEP - (512 + 256 + 128 + 20) - (128 + 256 + 512 + 784)}'''),
        ("f", DEC1, "784 / 20",
         r'''decimals=1, mistakes={"une image MNIST a 28 × 28 pixels": 28 / 20}'''),
        ("g", WHOLE, "dense(10_000, 20) + dense(20, 10_000)",
         r'''mistakes={"les biais comptent": 400_000,
          "le décodeur a un biais par pixel de sortie": 400_040}'''),
    ]),
    Paper("25.2", "Mélanger deux cercles : pixels contre paramètres", [
        ("a", VECTOR, '"(4, 3)"',
         r'''mistakes={"t = 0,25 : un quart du chemin de A vers B, pas trois quarts": "(8, 5)",
          "t = 0,25, pas le milieu": "(6, 4)"}'''),
        ("b", DEC2, "MIX_25_2['r']",
         r'''decimals=2, mistakes={"(1 − t) pour A, t pour B": 3.25,
          "t = 0,25, pas le milieu": 2.5}'''),
        ("c", DEC2, "MIX_25_2['g']",
         r'''decimals=2, mistakes={"(1 − t) multiplie l'intensité de A": 0.55,
          "t = 0,25, pas le milieu de l'intensité": 0.7}'''),
        ("d", DEC2, "0.75 * A_25_2['g']",
         r'''decimals=2, mistakes={"ce pixel n'est que dans A : vérifie le poids de A": 0.25,
          "ce pixel est-il aussi dans B ?": 0.85}'''),
        ("e", DEC2, "0.25 * B_25_2['g']",
         r'''decimals=2, mistakes={"ce pixel n'est que dans B : vérifie le poids de B": 0.3,
          "multiplie aussi par l'intensité de B": 0.25}'''),
        ("f", WHOLE, "2", r'''mistakes={"dans un mélange de contenu, chaque disque garde sa place": 1}'''),
        ("g", DEC1, "math.pi * 2.5 ** 2",
         r'''decimals=1, mistakes={"l'aire d'un disque est π r², pas 2 π r": 2 * math.pi * 2.5,
          "relis la question : on demande l'aire du disque mélangé": (math.pi + 16 * math.pi) / 2}'''),
        ("h", DEC1, "(math.pi * 1 ** 2 + math.pi * 4 ** 2) / 2",
         r'''decimals=1, mistakes={"relis la question : on demande la moyenne des aires": math.pi * 2.5 ** 2,
          "π r² pour chaque disque, pas 2 π r": (2 * math.pi + 8 * math.pi) / 2}'''),
    ]),
    Paper("25.3", "Suivre les formes dans l'autoencodeur convolutif", [
        ("a", SHAPE, '"(64, 16, 14, 14)"',
         r'''mistakes={"le pooling divise la hauteur et la largeur par deux": "(64, 16, 28, 28)",
          "PyTorch range les canaux avant la hauteur et la largeur": "(64, 14, 14, 16)",
          "l'axe des exemples reste en premier": "(16, 14, 14)"}'''),
        ("b", SHAPE, '"(64, 8, 7, 7)"',
         r'''mistakes={"un pooling ne change pas le nombre de canaux : d'où viennent-ils ?": "(64, 7, 7, 7)",
          "compte les filtres de la convolution qui précède ce pooling": "(64, 16, 7, 7)",
          "les canaux avant la hauteur et la largeur": "(64, 7, 7, 8)"}'''),
        ("c", SHAPE, '"(64, 3, 7, 7)"',
         r'''mistakes={"il n'y a pas de pooling après la troisième convolution": "(64, 3, 3, 3)",
          "les canaux avant la hauteur et la largeur": "(64, 7, 7, 3)"}'''),
        ("d", WHOLE, "P_ENC_CONV",
         r'''mistakes={"chaque filtre a aussi un biais": P_ENC_CONV - (16 + 8 + 3),
          "un noyau 3 × 3 a 9 poids par canal d'entrée": 16 * 1 * 3 + 16 + 8 * 16 * 3 + 8 + 3 * 8 * 3 + 3}'''),
        ("e", WHOLE, "P_DEC_CONV",
         r'''mistakes={"chaque convolution a un biais par filtre": P_DEC_CONV - (16 + 1),
          "les sur-échantillonnages n'ont pas de paramètres": P_DEC_CONV + 2}'''),
        ("f", WHOLE, "P_ENC_CONV + P_DEC_CONV",
         r'''mistakes={"les biais comptent": P_ENC_CONV + P_DEC_CONV - (16 + 8 + 3 + 16 + 1)}'''),
        ("g", DEC2, "784 / (3 * 7 * 7)",
         r'''decimals=2, mistakes={"valeurs de l'image divisées par valeurs du code, pas l'inverse": 0.19,
          "compte toutes les valeurs du code, canaux compris": 16.0}'''),
        ("h", WHOLE, "32 * 7 * 7",
         r'''mistakes={"multiplie aussi par les 32 canaux": 49,
          "il y a deux poolings": 32 * 14 * 14}'''),
    ]),
    Paper("25.4", "Retrouver 28×28 : padding et output_padding d'une convolution transposée", [
        ("a", WHOLE, "tconv_out(7, 3, 2, 1)",
         r'''mistakes={"(H − 1) s, pas H s": 7 * 2 - 2 + 3,
          "le padding retire 2p, il n'ajoute pas": 6 * 2 + 2 + 3}'''),
        ("b", WHOLE, "tconv_out(7, 3, 2, 1, 1)",
         r'''mistakes={"output_padding ajoute sa valeur à la sortie": tconv_out(7, 3, 2, 1)}'''),
        ("c", WHOLE, "tconv_out(14, 3, 2, 1, 1)",
         r'''mistakes={"n'oublie pas output_padding": tconv_out(14, 3, 2, 1),
          "(H − 1) s, pas H s": 14 * 2 - 2 + 3 + 1}'''),
        ("d", WHOLE, "tconv_out(7, 2, 2, 0)",
         r'''mistakes={"(H − 1) s, pas H s": 7 * 2 + 2}'''),
        ("e", WHOLE, "tconv_out(7, 4, 2, 1)",
         r'''mistakes={"(H − 1) s − 2p + k : vérifie le premier terme et le padding": 7 * 2 - 2 + 4}'''),
        ("f", WHOLE, "conv_out(7, 3, 2, 1)",
         r'''mistakes={"⌊(H + 2p − k)/s⌋ + 1 : vérifie le padding (des deux côtés) et le + 1": conv_out(7, 3, 2, 1) - 1}'''),
        ("g", WHOLE, "tconv_out(conv_out(7, 3, 2, 1), 3, 2, 1)",
         r'''mistakes={"(H − 1) s − 2p + k, avec H la hauteur de f)": 4 * 2 - 2 + 3,
          "sans output_padding": tconv_out(conv_out(7, 3, 2, 1), 3, 2, 1, 1)}'''),
        ("h", WHOLE, "32 * 16 * 3 * 3 + 16",
         r'''mistakes={"un biais par canal de sortie, en plus des poids": 32 * 16 * 3 * 3,
          "un biais par canal de sortie, pas d'entrée": 32 * 16 * 3 * 3 + 32}'''),
    ]),
    Paper("25.6", "Où tombent les tirages d'une gaussienne en grande dimension", [
        ("a", WHOLE, "20", r'''mistakes={"E[z_j²] = 1 pour chacune des d coordonnées : additionne-les": 1}'''),
        ("b", DEC2, "math.sqrt(20)",
         r'''decimals=2, mistakes={"c'est E‖z‖² : on demande sa racine": 20.0}'''),
        ("c", DEC3, "1 - math.exp(-1 / 2)",
         r'''decimals=3, mistakes={"1 − e^(−r²/2), pas e^(−r²/2)": math.exp(-1 / 2),
          "l'exposant est r²/2, pas r²": 1 - math.exp(-1)}'''),
        ("d", DEC3, "1 - math.exp(-9 / 2)",
         r'''decimals=3, mistakes={"r = 3, donc r² = 9": 1 - math.exp(-3 / 2),
          "1 − e^(−r²/2), pas e^(−r²/2)": math.exp(-9 / 2)}'''),
        ("e", DEC2, "math.sqrt(2 * math.log(2))",
         r'''decimals=2, mistakes={"on cherche r, pas r² : prends la racine": 2 * math.log(2),
          "ln 2 (logarithme népérien), pas log₁₀ 2": math.sqrt(2 * math.log10(2))}'''),
        ("f", WHOLE, "20 // 2",
         r'''mistakes={"diviser un tirage par 2 divise sa variance par 4 : vérifie la variance de chaque coordonnée de m": 20,
          "vérifie la variance de la somme de deux tirages indépendants": 5}'''),
        ("g", DEC2, "math.sqrt(10)",
         r'''decimals=2, mistakes={"c'est E‖m‖² : on demande sa racine": 10.0}'''),
        ("h", DEC2, "math.sqrt(512)",
         r'''decimals=2, mistakes={"la norme typique est la racine de d": 512.0}'''),
    ]),
    Paper("25.7", "La divergence KL entre N(μ, σ²) et N(0, 1)", [
        ("a", DEC3, "gaussian_kl(2, 1)",
         r'''decimals=3, mistakes={"n'oublie pas le facteur ½": 4.0}'''),
        ("b", DEC3, "gaussian_kl(0, math.e)",
         r'''decimals=3, mistakes={"n'oublie pas le facteur ½": 2 * gaussian_kl(0, math.e),
          "σ² = e, et ln σ² = 1 : ne les échange pas": 0.5 * (1 - 1 - math.e)}'''),
        ("c", DEC3, "gaussian_kl(0, 0.01)",
         r'''decimals=3, mistakes={"σ = 0,1 donne σ² = 0,01 : la formule prend la variance": gaussian_kl(0, 0.1),
          "n'oublie pas le facteur ½": 2 * gaussian_kl(0, 0.01),
          "n'oublie pas le terme − ln σ²": 0.5 * (0.01 - 1)}'''),
        ("d", DEC3, "gaussian_kl(1, 4)",
         r'''decimals=3, mistakes={"n'oublie pas le facteur ½": 2 * gaussian_kl(1, 4),
          "le logarithme est celui de σ² = 4, pas de σ": 0.5 * (1 + 4 - 1 - math.log(2))}'''),
        ("e", DEC3, "gaussian_kl(1, 1) + gaussian_kl(0, 1) + gaussian_kl(-1, 1)",
         r'''decimals=3, mistakes={"on additionne les KL des latents, on ne les moyenne pas": 1 / 3}'''),
        ("f", DEC3, "gaussian_kl(0.5, 2.5)",
         r'''decimals=3, mistakes={"n'oublie pas le facteur ½": 2 * gaussian_kl(0.5, 2.5),
          "μ² = 0,25, pas 0,5": 0.5 * (0.5 + 2.5 - 1 - math.log(2.5))}'''),
        ("g", DEC3, "0.5 * (5 - 1)",
         r'''decimals=3, mistakes={"n'oublie pas le facteur ½": 4.0,
          "dérive ½(e^s − 1 − s), pas ½ e^s": 2.5}'''),
        ("h", DEC3, "gaussian_kl(1, 4) / math.log(2)",
         r'''decimals=3, mistakes={"pour passer des nats aux bits, on divise par ln 2": gaussian_kl(1, 4) * math.log(2),
          "c'est la valeur en nats": gaussian_kl(1, 4)}'''),
    ]),
]

for _paper in PAPER:                      # a dict literal with a repeated message would silently lose one
    for _sub in _paper.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_paper.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 25.1 à 25.4 et 25.6, ∂ 25.7)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"F\"`) ; "
               "plusieurs lettres : entre guillemets (`\"AF\"`) ; une forme ou un vecteur : entre guillemets "
               "(`\"(3, 5, 1)\"`, `\"(1, -2)\"`) ; un mot : entre guillemets (`\"ABCD\"`). Arrondis comme "
               "l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore remplies affichent "
               "⏳. Les questions « dans ta copie », les dérivations ∂ 25.5 et 25.8 et la démarche de ∂ 25.7, les "
               "estimations 🧮 25.9 et 25.10, la réflexion (🗣️ 25.11, ⚖️ 25.12, 📄 25.13) et l'entretien se "
               "corrigent avec `05_solutions.md`.")

# PARTS_BEGIN
# ---------------------------------------------------------------------------
# Shared tools of parts A to C
# ---------------------------------------------------------------------------
TOOLS = r'''# Tools for the notebook exercises (parts A to C)
import copy
import math
import re
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from wb.nbtools import (StopCheck, as_numpy, checked, filled, fr, refuses, returned, run_pytest_group, same,
                        shape_text, verdict, written)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # a computer without PyTorch (Mac Intel): this chapter is done on Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : ce chapitre se fait sur Colab (00_setup/INSTALL_LOCAL.md). Les "
          "cellules qui utilisent PyTorch affichent ⏳ au lieu de s'exécuter ; les tests de ta librairie tournent "
          "sans lui : python -m pytest tests/test_ch25_vae.py -q")

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

DEVICE = cfg.device                        # "cuda" if a GPU is available, else "cpu" (wb.setup)
TEST_VAE = "tests/test_ch25_vae.py"        # the tests of your library vae.py
WORK_DIR = Path(tempfile.mkdtemp(prefix="wb_ch25_"))   # a temporary folder for the files the exercises write
FASHION_CLASSES = wb.datasets.FASHION_CLASSES
N_TRAIN = 10_000 if FAST_MODE else 60_000  # training images of the long experiments
N_FIXED = 10_000                           # the predictions (🔮) use these first 10 000 images, in both modes

# MNIST and Fashion-MNIST as float32 rows of 784 pixels in [0, 1]: a fixed order, so that the first N_FIXED
# training images are the same in both modes; 2 000 test images of each
_XM, _YM = wb.datasets.load_mnist("train", flatten=True, normalize=True, seed=None)
_XF, _YF = wb.datasets.load_fashion_mnist("train", flatten=True, normalize=True, seed=None)
_ORDER = np.random.default_rng(2500).permutation(60_000)
X_MNIST, Y_MNIST = _XM[_ORDER[:N_TRAIN]], _YM[_ORDER[:N_TRAIN]]
X_FASHION, Y_FASHION = _XF[_ORDER[:N_TRAIN]], _YF[_ORDER[:N_TRAIN]]
XM_TEST, YM_TEST = wb.datasets.load_mnist("test", n=2000, flatten=True, normalize=True, seed=0)
XF_TEST, YF_TEST = wb.datasets.load_fashion_mnist("test", n=2000, flatten=True, normalize=True, seed=0)
del _XM, _YM, _XF, _YF


def to_tensor(x):
    """A float32 torch tensor (on the CPU) from an array or a tensor."""
    return torch.as_tensor(np.asarray(as_numpy(x), dtype=np.float32))


def n_params(model):
    """The number of parameters (learned numbers) of a torch module."""
    return sum(p.numel() for p in model.parameters())


if torch is not None:
    class AutoEncoder(nn.Module):
        """An encoder and a decoder, one after the other: forward(x) = decoder(encoder(x)).
        Every autoencoder of this notebook reads images as rows of 784 pixels in [0, 1], and returns rows of 784
        pixels in [0, 1] (a convolutional one unflattens and flattens inside)."""

        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            return self.decoder(self.encoder(x))


def _batched(fn, x, batch_size=1000):
    """fn applied by batches to the rows of x, in eval mode and without a graph: a NumPy float32 array."""
    x = to_tensor(x)
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            out.append(fn(x[i:i + batch_size].to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def encode(model, x, batch_size=1000):
    """The codes model.encoder(x) of the images x (N, 784), as a NumPy array (eval mode, no graph)."""
    model.eval()
    return _batched(model.encoder, x, batch_size)


def decode(model, z, batch_size=1000):
    """The images model.decoder(z) of the codes z, as a NumPy array (N, 784) (eval mode, no graph)."""
    model.eval()
    return _batched(model.decoder, z, batch_size).reshape(len(z), -1)


def reconstruct(model, x, batch_size=1000):
    """The reconstructions model(x) of the images x (N, 784), as a NumPy array (N, 784)."""
    model.eval()
    return _batched(model, x, batch_size).reshape(len(x), -1)


def mse_per_image(a, b):
    """The mean squared error of each image: a NumPy array of N values (a and b: N images of 784 pixels)."""
    a, b = np.asarray(as_numpy(a), dtype=np.float64), np.asarray(as_numpy(b), dtype=np.float64)
    return ((a.reshape(len(a), -1) - b.reshape(len(b), -1)) ** 2).mean(axis=1)


def show_rows(rows, row_titles=None, title=None, size=1.0):
    """Draw images in rows: `rows` is a list of arrays of images (k, 784) or (k, 28, 28), one per row."""
    rows = [np.asarray(as_numpy(r), dtype=np.float32).reshape(-1, 28, 28) for r in rows]
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(size * n_cols, size * len(rows) + (0.4 if title else 0)),
                             squeeze=False)
    for i, row in enumerate(rows):
        for j in range(n_cols):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(row):
                ax.imshow(row[j], cmap="gray", vmin=0, vmax=1)
        if row_titles:
            axes[i, 0].text(-3, 14, row_titles[i], ha="right", va="center", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


print(f"tools ready · torch {torch.__version__ if torch is not None else 'absent'} · device {DEVICE} · "
      f"MNIST {X_MNIST.shape} · Fashion-MNIST {X_FASHION.shape} · {len(XM_TEST)} + {len(XF_TEST)} test images")'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="25")   # reload your saved files\n'


def ref_check(check: str) -> str:
    """The solutions version of a check cell: the tests run on the reference."""
    return check.replace('run_pytest_group(TEST_VAE, "', 'run_pytest_group(TEST_VAE, impl="ref", keyword="')


def tests_after(ex_id: str, probe: str, keyword: str) -> list:
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_VAE, "{keyword}")\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent tes fonctions.")')
    return [("check", cell), ("solution", ref_check(cell))]


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre `mon_travail/mylearn/vae.py` (créé "
            "par `python tools/start_chapter.py 25`), lis la docstring, remplace chaque `raise "
            "NotImplementedError(...)` de l'exercice par ton code et **enregistre**. NumPy et la bibliothèque "
            "standard seulement. La première cellule de vérification recharge ta librairie et l'essaie sur quelques "
            "cas ; la seconde lance les tests de l'exercice. `python -m pytest tests/test_ch25_vae.py -q` les lance "
            "tous dans un terminal.")


COMMON_VAE = ("**Contrat commun de `vae.py`** (les docstrings, plus ces précisions que vérifient les tests) : les "
              "entrées peuvent être des listes et ne sont jamais modifiées ; chaque tableau rendu est un **nouveau** "
              "tableau `float64`, jamais une vue d'une entrée (même quand il lui est égal) ; une entrée qui contient "
              "un NaN lève `ValueError`.")


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "{" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items() if letter != answer) + "}"


# ---------------------------------------------------------------------------
# Part A: dense autoencoders (25.14 to 25.23)
# ---------------------------------------------------------------------------
GIVEN_14 = r'''BOOT_14 = XF_TEST[np.flatnonzero(YF_TEST == 9)[0]]       # an ankle boot (class 9), 784 pixels in [0, 1]
SHIRT_14 = XF_TEST[np.flatnonzero(YF_TEST == 0)[0]]      # a T-shirt (class 0)
BLACK_14 = np.zeros(784, dtype=np.float32)               # an all-black image


def train_one_image_14(image, n_latent=20, bias=True, steps=500, seed=2514):
    """A 784-n_latent-784 autoencoder (Linear + ReLU, then Linear + Sigmoid), trained on ONE image only:
    `steps` steps of Adam (learning rate 0.001) on the binary cross-entropy. With bias=False, no layer has biases."""
    torch.manual_seed(seed)
    model = AutoEncoder(nn.Sequential(nn.Linear(784, n_latent, bias=bias), nn.ReLU()),
                        nn.Sequential(nn.Linear(n_latent, 784, bias=bias), nn.Sigmoid()))
    x = to_tensor(image).reshape(1, 784)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        loss = F.binary_cross_entropy(model(x), x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


show_rows([np.stack([BOOT_14, SHIRT_14, BLACK_14])], ["les trois images"], size=1.3)'''

TODO_14 = r'''black_14 = ...     # a) the black image, through the autoencoder trained on the boot: does it come out as the
                   #    boot (MSE with the boot < 0.01)? True or False
shirt_14 = ...     # b) the T-shirt, through the same autoencoder: does it come out closer to the boot than to
                   #    itself (smaller MSE with the boot than with the T-shirt)? True or False
no_bias_14 = ...   # c) the same autoencoder WITHOUT any bias (bias=False), trained on the boot: is the boot still
                   #    reconstructed (MSE < 0.001)? True or False'''

SOLUTION_14 = r'''black_14, shirt_14, no_bias_14 = False, True, True'''

EXPERIMENT_14 = r'''with checked("25.14"):
    models_14 = {"avec biais": train_one_image_14(BOOT_14), "sans biais": train_one_image_14(BOOT_14, bias=False)}
    print(f"{'':12s} {'botte → botte':>14s} {'noir → botte':>13s} {'T-shirt → botte':>16s} {'T-shirt → T-shirt':>18s} "
          f"{'code nul → botte':>17s}")
    rows_14, titles_14 = [], []
    for name_14, model_14 in models_14.items():
        out_14 = reconstruct(model_14, np.stack([BOOT_14, BLACK_14, SHIRT_14]))
        zero_code_14 = decode(model_14, np.zeros((1, 20), dtype=np.float32))[0]   # the decoder alone, code = 0
        print(f"{name_14:12s} {mse_per_image(out_14[:1], BOOT_14[None])[0]:14.5f} "
              f"{mse_per_image(out_14[1:2], BOOT_14[None])[0]:13.4f} {mse_per_image(out_14[2:], BOOT_14[None])[0]:16.4f} "
              f"{mse_per_image(out_14[2:], SHIRT_14[None])[0]:18.4f} {mse_per_image(zero_code_14[None], BOOT_14[None])[0]:17.4f}")
        rows_14.append(np.vstack([out_14, zero_code_14[None]]))
        titles_14.append(name_14)
    show_rows(rows_14, titles_14, "sorties : botte, image noire, T-shirt, puis le décodeur seul avec un code nul", size=1.3)
wb.check("25.14a", black_14)
wb.check("25.14b", shirt_14)
wb.check("25.14c", no_bias_14)'''

RECORD_14 = r'''wb.record("25.14a", False, mistakes={"regarde la colonne « noir → botte » : la sortie de l'image noire est-elle proche de la botte ?": True})
wb.record("25.14b", True, mistakes={"compare les colonnes « T-shirt → botte » et « T-shirt → T-shirt » : de quelle image la sortie est-elle la plus proche ?": False})
wb.record("25.14c", True, mistakes={"regarde la ligne « sans biais » : la botte est-elle encore reconstruite ?": False})'''

TODO_15 = r'''def make_dense_ae_15(n_latent=20):
    """The book's first autoencoder: an AutoEncoder whose encoder is nn.Sequential(nn.Linear(784, n_latent),
    nn.ReLU()) and whose decoder is nn.Sequential(nn.Linear(n_latent, 784), nn.Sigmoid())."""
    raise NotImplementedError("make_dense_ae_15")


def fit_ae_15(model, x, epochs, lr=1e-3, batch_size=128, seed=0):
    """Train `model` in place on the images x (a float32 NumPy array (N, 784) in [0, 1]) to give back its input:
    Adam(lr), the mean binary cross-entropy between model(batch) and the batch, mini-batches of `batch_size` in an
    order drawn each epoch by torch.randperm(N, generator=g), with ONE g = torch.Generator().manual_seed(seed) made
    before the first epoch. Returns the list of the mean losses of each epoch (each batch weighted by its size)."""
    raise NotImplementedError("fit_ae_15")'''

SOLUTION_15 = r'''def make_dense_ae_15(n_latent=20):
    """The book's first autoencoder: an AutoEncoder whose encoder is nn.Sequential(nn.Linear(784, n_latent),
    nn.ReLU()) and whose decoder is nn.Sequential(nn.Linear(n_latent, 784), nn.Sigmoid())."""
    return AutoEncoder(nn.Sequential(nn.Linear(784, n_latent), nn.ReLU()),
                       nn.Sequential(nn.Linear(n_latent, 784), nn.Sigmoid()))


def fit_ae_15(model, x, epochs, lr=1e-3, batch_size=128, seed=0):
    """Train `model` in place on the images x (a float32 NumPy array (N, 784) in [0, 1]) to give back its input:
    Adam(lr), the mean binary cross-entropy between model(batch) and the batch, mini-batches of `batch_size` in an
    order drawn each epoch by torch.randperm(N, generator=g), with ONE g = torch.Generator().manual_seed(seed) made
    before the first epoch. Returns the list of the mean losses of each epoch (each batch weighted by its size)."""
    device = next(model.parameters()).device
    x = to_tensor(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=g)
        total = 0.0
        for i in range(0, len(x), batch_size):
            xb = x[order[i:i + batch_size]].to(device)
            loss = F.binary_cross_entropy(model(xb), xb)        # the target is the input itself
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        losses.append(total / len(x))
    return losses'''

CHECK_15 = r'''with checked("25.15"):
    torch.manual_seed(0)
    model_15 = make_dense_ae_15(20)
    if returned("25.15", "make_dense_ae_15", model_15):
        layers_15 = [type(m).__name__ for m in model_15.encoder] + ["|"] + [type(m).__name__ for m in model_15.decoder]
        verdict("25.15", isinstance(model_15, AutoEncoder) and layers_15 == ["Linear", "ReLU", "|", "Linear", "Sigmoid"],
                "make_dense_ae_15 : un AutoEncoder (Linear, ReLU | Linear, Sigmoid).",
                f"attendu AutoEncoder(Sequential(Linear, ReLU), Sequential(Linear, Sigmoid)), reçu {layers_15}.")
        shapes_15 = [(m.in_features, m.out_features) for m in model_15.modules() if isinstance(m, nn.Linear)]
        verdict("25.15", shapes_15 == [(784, 20), (20, 784)], "les deux couches font 784 → 20 et 20 → 784.",
                f"tailles des couches : attendu [(784, 20), (20, 784)], reçu {shapes_15}.")
        out_15 = reconstruct(model_15, X_MNIST[:5])
        verdict("25.15", out_15.shape == (5, 784) and out_15.min() >= 0 and out_15.max() <= 1,
                "une sortie (5, 784) dans [0, 1].", f"la sortie doit être (5, 784) dans [0, 1], reçu {out_15.shape}.")
    torch.manual_seed(0)
    losses_15 = fit_ae_15(make_dense_ae_15(20), X_MNIST[:1024], epochs=3, seed=1)
    if returned("25.15", "fit_ae_15", losses_15):
        torch.manual_seed(0)
        again_15 = fit_ae_15(make_dense_ae_15(20), X_MNIST[:1024], epochs=3, seed=1)
        verdict("25.15", len(losses_15) == 3 and np.allclose(losses_15, again_15, rtol=1e-5)
                and losses_15[2] < losses_15[0] < 0.75,
                f"fit_ae_15 : trois losses qui baissent ({', '.join(fr(v, 3) for v in losses_15)}), les mêmes à chaque appel.",
                f"attendu une loss par epoch, qui baisse et soit la même à chaque appel, reçu {losses_15!r:.200} puis "
                f"{again_15!r:.200} (un seul générateur, créé avant la première epoch ; la cible est l'entrée).")'''

EXPERIMENT_15 = r'''with checked("25.15"):
    EPOCHS_15 = 5 if FAST_MODE else 20
    results_15 = {}
    for n_latent_15 in (20, 10, 3):
        torch.manual_seed(2515)
        model_15 = make_dense_ae_15(n_latent_15).to(DEVICE)
        start_15 = time.perf_counter()
        fit_ae_15(model_15, X_MNIST, EPOCHS_15, seed=2515)
        results_15[n_latent_15] = (reconstruct(model_15, XM_TEST), time.perf_counter() - start_15)
        print(f"{n_latent_15:2d} latents : MSE de test {mse_per_image(results_15[n_latent_15][0], XM_TEST).mean():.4f} "
              f"({results_15[n_latent_15][1]:.1f} s)")
    show_rows([XM_TEST[:10]] + [results_15[n][0][:10] for n in (20, 10, 3)],
              ["originaux", "20 latents", "10 latents", "3 latents"], f"{len(X_MNIST)} images, {EPOCHS_15} epochs")'''

GIVEN_TRAIN = r'''# From here on: the reference versions of make_dense_ae and of the training loop (your 25.15, plus a history)
def make_dense_ae(n_latent=20):
    """The autoencoder of 25.15: Linear(784, n_latent) + ReLU, then Linear(n_latent, 784) + Sigmoid."""
    return AutoEncoder(nn.Sequential(nn.Linear(784, n_latent), nn.ReLU()),
                       nn.Sequential(nn.Linear(n_latent, 784), nn.Sigmoid()))


def train_ae(model, x, epochs, lr=1e-3, batch_size=128, seed=0, loss="bce", x_val=None, verbose=False):
    """Train the autoencoder `model` in place (on DEVICE) to give back the images x (N, 784): Adam(lr), mini-batches
    in an order drawn from `seed`, the mean binary cross-entropy (loss="bce") or squared error (loss="mse").
    Returns the history: "loss" (mean of each epoch), "val_mse" (MSE of x_val after each epoch, if given),
    "seconds" (duration of each epoch)."""
    model.to(DEVICE)
    x = to_tensor(x)
    loss_fn = F.binary_cross_entropy if loss == "bce" else F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    history = {"loss": [], "val_mse": [], "seconds": []}
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total = 0.0
        order = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch_size):
            xb = x[order[i:i + batch_size]].to(DEVICE)
            batch_loss = loss_fn(model(xb), xb)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total += batch_loss.item() * len(xb)
        history["loss"].append(total / len(x))
        history["seconds"].append(time.perf_counter() - start)
        if x_val is not None:
            history["val_mse"].append(float(mse_per_image(reconstruct(model, x_val), x_val).mean()))
        if verbose:
            print(f"epoch {epoch:2d} · loss {history['loss'][-1]:.4f}" +
                  (f" · val MSE {history['val_mse'][-1]:.4f}" if x_val is not None else ""))
    return history'''

TODO_16 = r'''def make_deep_ae_16(n_latent=20, hidden=(512, 256, 128)):
    """A deep autoencoder: the encoder is Linear(784, hidden[0]) + ReLU, Linear(hidden[0], hidden[1]) + ReLU, ...,
    then Linear(hidden[-1], n_latent) WITHOUT activation (the code can be negative); the decoder is the mirror
    image: Linear(n_latent, hidden[-1]) + ReLU, ..., Linear(hidden[0], 784) + Sigmoid. Returns an AutoEncoder
    whose encoder and decoder are nn.Sequential."""
    raise NotImplementedError("make_deep_ae_16")'''

SOLUTION_16 = r'''def make_deep_ae_16(n_latent=20, hidden=(512, 256, 128)):
    """A deep autoencoder: the encoder is Linear(784, hidden[0]) + ReLU, Linear(hidden[0], hidden[1]) + ReLU, ...,
    then Linear(hidden[-1], n_latent) WITHOUT activation (the code can be negative); the decoder is the mirror
    image: Linear(n_latent, hidden[-1]) + ReLU, ..., Linear(hidden[0], 784) + Sigmoid. Returns an AutoEncoder
    whose encoder and decoder are nn.Sequential."""
    sizes = [784, *hidden, n_latent]
    encoder = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        encoder += [nn.Linear(n_in, n_out), nn.ReLU()]
    encoder.pop()                                       # no activation on the code
    back = sizes[::-1]                                  # n_latent, ..., 784
    decoder = []
    for n_in, n_out in zip(back[:-1], back[1:]):
        decoder += [nn.Linear(n_in, n_out), nn.ReLU()]
    decoder[-1] = nn.Sigmoid()                          # pixels in [0, 1]
    return AutoEncoder(nn.Sequential(*encoder), nn.Sequential(*decoder))'''

CHECK_16 = r'''with checked("25.16"):
    torch.manual_seed(0)
    model_16 = make_deep_ae_16(20)
    if returned("25.16", "make_deep_ae_16", model_16):
        def _layout_16(seq):
            return [(m.in_features, m.out_features) if isinstance(m, nn.Linear) else type(m).__name__ for m in seq]
        want_enc_16 = [(784, 512), "ReLU", (512, 256), "ReLU", (256, 128), "ReLU", (128, 20)]
        want_dec_16 = [(20, 128), "ReLU", (128, 256), "ReLU", (256, 512), "ReLU", (512, 784), "Sigmoid"]
        verdict("25.16", _layout_16(model_16.encoder) == want_enc_16, "l'encodeur : 784 → 512 → 256 → 128 → 20, sans activation sur le code.",
                f"encodeur attendu {want_enc_16}, reçu {_layout_16(model_16.encoder)}.")
        verdict("25.16", _layout_16(model_16.decoder) == want_dec_16, "le décodeur : son miroir, terminé par une sigmoïde.",
                f"décodeur attendu {want_dec_16}, reçu {_layout_16(model_16.decoder)}.")
        other_16 = make_deep_ae_16(2, hidden=(64,))
        verdict("25.16", _layout_16(other_16.encoder) == [(784, 64), "ReLU", (64, 2)]
                and _layout_16(other_16.decoder) == [(2, 64), "ReLU", (64, 784), "Sigmoid"],
                "make_deep_ae_16(2, hidden=(64,)) suit aussi n_latent et hidden.",
                "make_deep_ae_16 doit suivre n_latent et hidden (essayé avec 2 et (64,)).")'''

EXPERIMENT_16 = r'''with checked("25.16"):
    EPOCHS_16 = 8 if FAST_MODE else 20
    histories_16 = {}
    for name_16, maker_16 in {"dense 784-20-784": make_dense_ae, "profond 512-256-128-20": make_deep_ae_16}.items():
        torch.manual_seed(2516)
        model_16 = maker_16(20)
        histories_16[name_16] = (model_16, train_ae(model_16, X_MNIST, EPOCHS_16, seed=2516, x_val=XM_TEST))
        print(f"{name_16:24s} {n_params(model_16):9,d} paramètres · MSE de test {histories_16[name_16][1]['val_mse'][-1]:.4f} "
              f"· {sum(histories_16[name_16][1]['seconds']):.1f} s")
    mse_dense_16, mse_deep_16 = (h["val_mse"][-1] for _, h in histories_16.values())
    verdict("25.16", mse_deep_16 < mse_dense_16, "à 20 latents, le profond reconstruit mieux que le dense.",
            "le profond devrait battre le dense à 20 latents : vérifie ton architecture.")
    show_rows([XM_TEST[:10]] + [reconstruct(m, XM_TEST[:10]) for m, _ in histories_16.values()],
              ["originaux", "dense", "profond"], f"{EPOCHS_16} epochs sur {len(X_MNIST)} images")'''

GIVEN_DEEP = r'''# The reference deep autoencoder (your 25.16), and AE_DEEP: the one that parts A and C explore (25.17 to 25.21,
# 25.31), trained the same way in both modes (the first N_FIXED images, 8 epochs, seed 2516; about 10 s on a CPU)
def make_deep_ae(n_latent=20, hidden=(512, 256, 128)):
    """The deep autoencoder of 25.16: dense ReLU layers, a linear code, a sigmoid output."""
    sizes = [784, *hidden, n_latent]
    encoder, decoder = [], []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        encoder += [nn.Linear(n_in, n_out), nn.ReLU()]
    for n_in, n_out in zip(sizes[::-1][:-1], sizes[::-1][1:]):
        decoder += [nn.Linear(n_in, n_out), nn.ReLU()]
    return AutoEncoder(nn.Sequential(*encoder[:-1]), nn.Sequential(*decoder[:-1], nn.Sigmoid()))


torch.manual_seed(2516)
AE_DEEP = make_deep_ae(20)
train_ae(AE_DEEP, X_MNIST[:N_FIXED], 8, seed=2516)
print(f"AE_DEEP : MSE de test {mse_per_image(reconstruct(AE_DEEP, XM_TEST), XM_TEST).mean():.4f}")'''

TODO_17 = r'''ratio_17 = ...    # a) mean MSE of the clothes divided by the mean MSE of the digits: "A" less than 2,
                  #    "B" between 2 and 5, "C" more than 5
best_17 = ...     # b) the two best reconstructed classes of clothes (smallest mean MSE): "A" Trouser and Dress,
                  #    "B" Sandal and Sneaker, "C" Bag and Ankle boot, "D" T-shirt/top and Pullover
black_17 = ...    # c) is the mean MSE of the clothes smaller than that of an all-black guess (0 everywhere)?
                  #    True or False'''

SOLUTION_17 = r'''ratio_17, best_17, black_17 = "B", "B", True'''

EXPERIMENT_17 = r'''with checked("25.17"):
    mse_digits_17 = mse_per_image(reconstruct(AE_DEEP, XM_TEST), XM_TEST)
    out_clothes_17 = reconstruct(AE_DEEP, XF_TEST)
    mse_clothes_17 = mse_per_image(out_clothes_17, XF_TEST)
    mse_black_17 = mse_per_image(np.zeros_like(XF_TEST), XF_TEST)
    print(f"MSE moyenne · chiffres {mse_digits_17.mean():.4f} · vêtements {mse_clothes_17.mean():.4f} "
          f"(rapport {mse_clothes_17.mean() / mse_digits_17.mean():.2f}) · image noire sur les vêtements {mse_black_17.mean():.4f}")
    for c_17 in np.argsort([mse_clothes_17[YF_TEST == c].mean() for c in range(10)]):
        print(f"  {FASHION_CLASSES[c_17]:12s} {mse_clothes_17[YF_TEST == c_17].mean():.4f}")
    first_17 = [np.flatnonzero(YF_TEST == c)[0] for c in range(10)]
    show_rows([XF_TEST[first_17], out_clothes_17[first_17]], ["vêtements", "reconstruits"],
              "un vêtement de chaque classe, à travers l'autoencodeur de chiffres", size=1.1)
wb.check("25.17a", ratio_17)
wb.check("25.17b", best_17)
wb.check("25.17c", black_17)'''

RECORD_17 = (r'''wb.record("25.17a", "B", mistakes=''' + letters("B", {
    "A": "calcule le rapport des deux MSE moyennes affichées : il est plus grand que 2",
    "C": "calcule le rapport des deux MSE moyennes affichées : il reste sous 5"}) + r''')
wb.record("25.17b", "B", mistakes=''' + letters("B", {
    "A": "relis le tableau par classe, trié de la meilleure à la pire reconstruction : ce ne sont pas les deux premières",
    "C": "relis le tableau par classe : ces deux classes sont en bas, pas en haut",
    "D": "relis le tableau par classe, trié de la meilleure à la pire : ces deux-là sont au milieu"}) + r''')
wb.record("25.17c", True, mistakes={"compare la MSE moyenne des vêtements à celle de l'image noire": False})''')

GIVEN_18 = r'''# AE_2D: a deep autoencoder with a code of 2 numbers, trained on the first N_FIXED images (10 epochs, ≈ 12 s)
torch.manual_seed(2518)
AE_2D = make_deep_ae(2)
train_ae(AE_2D, X_MNIST[:N_FIXED], 10, seed=2518)
print(f"AE_2D : MSE de test {mse_per_image(reconstruct(AE_2D, XM_TEST), XM_TEST).mean():.4f}")'''

TODO_18 = r'''def grid_codes_18(lims, n):
    """The n × n codes of a regular grid over lims = (xmin, xmax, ymin, ymax), as an array (n * n, 2), row by row
    as in an image: the first row has y = ymax (the top), the last y = ymin; in each row, x goes from xmin to xmax.
    Both ends are included (np.linspace)."""
    raise NotImplementedError("grid_codes_18")


def plot_latent_map_18(model, x, labels, n=15):
    """Two panels side by side: (left) the codes model.encoder(x) as points (x, y), coloured by digit (labels),
    with a legend; (right) the n × n images decoded from grid_codes_18 over the same limits as the left panel
    (the min and max of the codes), assembled in ONE image of (28 n) × (28 n) pixels, the code of the top-left
    corner first. Ends with plt.show(). Returns the mosaic (an array (28 n, 28 n))."""
    raise NotImplementedError("plot_latent_map_18")'''

SOLUTION_18 = r'''def grid_codes_18(lims, n):
    """The n × n codes of a regular grid over lims = (xmin, xmax, ymin, ymax), as an array (n * n, 2), row by row
    as in an image: the first row has y = ymax (the top), the last y = ymin; in each row, x goes from xmin to xmax.
    Both ends are included (np.linspace)."""
    xmin, xmax, ymin, ymax = lims
    xs, ys = np.linspace(xmin, xmax, n), np.linspace(ymax, ymin, n)       # y from the top to the bottom
    gx, gy = np.meshgrid(xs, ys)                                          # gx[i, j] = xs[j], gy[i, j] = ys[i]
    return np.column_stack([gx.ravel(), gy.ravel()])


def plot_latent_map_18(model, x, labels, n=15):
    """Two panels side by side: (left) the codes model.encoder(x) as points (x, y), coloured by digit (labels),
    with a legend; (right) the n × n images decoded from grid_codes_18 over the same limits as the left panel
    (the min and max of the codes), assembled in ONE image of (28 n) × (28 n) pixels, the code of the top-left
    corner first. Ends with plt.show(). Returns the mosaic (an array (28 n, 28 n))."""
    codes = encode(model, x)
    lims = (codes[:, 0].min(), codes[:, 0].max(), codes[:, 1].min(), codes[:, 1].max())
    images = decode(model, grid_codes_18(lims, n).astype(np.float32)).reshape(n, n, 28, 28)
    mosaic = images.transpose(0, 2, 1, 3).reshape(28 * n, 28 * n)        # rows of images, then rows of pixels
    fig, (ax_map, ax_grid) = plt.subplots(1, 2, figsize=(12, 5.6))
    colors = wb.plot.class_colors(10)
    for digit in range(10):
        keep = np.asarray(labels) == digit
        ax_map.scatter(codes[keep, 0], codes[keep, 1], s=3, color=colors[digit], label=str(digit))
    ax_map.set_xlim(lims[0], lims[1])
    ax_map.set_ylim(lims[2], lims[3])
    ax_map.legend(markerscale=4, fontsize=8, ncol=2)
    ax_map.set_title("codes de 2 000 chiffres de test")
    ax_grid.imshow(mosaic, cmap="gray", extent=lims, aspect="auto")
    ax_grid.set_title(f"{n} × {n} codes de la grille, décodés")
    plt.tight_layout()
    plt.show()
    return mosaic'''

CHECK_18 = r'''with checked("25.18"):
    grid_18 = grid_codes_18((0.0, 1.0, -2.0, 2.0), 3)
    if returned("25.18", "grid_codes_18", grid_18):
        same("25.18", "grid_codes_18((0, 1, -2, 2), 3)", grid_18,
             [[0, 2], [0.5, 2], [1, 2], [0, 0], [0.5, 0], [1, 0], [0, -2], [0.5, -2], [1, -2]])
    mosaic_18 = plot_latent_map_18(AE_2D, XM_TEST, YM_TEST, n=15)
    if returned("25.18", "plot_latent_map_18", mosaic_18):
        codes_18 = encode(AE_2D, XM_TEST)
        corner_18 = decode(AE_2D, np.array([[codes_18[:, 0].min(), codes_18[:, 1].max()]], dtype=np.float32))
        verdict("25.18", np.shape(mosaic_18) == (420, 420) and np.allclose(np.asarray(mosaic_18)[:28, :28],
                                                                           corner_18.reshape(28, 28), atol=1e-5),
                "la mosaïque (420, 420) commence par le code du coin en haut à gauche (x min, y max).",
                f"attendu une mosaïque (420, 420) dont l'image en haut à gauche décode (x min, y max), reçu "
                f"{shape_text(mosaic_18)} (transpose(0, 2, 1, 3) avant le reshape).")'''

TODO_19 = r'''def add_noise_19(codes, scale, rng, column=None):
    """A new array: codes (N, d) plus uniform noise in [-scale, scale], drawn by rng.uniform(-scale, scale, size=...):
    on every value (size = codes.shape) if column is None, otherwise on that column only (size = N, the other
    columns unchanged). `codes` is not modified."""
    raise NotImplementedError("add_noise_19")


def noise_mse_19(model, x, scales, seed=0, column=None):
    """For each scale of `scales`: the codes encode(model, x), plus add_noise_19(..., scale,
    np.random.default_rng(seed), column) (a NEW generator for each scale), decoded; returns the list of the mean
    MSE between these decoded images and x."""
    raise NotImplementedError("noise_mse_19")'''

SOLUTION_19 = r'''def add_noise_19(codes, scale, rng, column=None):
    """A new array: codes (N, d) plus uniform noise in [-scale, scale], drawn by rng.uniform(-scale, scale, size=...):
    on every value (size = codes.shape) if column is None, otherwise on that column only (size = N, the other
    columns unchanged). `codes` is not modified."""
    noisy = np.array(codes, dtype=np.float32)               # a copy
    if column is None:
        noisy += rng.uniform(-scale, scale, size=noisy.shape)
    else:
        noisy[:, column] += rng.uniform(-scale, scale, size=len(noisy))
    return noisy


def noise_mse_19(model, x, scales, seed=0, column=None):
    """For each scale of `scales`: the codes encode(model, x), plus add_noise_19(..., scale,
    np.random.default_rng(seed), column) (a NEW generator for each scale), decoded; returns the list of the mean
    MSE between these decoded images and x."""
    codes = encode(model, x)
    return [float(mse_per_image(decode(model, add_noise_19(codes, s, np.random.default_rng(seed), column)), x).mean())
            for s in scales]'''

CHECK_19 = r'''with checked("25.19"):
    toy_19 = np.arange(6, dtype=np.float32).reshape(3, 2)
    noisy_19 = add_noise_19(toy_19, 0.5, np.random.default_rng(1))
    if returned("25.19", "add_noise_19", noisy_19):
        same("25.19", "add_noise_19(toy, 0.5, default_rng(1)) : tout le tableau",
             noisy_19, toy_19 + np.random.default_rng(1).uniform(-0.5, 0.5, size=(3, 2)), rtol=1e-6, atol=1e-6)
        same("25.19", "add_noise_19(toy, 0.5, default_rng(1), column=1) : la colonne 1 seulement",
             add_noise_19(toy_19, 0.5, np.random.default_rng(1), column=1),
             np.column_stack([toy_19[:, 0], toy_19[:, 1] + np.random.default_rng(1).uniform(-0.5, 0.5, size=3)]),
             rtol=1e-6, atol=1e-6)
        verdict("25.19", np.array_equal(toy_19, np.arange(6).reshape(3, 2)), "codes n'est pas modifié.",
                "add_noise_19 a modifié son entrée : travaille sur une copie.")
    mse_19 = noise_mse_19(AE_DEEP, XM_TEST[:200], [0.0, 1.0])
    if returned("25.19", "noise_mse_19", mse_19):
        clean_19 = mse_per_image(reconstruct(AE_DEEP, XM_TEST[:200]), XM_TEST[:200]).mean()
        verdict("25.19", len(mse_19) == 2 and abs(mse_19[0] - clean_19) < 1e-6 and mse_19[1] > mse_19[0],
                "noise_mse_19 : un bruit nul redonne l'erreur de reconstruction, un bruit de 1 l'augmente.",
                f"attendu [MSE sans bruit = {clean_19:.4f}, une valeur plus grande], reçu {mse_19!r:.200}.")'''

EXPERIMENT_19 = r'''with checked("25.19"):
    codes_19 = encode(AE_DEEP, XM_TEST)
    spread_19 = float(codes_19.std())                      # the typical spread of a latent value
    rel_19 = (0.0, 0.1, 0.3, 1.0, 3.0)
    curve_19 = noise_mse_19(AE_DEEP, XM_TEST, [r * spread_19 for r in rel_19], seed=19)
    print(f"écart-type des latents : {spread_19:.2f} ; leurs valeurs vont de {codes_19.min():.1f} à {codes_19.max():.1f}")
    for r_19, m_19 in zip(rel_19, curve_19):
        print(f"bruit de ± {r_19:.1f} écart-type sur tous les latents : MSE {m_19:.4f}")
    one_19 = [noise_mse_19(AE_DEEP, XM_TEST, [3 * spread_19], seed=19, column=j)[0] for j in range(codes_19.shape[1])]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.bar(range(len(one_19)), one_19)
    ax.axhline(curve_19[0], color="k", lw=1, ls="--", label="sans bruit")
    ax.set_xlabel("latent bruité (± 3 écarts-types, lui seul)")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.show()
    rng_19 = np.random.default_rng(1919)
    random_19 = rng_19.uniform(codes_19.min(axis=0), codes_19.max(axis=0), size=(10, codes_19.shape[1])).astype(np.float32)
    rows_19 = [XM_TEST[:10]] + [decode(AE_DEEP, add_noise_19(codes_19[:10], r * spread_19, np.random.default_rng(19)))
                                for r in (0.3, 1.0, 3.0)] + [decode(AE_DEEP, random_19)]
    show_rows(rows_19, ["originaux", "± 0,3 σ", "± 1 σ", "± 3 σ", "codes au hasard"])'''

TODO_20 = r'''# Nothing to write here: interpolate_latents goes in mon_travail/mylearn/vae.py
z0_20 = encode(AE_DEEP, XM_TEST[np.flatnonzero(YM_TEST == 1)[:1]])[0]    # the code of a 1 ...
z1_20 = encode(AE_DEEP, XM_TEST[np.flatnonzero(YM_TEST == 0)[:1]])[0]    # ... and that of a 0'''

CHECK_20 = r'''with checked("25.20"):
    V = mylearn.vae
    a_20, b_20 = np.array([0.0, 0.0]), np.array([1.0, 2.0])
    same("25.20", "interpolate_latents((0, 0), (1, 2), 3) : la ligne droite", V.interpolate_latents(a_20, b_20, 3),
         [[0, 0], [0.5, 1], [1, 2]])
    u_20, w_20 = np.array([2.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    t_20 = np.linspace(0, 1, 4)[:, None]
    same("25.20", "slerp de (2, 0, 0) à (0, 2, 0) en 4 étapes : l'arc de cercle de rayon 2",
         V.interpolate_latents(u_20, w_20, 4, method="slerp"),
         np.sin((1 - t_20) * np.pi / 2) * u_20 + np.sin(t_20 * np.pi / 2) * w_20)
    path_lin_20 = V.interpolate_latents(z0_20, z1_20, 8)
    path_slerp_20 = V.interpolate_latents(z0_20, z1_20, 8, method="slerp")
    print("normes des codes, ligne droite :", np.round(np.linalg.norm(path_lin_20, axis=1), 1))
    print("normes des codes, slerp        :", np.round(np.linalg.norm(path_slerp_20, axis=1), 1))
    show_rows([decode(AE_DEEP, np.asarray(path_lin_20, dtype=np.float32)),
               decode(AE_DEEP, np.asarray(path_slerp_20, dtype=np.float32))], ["ligne droite", "slerp"],
              "d'un 1 à un 0 dans l'espace latent de AE_DEEP")'''

TODO_21 = r'''def blend_pixels_21(x0, x1, n_steps):
    """The images (1 - t) x0 + t x1 for t = np.linspace(0, 1, n_steps): an array (n_steps, 784)."""
    raise NotImplementedError("blend_pixels_21")


def blend_codes_21(model, x0, x1, n_steps, method="linear"):
    """Encode x0 and x1, interpolate their codes with mylearn.vae.interpolate_latents(..., n_steps, method=method),
    and decode: an array (n_steps, 784)."""
    raise NotImplementedError("blend_codes_21")


def off_manifold_21(model, images):
    """How far each image is from what the autoencoder can redraw: the MSE between each image and its
    reconstruction, an array of len(images) values."""
    raise NotImplementedError("off_manifold_21")'''

SOLUTION_21 = r'''def blend_pixels_21(x0, x1, n_steps):
    """The images (1 - t) x0 + t x1 for t = np.linspace(0, 1, n_steps): an array (n_steps, 784)."""
    t = np.linspace(0, 1, n_steps)[:, None]
    return (1 - t) * np.asarray(x0, dtype=np.float32) + t * np.asarray(x1, dtype=np.float32)


def blend_codes_21(model, x0, x1, n_steps, method="linear"):
    """Encode x0 and x1, interpolate their codes with mylearn.vae.interpolate_latents(..., n_steps, method=method),
    and decode: an array (n_steps, 784)."""
    z0, z1 = encode(model, np.stack([x0, x1]))
    path = mylearn.vae.interpolate_latents(z0, z1, n_steps, method=method)
    return decode(model, np.asarray(path, dtype=np.float32))


def off_manifold_21(model, images):
    """How far each image is from what the autoencoder can redraw: the MSE between each image and its
    reconstruction, an array of len(images) values."""
    images = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    return mse_per_image(reconstruct(model, images), images)'''

CHECK_21 = r'''with checked("25.21"):
    toy_a_21, toy_b_21 = np.zeros(784, dtype=np.float32), np.ones(784, dtype=np.float32)
    mix_21 = blend_pixels_21(toy_a_21, toy_b_21, 5)
    if returned("25.21", "blend_pixels_21", mix_21):
        same("25.21", "blend_pixels_21(noir, blanc, 5)[:, 0]", np.asarray(mix_21)[:, 0], [0, 0.25, 0.5, 0.75, 1], rtol=1e-6)
    x0_21, x1_21 = XM_TEST[np.flatnonzero(YM_TEST == 1)[0]], XM_TEST[np.flatnonzero(YM_TEST == 0)[0]]
    codes_21 = blend_codes_21(AE_DEEP, x0_21, x1_21, 6)
    if returned("25.21", "blend_codes_21", codes_21):
        z_21 = encode(AE_DEEP, np.stack([x0_21, x1_21]))
        same("25.21", "blend_codes_21(AE_DEEP, un 1, un 0, 6) : les extrémités sont les reconstructions",
             np.asarray(codes_21)[[0, -1]], decode(AE_DEEP, z_21), rtol=1e-4, atol=1e-5)
    off_21 = off_manifold_21(AE_DEEP, XM_TEST[:3])
    if returned("25.21", "off_manifold_21", off_21):
        same("25.21", "off_manifold_21(AE_DEEP, 3 chiffres)", off_21,
             ((reconstruct(AE_DEEP, XM_TEST[:3]) - XM_TEST[:3]) ** 2).mean(axis=1), rtol=1e-5, atol=1e-7)'''

EXPERIMENT_21 = r'''with checked("25.21"):
    def _pick_21(digit, k=0):
        return XM_TEST[np.flatnonzero(YM_TEST == digit)[k]]
    for (a_21, b_21) in [(_pick_21(1), _pick_21(0)), (_pick_21(5, 0), _pick_21(5, 3))]:
        rows_21 = {"pixels": blend_pixels_21(a_21, b_21, 7), "codes, ligne droite": blend_codes_21(AE_DEEP, a_21, b_21, 7),
                   "codes, slerp": blend_codes_21(AE_DEEP, a_21, b_21, 7, method="slerp")}
        show_rows(list(rows_21.values()), list(rows_21.keys()), size=1.1)
        for name_21, images_21 in rows_21.items():
            print(f"  {name_21:20s} distance à ce que l'AE sait refaire, étape par étape : "
                  + " ".join(f"{v:.3f}" for v in off_manifold_21(AE_DEEP, images_21)))'''

TODO_22 = r'''def bottleneck_sweep_22(sizes, makers, epochs, seed=2522, x=None):
    """For each name, maker of the dict `makers` and each size of `sizes`: torch.manual_seed(seed), model =
    maker(size), train_ae(model, x, epochs, seed=seed) (x = X_MNIST if None), then the mean MSE of reconstruction
    on XM_TEST. Returns the dict {(name, size): that MSE}."""
    raise NotImplementedError("bottleneck_sweep_22")'''

SOLUTION_22 = r'''def bottleneck_sweep_22(sizes, makers, epochs, seed=2522, x=None):
    """For each name, maker of the dict `makers` and each size of `sizes`: torch.manual_seed(seed), model =
    maker(size), train_ae(model, x, epochs, seed=seed) (x = X_MNIST if None), then the mean MSE of reconstruction
    on XM_TEST. Returns the dict {(name, size): that MSE}."""
    x = X_MNIST if x is None else x
    results = {}
    for name, maker in makers.items():
        for size in sizes:
            torch.manual_seed(seed)
            model = maker(size)
            train_ae(model, x, epochs, seed=seed)
            results[(name, size)] = float(mse_per_image(reconstruct(model, XM_TEST), XM_TEST).mean())
    return results'''

CHECK_22 = r'''with checked("25.22"):
    tiny_22 = bottleneck_sweep_22([3], {"dense": make_dense_ae}, epochs=1, seed=7, x=X_MNIST[:512])
    if returned("25.22", "bottleneck_sweep_22", tiny_22):
        torch.manual_seed(7)
        model_22 = make_dense_ae(3)
        train_ae(model_22, X_MNIST[:512], 1, seed=7)
        want_22 = float(mse_per_image(reconstruct(model_22, XM_TEST), XM_TEST).mean())
        verdict("25.22", isinstance(tiny_22, dict) and set(tiny_22) == {("dense", 3)}
                and abs(tiny_22[("dense", 3)] - want_22) < 1e-5,
                f"bottleneck_sweep_22 sur un cas d'essai : {{('dense', 3): {want_22:.4f}}}.",
                f"attendu {{('dense', 3): {want_22:.4f}}}, reçu {tiny_22!r:.200} (torch.manual_seed(seed) juste avant "
                "maker(size), puis train_ae(..., seed=seed)).")'''

EXPERIMENT_22 = r'''with checked("25.22"):
    SIZES_22 = (2, 5, 10, 20, 50)
    EPOCHS_22 = 4 if FAST_MODE else 15
    start_22 = time.perf_counter()
    sweep_22 = bottleneck_sweep_22(SIZES_22, {"dense": make_dense_ae, "profond": make_deep_ae}, EPOCHS_22)
    print(f"{len(sweep_22)} entraînements en {time.perf_counter() - start_22:.0f} s")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name_22 in ("dense", "profond"):
        ax.plot(SIZES_22, [sweep_22[(name_22, s)] for s in SIZES_22], "o-", label=name_22)
    ax.set_xscale("log")
    ax.set_xticks(SIZES_22, [str(s) for s in SIZES_22])
    ax.set_xlabel("taille du goulot (latents)")
    ax.set_ylabel("MSE de test")
    ax.legend()
    plt.show()
    for s_22 in SIZES_22:
        print(f"{s_22:3d} latents · dense {sweep_22[('dense', s_22)]:.4f} · profond {sweep_22[('profond', s_22)]:.4f}")'''

TODO_23 = r'''def fit_linear_ae_23(xc, k, epochs=40, lr=3e-3, batch_size=256, seed=2523):
    """A LINEAR autoencoder on the centred images xc (N, 784): encoder nn.Linear(784, k, bias=False), decoder
    nn.Linear(k, 784, bias=False), no activation, both created just after torch.manual_seed(seed); trained with
    Adam(lr) on the mean squared error, mini-batches in an order drawn by ONE torch.Generator().manual_seed(seed).
    Returns the decoder's weight as a NumPy array (784, k): its k columns span the learned subspace."""
    raise NotImplementedError("fit_linear_ae_23")


def subspace_angles_deg_23(A, B):
    """The principal angles, in degrees, between the subspace spanned by the columns of A and that of B (both
    (p, k)), from the largest to the smallest: scipy.linalg.subspace_angles, converted."""
    raise NotImplementedError("subspace_angles_deg_23")'''

SOLUTION_23 = r'''def fit_linear_ae_23(xc, k, epochs=40, lr=3e-3, batch_size=256, seed=2523):
    """A LINEAR autoencoder on the centred images xc (N, 784): encoder nn.Linear(784, k, bias=False), decoder
    nn.Linear(k, 784, bias=False), no activation, both created just after torch.manual_seed(seed); trained with
    Adam(lr) on the mean squared error, mini-batches in an order drawn by ONE torch.Generator().manual_seed(seed).
    Returns the decoder's weight as a NumPy array (784, k): its k columns span the learned subspace."""
    torch.manual_seed(seed)
    model = AutoEncoder(nn.Linear(784, k, bias=False), nn.Linear(k, 784, bias=False))
    train_ae(model, xc, epochs, lr=lr, batch_size=batch_size, seed=seed, loss="mse")
    return model.decoder.weight.detach().cpu().numpy().astype(np.float64)


def subspace_angles_deg_23(A, B):
    """The principal angles, in degrees, between the subspace spanned by the columns of A and that of B (both
    (p, k)), from the largest to the smallest: scipy.linalg.subspace_angles, converted."""
    return np.degrees(subspace_angles(np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)))'''

CHECK_23 = r'''with checked("25.23"):
    plane_a_23 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])                       # the plane z = 0
    tilt_23 = np.radians(30)
    plane_b_23 = np.array([[1.0, 0.0], [0.0, np.cos(tilt_23)], [0.0, np.sin(tilt_23)]])  # the same, tilted by 30°
    ang_23 = subspace_angles_deg_23(plane_a_23, plane_b_23)
    if returned("25.23", "subspace_angles_deg_23", ang_23):
        same("25.23", "subspace_angles_deg_23(plan z = 0, le même incliné de 30°)", ang_23, [30.0, 0.0], atol=1e-9)
        same("25.23", "deux bases différentes du même plan", subspace_angles_deg_23(plane_a_23, plane_a_23 @ [[2.0, 1.0], [1.0, 3.0]]),
             [0.0, 0.0], atol=1e-6)
    xc_23 = X_MNIST[:2000] - X_MNIST[:2000].mean(axis=0)
    W_23 = fit_linear_ae_23(xc_23, 4, epochs=2)
    if returned("25.23", "fit_linear_ae_23", W_23):
        torch.manual_seed(2523)
        enc_23, dec_23 = nn.Linear(784, 4, bias=False), nn.Linear(4, 784, bias=False)
        train_ae(AutoEncoder(enc_23, dec_23), xc_23, 2, lr=3e-3, batch_size=256, seed=2523, loss="mse")
        same("25.23", "fit_linear_ae_23(2 000 images centrées, k=4, epochs=2) : la matrice du décodeur (784, 4)",
             W_23, dec_23.weight.detach().cpu().numpy(), rtol=1e-4, atol=1e-6)'''

EXPERIMENT_23 = r'''with checked("25.23"):
    K_23 = 16
    x_23 = X_MNIST[:N_FIXED].astype(np.float64)
    xc_23 = x_23 - x_23.mean(axis=0)                          # centred with the mean of the training images
    pca_23 = PCA(n_components=K_23, svd_solver="full").fit(xc_23)
    P_23 = pca_23.components_.T                               # (784, K): the first K principal axes
    W_23 = fit_linear_ae_23(xc_23.astype(np.float32), K_23)
    def _rec_mse_23(basis):
        Q, _ = np.linalg.qr(basis)                            # an orthonormal basis of the same subspace
        return float(((xc_23 - xc_23 @ Q @ Q.T) ** 2).mean())
    angles_23 = subspace_angles_deg_23(W_23, P_23)
    print(f"MSE de reconstruction · ACP {_rec_mse_23(P_23):.5f} · autoencodeur linéaire {_rec_mse_23(W_23):.5f}")
    print(f"angles entre les deux sous-espaces (degrés) : max {angles_23.max():.1f} · moyenne {angles_23.mean():.1f}")
    unit_23 = W_23 / np.linalg.norm(W_23, axis=0)
    print("|cos| entre chaque colonne du décodeur et l'axe de l'ACP de même rang :",
          np.round(np.abs((unit_23 * P_23).sum(axis=0)), 2))
    gram_23 = unit_23.T @ unit_23
    print(f"colonnes du décodeur orthogonales ? plus grand |cos| entre deux d'entre elles : "
          f"{np.abs(gram_23 - np.eye(K_23)).max():.2f}")
    show_rows([P_23.T[:8] / np.abs(P_23.T[:8]).max(axis=1, keepdims=True) / 2 + 0.5,
               unit_23.T[:8] / np.abs(unit_23.T[:8]).max(axis=1, keepdims=True) / 2 + 0.5],
              ["axes de l'ACP", "colonnes du décodeur"], "les 8 premières directions (gris = 0)", size=1.1)'''

PART_A = Part("A", "Autoencodeurs denses",
              "Tu fais apprendre une seule image à un autoencodeur (25.14), tu entraînes le premier autoencodeur du "
              "livre puis sa version profonde (25.15, 25.16), tu lui montres des vêtements (25.17), tu dessines son "
              "espace latent en 2D (25.18), tu le secoues (25.19), tu interpoles entre deux codes avec ta librairie "
              "(25.20, 25.21), tu fais varier le goulot (25.22) et tu compares l'autoencodeur linéaire à l'ACP (25.23).\n\n"
              "La cellule d'outils sert aux parties A à C. Elle importe PyTorch, choisit le device et charge MNIST et "
              "Fashion-MNIST en lignes de 784 pixels dans $[0, 1]$ (`float32`) : `X_MNIST`, `Y_MNIST`, `X_FASHION`, "
              "`Y_FASHION` (les `N_TRAIN` premières images d'un ordre fixé : 10 000 en mode rapide, 60 000 sinon ; "
              "les 10 000 premières, `N_FIXED`, sont les mêmes dans les deux modes), et 2 000 images de test de "
              "chacun (`XM_TEST`, `YM_TEST`, `XF_TEST`, `YF_TEST`). Elle donne la classe `AutoEncoder` (un encodeur "
              "puis un décodeur ; tous les autoencodeurs du notebook lisent et rendent des lignes de 784 pixels dans "
              "$[0, 1]$), et des outils qui travaillent sans graphe, en mode évaluation, et rendent des tableaux "
              "NumPy : `encode`, `decode`, `reconstruct`, `mse_per_image` (la MSE de chaque image) et `show_rows` "
              "(des images en lignes).",
              given=TOOLS, exercises=[
    Ex("25.14", "🔮", 1, 15, "Prédire : un autoencodeur entraîné sur une seule image",
       "prédire ce qu'un autoencodeur qui n'a vu qu'une image fait d'autres entrées, et où il range cette image.",
       "Ex 25.1 · ch. 23 (boucle d'entraînement) · fiche §25.3", thread="Fashion-MNIST", tracks="R, C",
       body=r"""La fiche raconte l'expérience du tigre : un autoencodeur entraîné longtemps sur une seule photo la reconstruit parfaitement, et le livre explique qu'il l'a mémorisée, au point de redonner le tigre à partir d'une image noire. Tu refais l'expérience en petit sur **une** botte de Fashion-MNIST : un autoencodeur 784-20-784 (`train_one_image_14`, 500 pas d'Adam, une graine fixée), puis le même **sans aucun biais** (`bias=False` dans ses deux couches).

**Avant d'exécuter**, écris tes trois prédictions dans la cellule suivante (`True` ou `False`) :
a) l'image **noire**, donnée à l'autoencodeur qui a appris la botte, ressort-elle comme la botte (MSE avec la botte < 0,01) ?
b) le **T-shirt**, donné au même autoencodeur, ressort-il plus proche de la botte que de lui-même (MSE avec la botte plus petite qu'avec le T-shirt) ?
c) **sans aucun biais**, l'autoencodeur reconstruit-il encore la botte (MSE < 0,001) ?

La cellule d'expérience n'entraîne rien tant que tes trois prédictions ne sont pas écrites (deux secondes ensuite). Elle affiche aussi ce que rend le décodeur seul pour un code nul : la part de l'image que portent ses biais.""",
       given=GIVEN_14, todo=TODO_14, solution=SOLUTION_14, hypothesis=True, record=RECORD_14,
       after=[("code", guarded(EXPERIMENT_14, ["black_14", "shirt_14", "no_bias_14"],
                               "⏳ Ex 25.14 : écris d'abord tes trois prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : lesquelles de tes prédictions étaient justes ? Pourquoi le T-shirt donne-t-il la "
                     "botte alors que l'image noire ne la donne pas (regarde ce que la première couche fait d'une "
                     "entrée nulle, puis la colonne « code nul ») ? Où ce réseau a-t-il rangé la botte : dans les biais "
                     "de sa sortie, ou dans les poids de son décodeur ? Qu'est-ce qui, chez le livre, a pu ranger le "
                     "tigre dans les biais ?")]),

    Ex("25.15", "📦", 2, 25, "Premier autoencodeur dense 784-20-784 en PyTorch",
       "écrire le premier autoencodeur du livre et sa boucle d'entraînement, dont la cible est l'entrée elle-même.",
       "Ex 25.1 · ch. 23 (Ex 23.18, boucle d'entraînement) · fiche §25.3, §25.4", thread="MNIST", tracks="R, M, C",
       body=r"""Le premier autoencodeur du livre sur MNIST : une couche dense de 784 vers 20 neurones avec une ReLU (l'encodeur), une couche de 20 vers 784 avec une sigmoïde (le décodeur), et une loss qui compare la sortie à l'**entrée**.

**Contrat**
- `make_dense_ae_15(n_latent=20)` rend un `AutoEncoder` dont l'encodeur est `nn.Sequential(nn.Linear(784, n_latent), nn.ReLU())` et le décodeur `nn.Sequential(nn.Linear(n_latent, 784), nn.Sigmoid())` ;
- `fit_ae_15(model, x, epochs, lr=1e-3, batch_size=128, seed=0)` entraîne `model` sur place : `x` est un tableau NumPy `float32` `(N, 784)` dans $[0, 1]$, l'optimiseur `Adam(lr)`, la loss la cross-entropy binaire **moyenne** entre `model(batch)` et le batch lui-même ; les mini-batches suivent un ordre tiré à chaque epoch par `torch.randperm(N, generator=g)`, avec **un seul** `g = torch.Generator().manual_seed(seed)` créé avant la première epoch ; les batches vont sur le device du modèle ; la fonction rend la liste des losses moyennes de chaque epoch (chaque batch pesé par sa taille).

La vérification regarde l'architecture et la sortie, puis entraîne deux fois trois epochs sur 1 024 images (la même graine doit donner les mêmes losses). La cellule suivante entraîne trois autoencodeurs, à 20, 10 et 3 latents (une dizaine de secondes en mode rapide), et affiche leurs reconstructions.

Dans tes notes : comment la qualité change-t-elle de 20 à 3 latents, et retrouves-tu ce que décrit la fiche (un 4 qui tourne au 9, des taches) ? Pourquoi une sigmoïde en sortie, et pourquoi la cross-entropy binaire convient-elle à des pixels qui ne sont pas des 0 et des 1 ?""",
       todo=TODO_15, check=CHECK_15, solution=SOLUTION_15 + "\n\n\n" + CHECK_15,
       after=[("code", EXPERIMENT_15), ("md", "La cellule suivante donne les versions de référence de "
               "`make_dense_ae` et de la boucle, `train_ae`, qui rend aussi un historique : les exercices suivants "
               "s'en servent."), ("code", GIVEN_TRAIN)]),

    Ex("25.16", "📦", 2, 25, "Autoencodeur profond 512-256-128-20",
       "construire un autoencodeur profond symétrique et mesurer ce que la profondeur apporte à goulot égal.",
       "Ex 25.15 · fiche §25.4", thread="MNIST", tracks="R, M, C",
       body=r"""Le livre passe à un autoencodeur **profond** : trois couches cachées de 512, 256 et 128 neurones avant le code, et le chemin inverse après. Ton code n'a pas d'activation : ses valeurs peuvent être négatives (le livre mettait une ReLU au goulot, d'où ses latents tous positifs).

**Contrat** : `make_deep_ae_16(n_latent=20, hidden=(512, 256, 128))` rend un `AutoEncoder` dont l'encodeur est `Linear(784, hidden[0])` + ReLU, `Linear(hidden[0], hidden[1])` + ReLU…, puis `Linear(hidden[-1], n_latent)` **sans** activation, et le décodeur le miroir : `Linear(n_latent, hidden[-1])` + ReLU…, `Linear(hidden[0], 784)` + sigmoïde ; encodeur et décodeur sont des `nn.Sequential`.

La vérification compare la suite des couches à celle attendue, pour deux réglages. La cellule suivante entraîne le dense de 25.15 et ton profond, avec le même `train_ae` (8 epochs en mode rapide, une quinzaine de secondes), et compare leurs erreurs de test.

Dans tes notes : combien de paramètres le profond a-t-il de plus, et que gagne-t-il ? Avec la fiche (« assez de latents et assez de capacité »), qu'est-ce qui limite chacun des deux modèles ?""",
       todo=TODO_16, check=CHECK_16, solution=SOLUTION_16 + "\n\n\n" + CHECK_16,
       after=[("code", EXPERIMENT_16), ("code", GIVEN_DEEP)]),

    Ex("25.17", "🔮", 2, 15, "Prédire : ce qu'un autoencodeur de chiffres fait d'un vêtement",
       "prédire, puis mesurer, l'erreur d'un autoencodeur sur des entrées d'un autre domaine que ses données.",
       "Ex 25.16 · 25.Q11 · fiche §25.5.4, §25.7.2", thread="Fashion-MNIST", tracks="C",
       body=r"""La fiche le dit : un autoencodeur de chiffres ne sait redessiner que des morceaux de chiffres. Mais de combien se trompe-t-il, et sur quels vêtements ? `AE_DEEP` (l'autoencodeur profond à 20 latents de la cellule précédente, entraîné sur des chiffres seulement) reçoit les 2 000 vêtements de test.

**Avant d'exécuter**, écris tes prédictions :
a) la MSE moyenne des vêtements divisée par celle des chiffres de test : `"A"` moins de 2, `"B"` entre 2 et 5, `"C"` plus de 5 ;
b) les **deux** classes de vêtements les mieux reconstruites (MSE moyennes les plus petites) : `"A"` Trouser et Dress (pantalon et robe), `"B"` Sandal et Sneaker (sandale et basket), `"C"` Bag et Ankle boot (sac et bottine), `"D"` T-shirt/top et Pullover ;
c) la MSE moyenne des vêtements est-elle plus petite que celle d'une réponse toute noire (0 partout) ? (`True` ou `False`)""",
       todo=TODO_17, solution=SOLUTION_17, hypothesis=True, record=RECORD_17,
       after=[("code", guarded(EXPERIMENT_17, ["ratio_17", "best_17", "black_17"],
                               "⏳ Ex 25.17 : écris d'abord tes trois prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : qu'ont en commun les vêtements les mieux reconstruits, et à quels chiffres "
                     "ressemblent leurs reconstructions ? Que t'apprend c) sur ce que l'autoencodeur a quand même "
                     "appris des images en général ? Quel usage pratique tire-t-on de cet écart d'erreur (25.37) ?")]),

    Ex("25.18", "🎨", 2, 30, "Reproduire la carte latente 2D et sa grille décodée (fig. 25.29-25.30)",
       "dessiner l'espace latent d'un autoencodeur à 2 latents et décoder une grille de codes.",
       "Ex 25.16 · fiche §25.5.2", thread="MNIST", tracks="C",
       body=r"""Le livre réduit chaque chiffre à **deux** nombres, place les codes de 10 000 chiffres sur un plan (fig. 25.29), puis décode une grille régulière de codes (fig. 25.30). La cellule donnée entraîne `AE_2D`, l'autoencodeur profond de 25.16 avec un code de 2 nombres.

**Contrat**
- `grid_codes_18(lims, n)` rend les $n \times n$ codes d'une grille régulière sur `lims = (xmin, xmax, ymin, ymax)`, en tableau `(n * n, 2)`, ligne par ligne **comme dans une image** : la première ligne a $y$ = `ymax` (le haut), la dernière $y$ = `ymin` ; dans chaque ligne, $x$ va de `xmin` à `xmax` ; les deux bords sont inclus ;
- `plot_latent_map_18(model, x, labels, n=15)` trace deux panneaux côte à côte : à gauche, les codes de `x` en points, colorés par chiffre (`labels`), avec une légende ; à droite, les $n \times n$ images décodées de `grid_codes_18` sur les **mêmes limites** que le panneau de gauche (le min et le max des codes), assemblées en **une** image de $28n \times 28n$ pixels, le code du coin en haut à gauche en premier ; elle finit par `plt.show()` et rend cette mosaïque.

La vérification essaie `grid_codes_18` sur une petite grille, puis trace la carte des 2 000 chiffres de test et compare le coin en haut à gauche de ta mosaïque au décodage de son code.

Dans tes notes : quels chiffres ont des régions bien à eux, lesquels se chevauchent ? Où sont les zones vides, et que décode la grille à ces endroits ? Pourquoi est-ce un problème pour générer des chiffres en tirant des codes au hasard ?""",
       given=GIVEN_18, todo=TODO_18, check=CHECK_18, solution=SOLUTION_18 + "\n\n\n" + CHECK_18),

    Ex("25.19", "🔬", 2, 25, "Secouer les variables latentes : bruit global, une seule composante, latents aléatoires",
       "mesurer la tolérance du décodeur à un bruit sur les codes, latent par latent, et le décodage de codes au hasard.",
       "Ex 25.16 · fiche §25.5, §25.5.1", thread="MNIST", tracks="C",
       body=r"""Le livre secoue les 20 latents de ses chiffres avec des bruits uniformes de plus en plus forts, puis un seul latent à la fois, puis remplace les codes par des nombres au hasard. Tu refais les trois expériences sur `AE_DEEP` ; comme ses latents n'ont pas l'échelle de ceux du livre, les bruits sont comptés en **écarts-types** des latents.

**Contrat**
- `add_noise_19(codes, scale, rng, column=None)` rend un **nouveau** tableau : `codes` `(N, d)` plus un bruit uniforme dans $[-\text{scale}, \text{scale}]$, tiré par `rng.uniform(-scale, scale, size=...)`, sur toutes les valeurs (`size=codes.shape`) si `column` vaut `None`, sinon sur cette colonne seulement (`size=N`) ; `codes` n'est pas modifié ;
- `noise_mse_19(model, x, scales, seed=0, column=None)` : pour chaque valeur de `scales`, les codes `encode(model, x)`, bruités par `add_noise_19(..., scale, np.random.default_rng(seed), column)` (un **nouveau** générateur pour chaque valeur), puis décodés ; elle rend la liste des MSE moyennes entre ces images et `x`.

La vérification compare tes bruits à ceux d'un générateur de même graine. La cellule suivante trace l'erreur selon le bruit, puis l'erreur quand un seul latent est bruité (un diagramme en barres), et décode des codes tirés au hasard entre le min et le max de chaque latent.

Dans tes notes : à partir de quel bruit les chiffres deviennent-ils illisibles ? Les latents comptent-ils tous autant ? Que donnent les codes au hasard, et pourquoi (25.18) ?""",
       todo=TODO_19, check=CHECK_19, solution=SOLUTION_19 + "\n\n\n" + CHECK_19, after=[("code", EXPERIMENT_19)]),

    Ex("25.20", "🔨", 2, 20, "Interpoler entre deux codes : linéaire et sphérique",
       "écrire l'interpolation linéaire et sphérique (slerp) de deux codes dans ta librairie.",
       "Ex 25.6 · fiche §25.5.3 (encadré slerp)", tracks="R, M, C", mylearn="vae.py",
       body=mylearn_howto() + "\n\n" + COMMON_VAE + r"""

Écris `interpolate_latents(z0, z1, n_steps, method="linear")` dans `vae.py`. **Contrat** propre à cette fonction :
- `z0` et `z1` sont deux vecteurs 1-D de même forme, sinon `ValueError` ; `n_steps` est un entier ≥ 2 (un `bool` ou un flottant sont refusés), `method` vaut `"linear"` ou `"slerp"`, sinon `ValueError` ;
- les valeurs de $t$ sont `np.linspace(0, 1, n_steps)`, et la fonction rend un tableau `(n_steps, d)` dont la première ligne est **exactement** `z0` et la dernière **exactement** `z1` ;
- `"linear"` : $(1 - t)\,\mathbf{z}_0 + t\,\mathbf{z}_1$ ; `"slerp"` : la formule de la fiche, avec $\Omega$ l'angle entre les deux vecteurs, qui s'applique à des vecteurs de **normes quelconques** ; elle revient à la ligne droite quand une norme est nulle ou quand $\sin\Omega$ est de l'ordre des erreurs d'arrondi (vecteurs parallèles ou opposés), sans émettre de `RuntimeWarning` : teste ces cas **avant** de diviser.

La cellule ci-dessous prend les codes d'un 1 et d'un 0 dans l'espace latent de `AE_DEEP`. La première cellule de vérification essaie ta fonction sur une ligne droite et un quart de cercle, puis décode les deux chemins du 1 au 0 et affiche la norme des codes en route. La seconde lance les tests (`-k "test_interpolate_latents_"`).

Dans tes notes : la norme des codes reste-t-elle constante le long du slerp, et le long de la ligne droite ? Ici, les images des deux chemins se ressemblent-elles ? Dans quel espace latent la différence devrait-elle compter le plus (fiche, et 25.Q9) ?""",
       todo=TODO_20, check=RELOAD + CHECK_20, solution=TODO_20 + "\n\n" + CHECK_20,
       after=tests_after("25.20", "mylearn.vae.interpolate_latents([0.0], [1.0], 2)", "test_interpolate_latents_")),

    Ex("25.21", "🔬", 2, 25, "Mélanger dans l'espace des pixels ou dans l'espace latent",
       "comparer un fondu de pixels à une interpolation de codes, et mesurer à quel point chaque étape est « un chiffre ».",
       "Ex 25.20 · Ex 25.16 · ✏️ 25.2 · fiche §25.2.3, §25.5.3", thread="MNIST", tracks="R, C",
       body=r"""Le ✏️ 25.2 opposait le mélange de contenu (les pixels) au mélange paramétrique. Un autoencodeur donne des « paramètres » appris : ses codes. Tu compares trois façons de passer d'un chiffre à un autre, en sept étapes : le fondu des pixels, la ligne droite entre les codes, le slerp entre les codes.

**Contrat**
- `blend_pixels_21(x0, x1, n_steps)` rend les images $(1 - t)\,\mathbf{x}_0 + t\,\mathbf{x}_1$ pour $t$ = `np.linspace(0, 1, n_steps)`, un tableau `(n_steps, 784)` ;
- `blend_codes_21(model, x0, x1, n_steps, method="linear")` encode `x0` et `x1`, interpole leurs codes avec **ta** `mylearn.vae.interpolate_latents(..., n_steps, method=method)` et décode : un tableau `(n_steps, 784)` ;
- `off_manifold_21(model, images)` rend, pour chaque image, la MSE entre elle et sa reconstruction par `model` : une mesure de « ce que l'autoencodeur ne sait pas redessiner ».

La vérification essaie tes trois fonctions sur des cas connus. La cellule suivante montre les trois mélanges d'un 1 à un 0, puis entre deux 5 différents, et affiche pour chaque étape la distance à ce que l'autoencodeur sait refaire.

Dans tes notes : à mi-chemin, laquelle des trois images ressemble le plus à un chiffre, et que disent les distances ? Pourquoi les étapes d'un fondu de pixels sont-elles, elles, « loin » de ce que l'autoencodeur sait refaire ? Le mélange de deux 5 se passe-t-il mieux que celui d'un 1 et d'un 0, et pourquoi ?""",
       todo=TODO_21, check=CHECK_21, solution=SOLUTION_21 + "\n\n\n" + CHECK_21, after=[("code", EXPERIMENT_21)]),

    Ex("25.22", "🔬", 3, 40, "Taille du goulot et profondeur : la courbe d'erreur de reconstruction",
       "tracer l'erreur de reconstruction selon la taille du goulot, pour un autoencodeur dense et un profond.",
       "Ex 25.16 · fiche §25.4", thread="MNIST", tracks="C",
       body=r"""La fiche résume la leçon du §25.4 : il faut **assez de latents** et **assez de capacité**. Tu mesures les deux ensemble : l'erreur de test de l'autoencodeur dense (`make_dense_ae`) et du profond (`make_deep_ae`) pour des goulots de 2, 5, 10, 20 et 50 latents.

**Contrat** : `bottleneck_sweep_22(sizes, makers, epochs, seed=2522, x=None)` : pour chaque nom et chaque fonction du dictionnaire `makers`, et chaque taille de `sizes`, `torch.manual_seed(seed)`, puis `model = maker(size)`, `train_ae(model, x, epochs, seed=seed)` (avec `x = X_MNIST` si `x` vaut `None`), puis la MSE moyenne de reconstruction sur `XM_TEST` ; elle rend le dictionnaire `{(nom, taille): MSE}`.

La vérification compare ta fonction à un entraînement fait à la main sur un cas d'essai. La cellule suivante lance les dix entraînements (une quarantaine de secondes en mode rapide, 4 epochs) et trace les deux courbes.

Dans tes notes : comment l'erreur baisse-t-elle quand le goulot grandit (régulièrement, ou de moins en moins) ? Pour quelles tailles la profondeur aide-t-elle le plus ? Avec 4 epochs, quel modèle est désavantagé, et comment le vérifierais-tu ?""",
       todo=TODO_22, check=CHECK_22,
       solution=SOLUTION_22 + "\n\n\n" + CHECK_22,
       after=[("code", EXPERIMENT_22)]),

    Ex("25.23", "🔬", 3, 40, "L'autoencodeur linéaire face à l'ACP de scikit-learn",
       "vérifier par l'expérience qu'un autoencodeur linéaire retrouve le sous-espace de l'ACP, mais pas ses axes.",
       "∂ 25.8 · ch. 12 (ACP) · fiche §25.6", thread="MNIST", tracks="M, C",
       body=r"""Le ∂ 25.8 l'a démontré dans le plan : un autoencodeur linéaire entraîné à l'erreur quadratique retrouve le **sous-espace** de l'ACP, pas ses axes. Tu le vérifies sur MNIST, avec 16 latents. Deux sous-espaces se comparent par leurs **angles principaux** : tous nuls quand les sous-espaces sont les mêmes, quelle que soit la base qui les décrit (`scipy.linalg.subspace_angles`).

**Contrat**
- `fit_linear_ae_23(xc, k, epochs=40, lr=3e-3, batch_size=256, seed=2523)` : un autoencodeur **linéaire** sur les images centrées `xc` `(N, 784)` : encodeur `nn.Linear(784, k, bias=False)` et décodeur `nn.Linear(k, 784, bias=False)`, sans activation, créés dans cet ordre juste après `torch.manual_seed(seed)`, entraînés par `train_ae` avec l'erreur quadratique (`loss="mse"`) et les mêmes `epochs`, `lr`, `batch_size` et `seed` ; elle rend le poids du décodeur en tableau NumPy `(784, k)` : ses $k$ colonnes engendrent le sous-espace appris ;
- `subspace_angles_deg_23(A, B)` rend les angles principaux, en degrés, entre le sous-espace des colonnes de `A` et celui de `B`, du plus grand au plus petit.

La vérification essaie tes angles sur deux plans de l'espace, puis ta fonction d'entraînement sur un petit cas. La cellule suivante centre 10 000 images, calcule l'ACP à 16 composantes (`svd_solver="full"`), entraîne ton autoencodeur (quelques secondes), et compare : les erreurs de reconstruction, les angles entre les sous-espaces, les colonnes du décodeur face aux axes de l'ACP.

Dans tes notes : les deux sous-espaces sont-ils les mêmes ? Pourquoi les plus grands angles sont-ils ceux des dernières directions (pense aux valeurs propres voisines) ? Les colonnes du décodeur sont-elles les axes de l'ACP, et sont-elles orthogonales ? Pourquoi un test « mêmes axes » aurait-il échoué ?""",
       todo=TODO_23, check=CHECK_23, solution=SOLUTION_23 + "\n\n\n" + CHECK_23, after=[("code", EXPERIMENT_23)]),
])

# ---------------------------------------------------------------------------
# Part B: convolutional and denoising autoencoders (25.24 to 25.27)
# ---------------------------------------------------------------------------
GIVEN_B = r'''SIGMA_B = 0.3                                   # the noise of the Fashion-MNIST denoisers (25.26, 25.27, 25.39)
N_DENOISE = 10_000 if FAST_MODE else 60_000      # their training images
VAL_CLEAN_B = XF_TEST[:1000]                     # 1 000 validation clothes, and their noisy version (a fixed noise)
VAL_NOISY_B = np.clip(VAL_CLEAN_B + SIGMA_B * np.random.default_rng(2526).standard_normal(VAL_CLEAN_B.shape),
                      0, 1).astype(np.float32)


def layout(seq):
    """The layers of a nn.Sequential, written briefly: what the checks compare."""
    out = []
    for m in seq:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            extra = f", op={m.output_padding[0]}" if isinstance(m, nn.ConvTranspose2d) else ""
            out.append(f"{type(m).__name__}({m.in_channels}, {m.out_channels}, k={m.kernel_size[0]}, "
                       f"s={m.stride[0]}, p={m.padding[0]}{extra})")
        elif isinstance(m, nn.Linear):
            out.append(f"Linear({m.in_features}, {m.out_features})")
        elif isinstance(m, nn.Upsample):
            out.append(f"Upsample({m.scale_factor:g})")
        elif isinstance(m, nn.MaxPool2d):
            out.append(f"MaxPool2d({m.kernel_size})")
        elif isinstance(m, nn.Unflatten):
            out.append(f"Unflatten{tuple(m.unflattened_size)}")
        else:
            out.append(type(m).__name__)
    return out'''

TODO_24 = r'''def make_conv_ae_24(relu_code=False):
    """The book's convolutional autoencoder, for rows of 784 pixels:
    encoder: nn.Unflatten(1, (1, 28, 28)), Conv2d(1, 16, 3, padding=1), ReLU, MaxPool2d(2),
             Conv2d(16, 8, 3, padding=1), ReLU, MaxPool2d(2), Conv2d(8, 3, 3, padding=1)        -> a code (N, 3, 7, 7)
             (then a last ReLU on the code only if relu_code is True, as in the book)
    decoder: nn.Upsample(scale_factor=2), Conv2d(3, 16, 3, padding=1), ReLU, nn.Upsample(scale_factor=2),
             Conv2d(16, 1, 3, padding=1), Sigmoid, nn.Flatten()                              -> rows of 784 pixels
    Returns an AutoEncoder whose encoder and decoder are these two nn.Sequential."""
    raise NotImplementedError("make_conv_ae_24")'''

SOLUTION_24 = r'''def make_conv_ae_24(relu_code=False):
    """The book's convolutional autoencoder, for rows of 784 pixels:
    encoder: nn.Unflatten(1, (1, 28, 28)), Conv2d(1, 16, 3, padding=1), ReLU, MaxPool2d(2),
             Conv2d(16, 8, 3, padding=1), ReLU, MaxPool2d(2), Conv2d(8, 3, 3, padding=1)        -> a code (N, 3, 7, 7)
             (then a last ReLU on the code only if relu_code is True, as in the book)
    decoder: nn.Upsample(scale_factor=2), Conv2d(3, 16, 3, padding=1), ReLU, nn.Upsample(scale_factor=2),
             Conv2d(16, 1, 3, padding=1), Sigmoid, nn.Flatten()                              -> rows of 784 pixels
    Returns an AutoEncoder whose encoder and decoder are these two nn.Sequential."""
    encoder = nn.Sequential(nn.Unflatten(1, (1, 28, 28)),
                            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 16 x 14 x 14
                            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 8 x 7 x 7
                            nn.Conv2d(8, 3, 3, padding=1))                                  # 3 x 7 x 7: the code
    if relu_code:
        encoder.append(nn.ReLU())                                                            # the book's version
    decoder = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),   # 16 x 14 x 14
                            nn.Upsample(scale_factor=2), nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),  # 1 x 28 x 28
                            nn.Flatten())
    return AutoEncoder(encoder, decoder)'''

CHECK_24 = r'''with checked("25.24"):
    torch.manual_seed(0)
    model_24 = make_conv_ae_24()
    if returned("25.24", "make_conv_ae_24", model_24):
        want_enc_24 = ["Unflatten(1, 28, 28)", "Conv2d(1, 16, k=3, s=1, p=1)", "ReLU", "MaxPool2d(2)",
                       "Conv2d(16, 8, k=3, s=1, p=1)", "ReLU", "MaxPool2d(2)", "Conv2d(8, 3, k=3, s=1, p=1)"]
        want_dec_24 = ["Upsample(2)", "Conv2d(3, 16, k=3, s=1, p=1)", "ReLU", "Upsample(2)",
                       "Conv2d(16, 1, k=3, s=1, p=1)", "Sigmoid", "Flatten"]
        verdict("25.24", layout(model_24.encoder) == want_enc_24, "l'encodeur suit le contrat.",
                f"encodeur attendu {want_enc_24}, reçu {layout(model_24.encoder)}.")
        verdict("25.24", layout(make_conv_ae_24(relu_code=True).encoder) == want_enc_24 + ["ReLU"],
                "make_conv_ae_24(relu_code=True) ajoute la ReLU du livre sur le code.",
                "avec relu_code=True, l'encodeur doit finir par une ReLU de plus (la version du livre).")
        verdict("25.24", layout(model_24.decoder) == want_dec_24, "le décodeur suit le contrat.",
                f"décodeur attendu {want_dec_24}, reçu {layout(model_24.decoder)}.")
        code_24 = encode(model_24, X_MNIST[:5])
        verdict("25.24", code_24.shape == (5, 3, 7, 7) and reconstruct(model_24, X_MNIST[:5]).shape == (5, 784),
                f"un code {code_24.shape} pour 5 images, une sortie (5, 784).",
                f"attendu un code (5, 3, 7, 7) et une sortie (5, 784), reçu un code {code_24.shape}.")'''

EXPERIMENT_24 = r'''with checked("25.24"):
    EPOCHS_24 = 3 if FAST_MODE else 10
    convs_24 = {}
    for relu_24 in (False, True):
        torch.manual_seed(2524)
        model_24 = make_conv_ae_24(relu_code=relu_24)
        history_24 = train_ae(model_24, X_MNIST, EPOCHS_24, seed=2524, x_val=XM_TEST)
        convs_24[relu_24] = model_24
        print(f"convolutif{' avec la ReLU du livre sur le code' if relu_24 else ', code sans activation':40s} "
              f"{n_params(model_24):,d} paramètres · MSE de test par epoch : "
              + " ".join(f"{v:.4f}" for v in history_24["val_mse"]) + f" ({sum(history_24['seconds']):.0f} s)")
    print(f"AE_DEEP (20 latents, 8 epochs) : MSE de test {mse_per_image(reconstruct(AE_DEEP, XM_TEST), XM_TEST).mean():.4f}")
    codes_relu_24 = encode(convs_24[True], XM_TEST[:500])
    print(f"avec la ReLU : part des valeurs du code exactement nulles {np.mean(codes_relu_24 == 0):.0%}")
    conv_24 = convs_24[False]
    x0_24, x1_24 = XM_TEST[np.flatnonzero(YM_TEST == 1)[0]], XM_TEST[np.flatnonzero(YM_TEST == 0)[0]]
    c0_24, c1_24 = encode(conv_24, np.stack([x0_24, x1_24])).reshape(2, -1)        # codes flattened to 147 values
    path_24 = np.stack([(1 - t) * c0_24 + t * c1_24 for t in np.linspace(0, 1, 7)]).reshape(7, 3, 7, 7)
    show_rows([XM_TEST[:8], reconstruct(conv_24, XM_TEST[:8]), reconstruct(convs_24[True], XM_TEST[:8]),
               decode(conv_24, path_24.astype(np.float32))],
              ["originaux", "code sans activation", "ReLU sur le code", "d'un 1 à un 0"],
              f"{EPOCHS_24} epochs sur {len(X_MNIST)} images")'''

BUGGY_25 = r'''def train_denoiser_25(model, x, epochs, sigma=0.5, lr=1e-3, batch_size=128, seed=0, loss_fn=F.binary_cross_entropy):
    """Train `model` to remove a gaussian noise of standard deviation sigma from the images x (N, 784) in [0, 1]."""
    x = to_tensor(x)                                                   # (A)
    g = torch.Generator().manual_seed(seed)                            # (B)
    noise = sigma * torch.randn(x.shape, generator=g)                  # (C)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)            # (D)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=g)                    # (E)
        for i in range(0, len(x), batch_size):
            idx = order[i:i + batch_size]
            inputs = x[idx] + noise[idx]                               # (F)
            loss = loss_fn(model(inputs.to(DEVICE)), inputs.to(DEVICE))   # (G)
            optimizer.zero_grad()
            loss.backward()                                            # (H)
            optimizer.step()
    return model'''

GIVEN_25 = r'''# The code to debug (do not change this cell: copy the function into the next one to fix it)
''' + BUGGY_25 + r'''


class SpyModel_25(nn.Module):
    """A fake model for the checks: it remembers the inputs it receives, and returns a constant grey image."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().cpu().clone())
        return torch.sigmoid(self.w).expand_as(x)


def spy_run_25(train_fn, epochs=2, sigma=1.0):
    """Run train_fn on 8 distinct images with a SpyModel_25 and a loss that remembers its targets:
    returns (the images, the inputs seen, the targets seen), one tensor per mini-batch."""
    x = np.random.default_rng(25).uniform(0.2, 0.8, size=(8, 784)).astype(np.float32)
    targets = []

    def spy_loss(output, target):
        targets.append(target.detach().cpu().clone())
        return F.binary_cross_entropy(output, target.clamp(0, 1))

    spy = SpyModel_25()
    train_fn(spy, x, epochs, sigma=sigma, batch_size=4, seed=3, loss_fn=spy_loss)
    return torch.from_numpy(x), spy.inputs, targets


def diagnose_25(train_fn):
    """The three diagnostics of 25.25, in order: the target, the noise of each epoch, the clipping.
    Prints ✅ or ❌ for each one, stops at the first ❌ (⏳ for the next ones)."""
    x, inputs, targets = spy_run_25(train_fn)
    points = [
        ("la cible est l'image propre",
         all(any(torch.equal(row, clean) for clean in x) for batch in targets for row in batch),
         "la cible d'un mini-batch doit être l'image propre, pas l'entrée bruitée."),
        ("un bruit neuf à chaque epoch",
         not torch.equal(torch.cat(inputs[:len(inputs) // 2]).sort(dim=0).values,
                         torch.cat(inputs[len(inputs) // 2:]).sort(dim=0).values),
         "une même image reçoit le même bruit à chaque epoch : le réseau peut apprendre le bruit par cœur."),
        ("des entrées dans [0, 1]",
         all(float(b.min()) >= 0 and float(b.max()) <= 1 for b in inputs),
         "les entrées bruitées sortent de [0, 1] : le réseau voit des pixels qui n'existent pas (écrête-les)."),
    ]
    for k, (name, ok, why) in enumerate(points, start=1):
        if ok:
            print(f"✅ Ex 25.25 : point {k}, {name}.")
        else:
            print(f"❌ Ex 25.25 : point {k} : {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex 25.25 : point {later} : corrige d'abord le point {k}.")
            return False
    return True


print("Le code fautif, essayé par les diagnostics :")
_ = diagnose_25(train_denoiser_25)'''

TODO_25 = r'''faulty_25 = ...   # the letters of the faulty lines, e.g. "AB"


def train_denoiser_fixed_25(model, x, epochs, sigma=0.5, lr=1e-3, batch_size=128, seed=0,
                            loss_fn=F.binary_cross_entropy):
    """The fixed version of train_denoiser_25: same arguments, same mini-batches (one torch.Generator from `seed`,
    a torch.randperm(N, generator=g) at the start of each epoch). For each mini-batch: a new noise
    sigma * torch.randn(batch.shape, generator=g), noisy inputs clipped to [0, 1] (torch.clamp), and the loss
    loss_fn(model(noisy inputs), clean batch). Returns the model."""
    raise NotImplementedError("train_denoiser_fixed_25")'''

SOLUTION_25 = r'''faulty_25 = "CFG"


def train_denoiser_fixed_25(model, x, epochs, sigma=0.5, lr=1e-3, batch_size=128, seed=0,
                            loss_fn=F.binary_cross_entropy):
    """The fixed version of train_denoiser_25: same arguments, same mini-batches (one torch.Generator from `seed`,
    a torch.randperm(N, generator=g) at the start of each epoch). For each mini-batch: a new noise
    sigma * torch.randn(batch.shape, generator=g), noisy inputs clipped to [0, 1] (torch.clamp), and the loss
    loss_fn(model(noisy inputs), clean batch). Returns the model."""
    x = to_tensor(x)
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch_size):
            clean = x[order[i:i + batch_size]]
            inputs = torch.clamp(clean + sigma * torch.randn(clean.shape, generator=g), 0, 1)  # a new noise, clipped
            loss = loss_fn(model(inputs.to(DEVICE)), clean.to(DEVICE))                        # the target is clean
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model'''

CHECK_25 = r'''wb.check("25.25", faulty_25)
with checked("25.25"):
    if written(lambda: train_denoiser_fixed_25(SpyModel_25(), np.zeros((1, 784), np.float32), 0)):
        if diagnose_25(train_denoiser_fixed_25):
            torch.manual_seed(2525)
            small_25 = make_deep_ae(32, hidden=(256,))
            train_denoiser_fixed_25(small_25.to(DEVICE), X_MNIST[:5000], 3 if FAST_MODE else 10, sigma=0.5, seed=2525)
            noisy_25 = np.clip(XM_TEST[:8] + 0.5 * np.random.default_rng(25).standard_normal((8, 784)), 0, 1).astype(np.float32)
            show_rows([XM_TEST[:8], noisy_25, reconstruct(small_25, noisy_25)], ["propres", "bruitées (σ = 0,5)", "débruitées"],
                      "un petit débruiteur dense entraîné par ta fonction")
    else:
        print("⏳ Ex 25.25 : les diagnostics attendent ta fonction corrigée.")'''

RECORD_25 = ('wb.record("25.25", "CFG", mistakes=' + bug_letters_mistakes("CFG", "ABDEH", {
    "C": "un des diagnostics compare le bruit d'une même image d'une epoch à l'autre",
    "F": "un des diagnostics regarde les valeurs des entrées bruitées",
    "G": "un des diagnostics regarde la cible de la loss"}) + ")")

GIVEN_DENOISER = r'''# The reference version of your fixed loop (25.25), with a history: the denoisers of 25.26, 25.27 and 25.39 use it
def train_denoiser(model, x, epochs, sigma=SIGMA_B, lr=1e-3, batch_size=128, seed=0, x_val=None, verbose=False):
    """Train the autoencoder `model` in place (on DEVICE) to turn noisy images back into the clean images x (N, 784):
    a new gaussian noise of standard deviation sigma for each mini-batch, noisy inputs clipped to [0, 1], the binary
    cross-entropy against the CLEAN batch, Adam(lr). x_val = (noisy, clean) adds "val_psnr" to the history."""
    model.to(DEVICE)
    x = to_tensor(x)
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "val_psnr": [], "seconds": []}
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        total = 0.0
        order = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch_size):
            clean = x[order[i:i + batch_size]]
            inputs = torch.clamp(clean + sigma * torch.randn(clean.shape, generator=g), 0, 1)
            loss = F.binary_cross_entropy(model(inputs.to(DEVICE)), clean.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(clean)
        history["loss"].append(total / len(x))
        history["seconds"].append(time.perf_counter() - start)
        if x_val is not None:
            history["val_psnr"].append(psnr(x_val[1], reconstruct(model, x_val[0])))
        if verbose:
            print(f"epoch {epoch:2d} · loss {history['loss'][-1]:.4f}" +
                  (f" · val PSNR {history['val_psnr'][-1]:.2f} dB" if x_val is not None else ""))
    return history


def psnr(clean, images):
    """The mean PSNR (dB) of images against clean (pixels in [0, 1]): the mean over the images of 10 log10(1 / MSE)."""
    return float(np.mean(10 * np.log10(1.0 / np.maximum(mse_per_image(images, clean), 1e-12))))'''

TODO_26 = r'''def make_denoiser_26(channels=32):
    """A convolutional denoiser for rows of 784 pixels, with c = channels:
    encoder: nn.Unflatten(1, (1, 28, 28)), Conv2d(1, c, 3, padding=1), ReLU, MaxPool2d(2),
             Conv2d(c, c, 3, padding=1), ReLU, MaxPool2d(2)                         -> (N, c, 7, 7)
    decoder: Conv2d(c, c, 3, padding=1), ReLU, nn.Upsample(scale_factor=2), Conv2d(c, c, 3, padding=1), ReLU,
             nn.Upsample(scale_factor=2), Conv2d(c, 1, 3, padding=1), Sigmoid, nn.Flatten()
    Returns an AutoEncoder."""
    raise NotImplementedError("make_denoiser_26")


def psnr_26(clean, images):
    """The mean PSNR in decibels of `images` against `clean` (both (N, 784), pixels in [0, 1]): for each image,
    10 log10(1 / MSE), then the mean over the images. A float."""
    raise NotImplementedError("psnr_26")'''

SOLUTION_26 = r'''def make_denoiser_26(channels=32):
    """A convolutional denoiser for rows of 784 pixels, with c = channels:
    encoder: nn.Unflatten(1, (1, 28, 28)), Conv2d(1, c, 3, padding=1), ReLU, MaxPool2d(2),
             Conv2d(c, c, 3, padding=1), ReLU, MaxPool2d(2)                         -> (N, c, 7, 7)
    decoder: Conv2d(c, c, 3, padding=1), ReLU, nn.Upsample(scale_factor=2), Conv2d(c, c, 3, padding=1), ReLU,
             nn.Upsample(scale_factor=2), Conv2d(c, 1, 3, padding=1), Sigmoid, nn.Flatten()
    Returns an AutoEncoder."""
    c = channels
    encoder = nn.Sequential(nn.Unflatten(1, (1, 28, 28)),
                            nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
    decoder = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
                            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2),
                            nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid(), nn.Flatten())
    return AutoEncoder(encoder, decoder)


def psnr_26(clean, images):
    """The mean PSNR in decibels of `images` against `clean` (both (N, 784), pixels in [0, 1]): for each image,
    10 log10(1 / MSE), then the mean over the images. A float."""
    mse = ((np.asarray(images, dtype=np.float64) - np.asarray(clean, dtype=np.float64)) ** 2).mean(axis=1)
    return float(np.mean(10 * np.log10(1.0 / mse)))'''

CHECK_26 = r'''with checked("25.26"):
    torch.manual_seed(0)
    model_26 = make_denoiser_26(8)
    if returned("25.26", "make_denoiser_26", model_26):
        want_26 = (["Unflatten(1, 28, 28)", "Conv2d(1, 8, k=3, s=1, p=1)", "ReLU", "MaxPool2d(2)",
                    "Conv2d(8, 8, k=3, s=1, p=1)", "ReLU", "MaxPool2d(2)"],
                   ["Conv2d(8, 8, k=3, s=1, p=1)", "ReLU", "Upsample(2)", "Conv2d(8, 8, k=3, s=1, p=1)", "ReLU",
                    "Upsample(2)", "Conv2d(8, 1, k=3, s=1, p=1)", "Sigmoid", "Flatten"])
        verdict("25.26", (layout(model_26.encoder), layout(model_26.decoder)) == want_26,
                "make_denoiser_26(8) suit le contrat.",
                f"attendu {want_26}, reçu {(layout(model_26.encoder), layout(model_26.decoder))}.")
    toy_clean_26 = np.zeros((2, 784))
    toy_26 = toy_clean_26 + np.array([[0.1], [0.01]])          # MSE 0.01 then 0.0001: 20 dB then 40 dB
    value_26 = psnr_26(toy_clean_26, toy_26)
    if returned("25.26", "psnr_26", value_26):
        same("25.26", "psnr_26 de deux images d'erreurs 0,1 et 0,01 sur chaque pixel (20 et 40 dB)", [value_26], [30.0],
             rtol=1e-9)'''

EXPERIMENT_26 = r'''with checked("25.26"):
    EPOCHS_26 = 3 if FAST_MODE else 10
    torch.manual_seed(2526)
    denoiser_26 = make_denoiser_26(32)
    history_26 = train_denoiser(denoiser_26, X_FASHION[:N_DENOISE], EPOCHS_26, lr=3e-3, seed=2526)
    out_26 = reconstruct(denoiser_26, VAL_NOISY_B)
    print(f"{n_params(denoiser_26):,d} paramètres · {sum(history_26['seconds']):.0f} s · PSNR des images bruitées "
          f"{psnr_26(VAL_CLEAN_B, VAL_NOISY_B):.2f} dB · débruitées {psnr_26(VAL_CLEAN_B, out_26):.2f} dB")
    verdict("25.26", psnr_26(VAL_CLEAN_B, out_26) > psnr_26(VAL_CLEAN_B, VAL_NOISY_B) + 5,
            "le débruiteur gagne plus de 5 dB.", "le débruiteur devrait gagner plus de 5 dB : vérifie ton architecture.")
    show_rows([VAL_CLEAN_B[:8], VAL_NOISY_B[:8], out_26[:8]], ["propres", f"bruitées (σ = {SIGMA_B})", "débruitées"])'''

TODO_27 = r'''def make_strided_27(channels=32):
    """The same denoiser without pooling nor upsampling, with c = channels:
    encoder: nn.Unflatten(1, (1, 28, 28)), Conv2d(1, c, 3, stride=2, padding=1), ReLU,
             Conv2d(c, c, 3, stride=2, padding=1), ReLU                                  -> (N, c, 7, 7)
    decoder: ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), ReLU,
             ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), ReLU,
             Conv2d(c, 1, 3, padding=1), Sigmoid, nn.Flatten()
    Returns an AutoEncoder."""
    raise NotImplementedError("make_strided_27")


def compare_27(makers, epochs, seed=2527):
    """For each name, maker of the dict `makers`: torch.manual_seed(seed), model = maker(), train_denoiser(model,
    X_FASHION[:N_DENOISE], epochs, seed=seed). Returns {name: {"params": n_params(model), "sec_per_epoch": the
    median of history["seconds"], "psnr": the mean PSNR of model on VAL_NOISY_B against VAL_CLEAN_B}}."""
    raise NotImplementedError("compare_27")'''

SOLUTION_27 = r'''def make_strided_27(channels=32):
    """The same denoiser without pooling nor upsampling, with c = channels:
    encoder: nn.Unflatten(1, (1, 28, 28)), Conv2d(1, c, 3, stride=2, padding=1), ReLU,
             Conv2d(c, c, 3, stride=2, padding=1), ReLU                                  -> (N, c, 7, 7)
    decoder: ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), ReLU,
             ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), ReLU,
             Conv2d(c, 1, 3, padding=1), Sigmoid, nn.Flatten()
    Returns an AutoEncoder."""
    c = channels
    encoder = nn.Sequential(nn.Unflatten(1, (1, 28, 28)),
                            nn.Conv2d(1, c, 3, stride=2, padding=1), nn.ReLU(),        # 28 -> 14
                            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.ReLU())        # 14 -> 7
    decoder = nn.Sequential(nn.ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 7 -> 14
                            nn.ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # 14 -> 28
                            nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid(), nn.Flatten())
    return AutoEncoder(encoder, decoder)


def compare_27(makers, epochs, seed=2527):
    """For each name, maker of the dict `makers`: torch.manual_seed(seed), model = maker(), train_denoiser(model,
    X_FASHION[:N_DENOISE], epochs, seed=seed). Returns {name: {"params": n_params(model), "sec_per_epoch": the
    median of history["seconds"], "psnr": the mean PSNR of model on VAL_NOISY_B against VAL_CLEAN_B}}."""
    results = {}
    for name, maker in makers.items():
        torch.manual_seed(seed)
        model = maker()
        history = train_denoiser(model, X_FASHION[:N_DENOISE], epochs, seed=seed)
        results[name] = {"params": n_params(model), "sec_per_epoch": float(np.median(history["seconds"])),
                         "psnr": psnr(VAL_CLEAN_B, reconstruct(model, VAL_NOISY_B))}
    return results'''

CHECK_27 = r'''with checked("25.27"):
    torch.manual_seed(0)
    model_27 = make_strided_27(8)
    if returned("25.27", "make_strided_27", model_27):
        want_27 = (["Unflatten(1, 28, 28)", "Conv2d(1, 8, k=3, s=2, p=1)", "ReLU", "Conv2d(8, 8, k=3, s=2, p=1)", "ReLU"],
                   ["ConvTranspose2d(8, 8, k=3, s=2, p=1, op=1)", "ReLU", "ConvTranspose2d(8, 8, k=3, s=2, p=1, op=1)",
                    "ReLU", "Conv2d(8, 1, k=3, s=1, p=1)", "Sigmoid", "Flatten"])
        verdict("25.27", (layout(model_27.encoder), layout(model_27.decoder)) == want_27,
                "make_strided_27(8) suit le contrat.",
                f"attendu {want_27}, reçu {(layout(model_27.encoder), layout(model_27.decoder))}.")
        verdict("25.27", encode(model_27, X_FASHION[:2]).shape == (2, 8, 7, 7) and reconstruct(model_27, X_FASHION[:2]).shape == (2, 784),
                "un code (2, 8, 7, 7) et une sortie (2, 784).", "le code doit être (2, 8, 7, 7) et la sortie (2, 784).")
    tiny_27 = compare_27({"petit": lambda: make_strided_27(4)}, 1, seed=3)
    if returned("25.27", "compare_27", tiny_27):
        torch.manual_seed(3)
        ref_27 = make_strided_27(4)
        train_denoiser(ref_27, X_FASHION[:N_DENOISE], 1, seed=3)
        row_27 = tiny_27.get("petit", {}) if isinstance(tiny_27, dict) else {}
        verdict("25.27", set(row_27) == {"params", "sec_per_epoch", "psnr"} and row_27["params"] == n_params(ref_27)
                and abs(row_27["psnr"] - psnr(VAL_CLEAN_B, reconstruct(ref_27, VAL_NOISY_B))) < 1e-3,
                "compare_27 sur un cas d'essai : les paramètres et le PSNR attendus.",
                f"compare_27 doit rendre {{nom: {{'params', 'sec_per_epoch', 'psnr'}}}} avec les valeurs d'un entraînement "
                f"fait comme le contrat, reçu {tiny_27!r:.200}.")'''

EXPERIMENT_27 = r'''with checked("25.27"):
    EPOCHS_27 = 2 if FAST_MODE else 10                 # 🚀 the full mode trains for long: a GPU helps
    table_27 = compare_27({"pooling + sur-échantillonnage": lambda: make_denoiser_26(32),
                           "convolutions à pas": lambda: make_strided_27(32)}, EPOCHS_27)
    for name_27, row_27 in table_27.items():
        print(f"{name_27:30s} {row_27['params']:7,d} paramètres · {row_27['sec_per_epoch']:5.2f} s par epoch · "
              f"PSNR {row_27['psnr']:.2f} dB")'''

PART_B = Part("B", "Autoencodeurs convolutifs et débruiteurs",
              "Tu construis l'autoencodeur convolutif du livre (25.24), tu répares un débruiteur qui apprenait le "
              "bruit (25.25), tu débruites Fashion-MNIST (25.26), puis tu remplaces pooling et sur-échantillonnage "
              "par des convolutions à pas (25.27). La cellule suivante fixe le bruit des débruiteurs de vêtements "
              "($\\sigma = 0{,}3$), leurs images d'entraînement (`N_DENOISE` : 10 000 en mode rapide) et 1 000 "
              "vêtements de validation avec leur version bruitée (un bruit fixé), et donne `layout`, qui écrit "
              "brièvement les couches d'un `nn.Sequential` (les vérifications comparent ces listes).",
              given=GIVEN_B, exercises=[
    Ex("25.24", "📦", 2, 30, "Autoencodeur convolutif à 147 latents",
       "écrire l'autoencodeur convolutif du livre et comparer son code de 147 valeurs à celui de l'autoencodeur dense.",
       "✏️ 25.3 · ch. 21 (convolutions, pooling) · fiche §25.7, §25.7.1, §25.7.2", thread="MNIST", tracks="R, C",
       body=r"""Le ✏️ 25.3 a suivi les formes de l'autoencodeur convolutif du livre ; tu l'écris. Pour garder la convention du notebook (des lignes de 784 pixels), l'encodeur commence par `nn.Unflatten(1, (1, 28, 28))` et le décodeur finit par `nn.Flatten()`.

Une différence avec le livre : il met une ReLU après la dernière convolution de l'encodeur, sur le code ; ton code n'a pas d'activation, sauf si tu demandes la version du livre.

**Contrat** : `make_conv_ae_24(relu_code=False)` rend un `AutoEncoder` dont l'encodeur est `nn.Unflatten(1, (1, 28, 28))`, `Conv2d(1, 16, 3, padding=1)`, ReLU, `MaxPool2d(2)`, `Conv2d(16, 8, 3, padding=1)`, ReLU, `MaxPool2d(2)`, `Conv2d(8, 3, 3, padding=1)`, suivi d'une ReLU seulement si `relu_code` vaut `True`, et le décodeur `nn.Upsample(scale_factor=2)`, `Conv2d(3, 16, 3, padding=1)`, ReLU, `nn.Upsample(scale_factor=2)`, `Conv2d(16, 1, 3, padding=1)`, sigmoïde, `nn.Flatten()`.

La vérification compare tes couches au contrat et regarde les formes du code et de la sortie. La cellule suivante entraîne les deux versions, sans et avec la ReLU du livre (3 epochs chacune en mode rapide, une trentaine de secondes en tout), compare leurs erreurs à celle de `AE_DEEP`, et décode une ligne droite entre les codes d'un 1 et d'un 0.

Dans tes notes : combien de paramètres a ce modèle, face au 1,1 million de `AE_DEEP`, et son code est-il plus petit ou plus grand ? Que s'est-il passé avec la ReLU sur le code (regarde la part des valeurs nulles, et pense au gradient d'une ReLU en dessous de 0) ? Que donne le mélange des codes : une transformation, ou un fondu (fiche §25.7.1) ?""",
       todo=TODO_24, check=CHECK_24, solution=SOLUTION_24 + "\n\n\n" + CHECK_24, after=[("code", EXPERIMENT_24)]),

    Ex("25.25", "🐛", 2, 25, "Le débruiteur qui apprenait le bruit",
       "trouver les trois erreurs d'une boucle d'entraînement de débruiteur, et la corriger.",
       "Ex 25.24 · fiche §25.8", thread="MNIST", tracks="C",
       body=r"""Un collègue a écrit la boucle d'entraînement d'un débruiteur (la cellule suivante), et ses images « débruitées » restent pleines de neige. Trois lignes sont fausses. Les diagnostics de la cellule donnée l'essaient avec un faux modèle qui note ce qu'il reçoit (pas d'entraînement réel : ils sont instantanés), dans cet ordre : 1) la cible est-elle l'image propre ? 2) une même image reçoit-elle un bruit neuf à chaque epoch ? 3) les entrées bruitées restent-elles dans $[0, 1]$ ?

1. Écris dans `faulty_25` les lettres des **lignes fautives** (une chaîne comme `"AB"`).
2. Écris `train_denoiser_fixed_25`, la version corrigée : mêmes arguments ; un seul `torch.Generator` créé depuis `seed`, un `torch.randperm(N, generator=g)` au début de chaque epoch ; pour chaque mini-batch, un bruit **neuf** `sigma * torch.randn(batch.shape, generator=g)`, des entrées bruitées **écrêtées** dans $[0, 1]$ (`torch.clamp`), et la loss `loss_fn(model(entrées bruitées), batch propre)` ; la fonction rend le modèle.

La vérification lance les trois diagnostics sur ta version ; s'ils passent, elle entraîne un petit débruiteur dense avec ta fonction et montre le résultat.

Dans tes notes : pour chaque erreur, qu'apprendrait le réseau, et comment le verrais-tu sur les images ou sur les courbes ? Laquelle est la plus grave ?""",
       given=GIVEN_25, todo=TODO_25, check=CHECK_25, solution=SOLUTION_25 + "\n\n\n" + CHECK_25, record=RECORD_25,
       after=[("md", "La cellule suivante donne la version de référence de la boucle corrigée, `train_denoiser`, et "
                     "`psnr`, qui servent aux exercices suivants."), ("code", GIVEN_DENOISER)]),

    Ex("25.26", "📦", 2, 30, "Autoencodeur débruiteur (bruit gaussien, écrêtage)",
       "construire un débruiteur convolutif pour Fashion-MNIST et mesurer son gain en PSNR.",
       "Ex 25.24 · Ex 25.25 · fiche §25.8", thread="Fashion-MNIST", tracks="R, C",
       body=r"""Le débruiteur du livre (celui de F. Chollet) garde 32 filtres à chaque étage : deux convolutions suivies d'un max pooling à l'encodage, et leur miroir au décodage. Tu l'entraînes sur des vêtements avec un bruit gaussien d'écart-type 0,3, et tu le mesures en **PSNR**.

**Contrat**
- `make_denoiser_26(channels=32)` rend un `AutoEncoder` ; avec $c$ = `channels`, l'encodeur est `nn.Unflatten(1, (1, 28, 28))`, `Conv2d(1, c, 3, padding=1)`, ReLU, `MaxPool2d(2)`, `Conv2d(c, c, 3, padding=1)`, ReLU, `MaxPool2d(2)`, et le décodeur `Conv2d(c, c, 3, padding=1)`, ReLU, `nn.Upsample(scale_factor=2)`, `Conv2d(c, c, 3, padding=1)`, ReLU, `nn.Upsample(scale_factor=2)`, `Conv2d(c, 1, 3, padding=1)`, sigmoïde, `nn.Flatten()` ;
- `psnr_26(clean, images)` rend le PSNR moyen en décibels de `images` face à `clean` (deux tableaux `(N, 784)`, pixels dans $[0, 1]$) : pour chaque image $10 \log_{10}(1/\text{MSE})$, puis la moyenne sur les images.

La vérification essaie ton architecture avec 8 filtres et ton PSNR sur deux images d'erreurs connues. La cellule suivante entraîne ton débruiteur avec `train_denoiser` (3 epochs sur 10 000 vêtements en mode rapide, une trentaine de secondes, avec un learning rate de 0,003), compare le PSNR des images bruitées et débruitées, et les montre.

Dans tes notes : combien de décibels gagne le débruiteur, et à quoi cela correspond-il en MSE (fiche : diviser la MSE par 10 ajoute 10 dB) ? Quels détails perd-il ? Pourquoi moyenne-t-on le PSNR image par image plutôt que de calculer celui de la MSE moyenne ?""",
       todo=TODO_26, check=CHECK_26, solution=SOLUTION_26 + "\n\n\n" + CHECK_26, after=[("code", EXPERIMENT_26)]),

    Ex("25.27", "🔬", 3, 35, "Sur-échantillonnage explicite ou convolutions à pas : temps et qualité",
       "comparer, à nombre de filtres égal, un débruiteur à pooling et sur-échantillonnage et un débruiteur à convolutions à pas.",
       "Ex 25.26 · ✏️ 25.4 · fiche §25.8 (et son encadré ⚠️)", thread="Fashion-MNIST", tracks="C",
       body=r"""Le livre remplace le pooling et le sur-échantillonnage de son débruiteur par des convolutions à pas : un `Conv2d` de stride 2 réduit l'image, un `ConvTranspose2d` de stride 2 l'agrandit (le ✏️ 25.4 a calculé ses tailles). Tu mesures ce que cela change, au même nombre de filtres.

**Contrat**
- `make_strided_27(channels=32)` rend un `AutoEncoder` ; l'encodeur est `nn.Unflatten(1, (1, 28, 28))`, `Conv2d(1, c, 3, stride=2, padding=1)`, ReLU, `Conv2d(c, c, 3, stride=2, padding=1)`, ReLU ; le décodeur `ConvTranspose2d(c, c, 3, stride=2, padding=1, output_padding=1)`, ReLU, le même encore, ReLU, puis `Conv2d(c, 1, 3, padding=1)`, sigmoïde, `nn.Flatten()` ;
- `compare_27(makers, epochs, seed=2527)` : pour chaque nom et chaque fonction du dictionnaire `makers`, `torch.manual_seed(seed)`, `model = maker()`, `train_denoiser(model, X_FASHION[:N_DENOISE], epochs, seed=seed)` ; elle rend `{nom: {"params": ..., "sec_per_epoch": la médiane des durées, "psnr": le PSNR moyen sur VAL_NOISY_B face à VAL_CLEAN_B}}`.

La vérification compare tes couches au contrat, puis `compare_27` à un entraînement fait à la main. La cellule suivante entraîne les deux débruiteurs (2 epochs en mode rapide, une trentaine de secondes ; 🚀 le mode complet est long sur un CPU).

Dans tes notes : lequel va le plus vite, et pourquoi (compte les positions où chaque convolution calcule) ? Lequel débruite le mieux à nombre d'epochs égal ? Une convolution transposée peut laisser un motif en damier : le vois-tu ici ?""",
       todo=TODO_27, check=CHECK_27, solution=SOLUTION_27 + "\n\n\n" + CHECK_27, after=[("code", EXPERIMENT_27)]),
])

# ---------------------------------------------------------------------------
# Part C: the VAE (25.28 to 25.39)
# ---------------------------------------------------------------------------
_HISTORY_FILE = Path(__file__).with_name("history_ch25.json")       # tools/chapters/history_ch25.py writes it
HISTORY = (json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists()
           else {"collapse34": None, "calib": {"goal_39": 0.0}})

CHECK_28 = r'''with checked("25.28"):
    V = mylearn.vae
    mu_28 = np.array([[0.5, -1.0, 2.0], [0.0, 3.0, -0.5]])
    lv_28 = np.array([[0.0, 1.0, -2.0], [0.5, -1.0, 2.0]])
    z_28, eps_28 = V.reparameterize(mu_28, lv_28, rng=np.random.default_rng(0))
    same("25.28", "z rendu par reparameterize : mu + exp(0.5 log_var) * eps, avec son eps",
         z_28, mu_28 + np.exp(0.5 * lv_28) * np.asarray(eps_28))
    same("25.28", "la même graine donne le même eps", V.reparameterize(mu_28, lv_28, rng=0)[1],
         V.reparameterize(mu_28, lv_28, rng=0)[1])
    big_28 = V.reparameterize(np.full((200_000, 1), 2.0), np.full((200_000, 1), np.log(9.0)), rng=1)[0]
    verdict("25.28", abs(float(np.mean(big_28)) - 2.0) < 0.03 and abs(float(np.var(big_28)) - 9.0) < 0.15,
            f"200 000 tirages de N(2, 9) : moyenne {fr(float(np.mean(big_28)))}, variance {fr(float(np.var(big_28)))}.",
            "200 000 tirages avec mu = 2 et log_var = ln 9 doivent avoir une moyenne proche de 2 et une variance proche "
            f"de 9 : reçu {float(np.mean(big_28)):.3f} et {float(np.var(big_28)):.3f}.")
    dz_28 = np.array([[1.0, -2.0, 0.5], [0.3, 0.0, -1.0]])
    mu_t = torch.tensor(mu_28, requires_grad=True)
    lv_t = torch.tensor(lv_28, requires_grad=True)
    (torch.tensor(dz_28) * (mu_t + torch.exp(0.5 * lv_t) * torch.tensor(np.asarray(eps_28)))).sum().backward()
    dmu_28, dlv_28 = V.reparameterize_backward(dz_28, eps_28, lv_28)
    same("25.28", "dmu de reparameterize_backward, contre torch.autograd", dmu_28, mu_t.grad.numpy())
    same("25.28", "dlog_var de reparameterize_backward, contre torch.autograd", dlv_28, lv_t.grad.numpy())'''

CHECK_29 = r'''with checked("25.29"):
    V = mylearn.vae
    mu_29 = np.array([[0.5, -1.0, 2.0], [0.0, 3.0, -0.5]])
    lv_29 = np.array([[0.0, 1.0, -2.0], [0.5, -1.0, 2.0]])
    mu_t = torch.tensor(mu_29, requires_grad=True)
    lv_t = torch.tensor(lv_29, requires_grad=True)
    kl_t = torch.distributions.kl_divergence(torch.distributions.Normal(mu_t, torch.exp(0.5 * lv_t)),
                                             torch.distributions.Normal(0.0, 1.0)).sum(dim=1)
    same("25.29", "gaussian_kl, contre torch.distributions.kl_divergence (une valeur par exemple)",
         V.gaussian_kl(mu_29, lv_29), kl_t.detach().numpy())
    kl_t.sum().backward()
    gmu_29, glv_29 = V.gaussian_kl_grad(mu_29, lv_29)
    same("25.29", "gaussian_kl_grad : d KL / d mu, contre torch.autograd", gmu_29, mu_t.grad.numpy())
    same("25.29", "gaussian_kl_grad : d KL / d log_var, contre torch.autograd", glv_29, lv_t.grad.numpy())
    same("25.29", "la KL de N(0, 1) contre elle-même", V.gaussian_kl(np.zeros((1, 4)), np.zeros((1, 4))), [0.0])'''

TODO_30 = r'''class VAE30(nn.Module):
    """A VAE for rows of 784 pixels.
    encoder: nn.Sequential(Linear(784, hidden[0]), ReLU, Linear(hidden[0], hidden[1]), ReLU, ...);
    to_mu and to_log_var: two nn.Linear(hidden[-1], n_latent) on the output of the encoder;
    decoder: nn.Sequential(Linear(n_latent, hidden[-1]), ReLU, ..., Linear(hidden[0], 784)): LOGITS, no sigmoid.
    Attributes n_latent and hidden (a tuple) keep the two arguments."""

    def __init__(self, n_latent=20, hidden=(512, 256)):
        super().__init__()
        raise NotImplementedError("VAE30.__init__")

    def encode(self, x):
        """(mu, log_var), both (N, n_latent)."""
        raise NotImplementedError("VAE30.encode")

    def forward(self, x):
        """(logits, mu, log_var): z = mu + exp(0.5 log_var) * eps with eps = torch.randn_like(mu), decoded."""
        raise NotImplementedError("VAE30.forward")


def vae_loss_30(x, logits, mu, log_var, beta=1.0):
    """(loss, rec, kl), three tensors: rec = the binary cross-entropy (from the logits) SUMMED over the 784 pixels,
    then averaged over the batch; kl = the KL of N(mu, exp(log_var)) from N(0, I), summed over the latents, then
    averaged over the batch; loss = rec + beta * kl."""
    raise NotImplementedError("vae_loss_30")'''

SOLUTION_30 = r'''class VAE30(nn.Module):
    """A VAE for rows of 784 pixels.
    encoder: nn.Sequential(Linear(784, hidden[0]), ReLU, Linear(hidden[0], hidden[1]), ReLU, ...);
    to_mu and to_log_var: two nn.Linear(hidden[-1], n_latent) on the output of the encoder;
    decoder: nn.Sequential(Linear(n_latent, hidden[-1]), ReLU, ..., Linear(hidden[0], 784)): LOGITS, no sigmoid.
    Attributes n_latent and hidden (a tuple) keep the two arguments."""

    def __init__(self, n_latent=20, hidden=(512, 256)):
        super().__init__()
        self.n_latent, self.hidden = n_latent, tuple(hidden)
        layers, n_in = [], 784
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
            n_in = n_out
        self.encoder = nn.Sequential(*layers)
        self.to_mu = nn.Linear(n_in, n_latent)
        self.to_log_var = nn.Linear(n_in, n_latent)
        layers, n_in = [], n_latent
        for n_out in reversed(hidden):
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
            n_in = n_out
        self.decoder = nn.Sequential(*layers, nn.Linear(n_in, 784))

    def encode(self, x):
        """(mu, log_var), both (N, n_latent)."""
        h = self.encoder(x)
        return self.to_mu(h), self.to_log_var(h)

    def forward(self, x):
        """(logits, mu, log_var): z = mu + exp(0.5 log_var) * eps with eps = torch.randn_like(mu), decoded."""
        mu, log_var = self.encode(x)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)        # the reparameterization trick
        return self.decoder(z), mu, log_var


def vae_loss_30(x, logits, mu, log_var, beta=1.0):
    """(loss, rec, kl), three tensors: rec = the binary cross-entropy (from the logits) SUMMED over the 784 pixels,
    then averaged over the batch; kl = the KL of N(mu, exp(log_var)) from N(0, I), summed over the latents, then
    averaged over the batch; loss = rec + beta * kl."""
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()
    kl = 0.5 * (mu ** 2 + torch.exp(log_var) - 1 - log_var).sum(dim=1).mean()
    return rec + beta * kl, rec, kl'''

CHECK_30 = r'''with checked("25.30"):
    torch.manual_seed(0)
    vae_30 = VAE30(20)
    if returned("25.30", "VAE30(20)", vae_30):
        want_30 = (["Linear(784, 512)", "ReLU", "Linear(512, 256)", "ReLU"], "Linear(256, 20)",
                   ["Linear(20, 256)", "ReLU", "Linear(256, 512)", "ReLU", "Linear(512, 784)"])
        got_30 = (layout(vae_30.encoder), layout([vae_30.to_mu])[0], layout(vae_30.decoder))
        verdict("25.30", got_30 == want_30 and layout([vae_30.to_log_var]) == ["Linear(256, 20)"]
                and vae_30.n_latent == 20 and tuple(vae_30.hidden) == (512, 256),
                "VAE30(20) : encodeur, to_mu, to_log_var et décodeur (des logits) suivent le contrat.",
                f"attendu {want_30} (et to_log_var comme to_mu), reçu {got_30}.")
        x_30 = to_tensor(X_MNIST[:5])
        torch.manual_seed(1)
        logits_30, mu_30, lv_30 = vae_30(x_30)
        torch.manual_seed(1)
        again_30 = vae_30(x_30)[0]
        mu_ref_30, lv_ref_30 = vae_30.encode(x_30)
        verdict("25.30", tuple(logits_30.shape) == (5, 784) and tuple(mu_30.shape) == (5, 20)
                and torch.equal(logits_30, again_30) and not torch.equal(logits_30, vae_30(x_30)[0])
                and torch.equal(mu_30, mu_ref_30) and torch.equal(lv_30, lv_ref_30),
                "forward rend (logits, mu, log_var), tire un code neuf à chaque appel (torch.randn_like).",
                "forward doit rendre (logits (5, 784), mu, log_var) avec mu, log_var = encode(x), et tirer eps par "
                "torch.randn_like(mu) : la même graine, la même sortie ; un autre appel, une autre sortie.")
    g_30 = torch.Generator().manual_seed(30)
    x_t = torch.rand(6, 784, generator=g_30)
    logits_t, mu_t, lv_t = (torch.randn(6, 784, generator=g_30), torch.randn(6, 4, generator=g_30, requires_grad=True),
                            torch.randn(6, 4, generator=g_30))
    result_30 = vae_loss_30(x_t, logits_t, mu_t, lv_t, beta=2.0)
    if returned("25.30", "vae_loss_30", result_30):
        rec_ref_30 = F.binary_cross_entropy_with_logits(logits_t, x_t, reduction="sum") / 6     # summed, then / batch
        kl_ref_30 = torch.distributions.kl_divergence(torch.distributions.Normal(mu_t, torch.exp(0.5 * lv_t)),
                                                      torch.distributions.Normal(0.0, 1.0)).sum(dim=1).mean()
        same("25.30", "vae_loss_30(..., beta=2) : (loss, rec, kl)", [float(v.detach()) for v in result_30],
             [float(rec_ref_30 + 2 * kl_ref_30), float(rec_ref_30), float(kl_ref_30)], rtol=1e-5)
        result_30[0].backward()
        verdict("25.30", mu_t.grad is not None and float(mu_t.grad.abs().sum()) > 0,
                "la loss se dérive par rapport à mu.", "la loss doit rester un tenseur dérivable (pas de .item()).")'''

GIVEN_VAE = r'''# The reference VAE (your 25.30), its loss, its training loop, and VAE_C: the VAE that part C explores,
# trained the same way in both modes (the first N_FIXED images, 10 epochs, seed 2530; about 15 s on a CPU)
def make_vae_decoder(n_latent, hidden=(512, 256)):
    """The decoder of the VAE: Linear(n_latent, hidden[-1]) + ReLU, ..., Linear(hidden[0], 784) (logits)."""
    layers, n_in = [], n_latent
    for n_out in reversed(hidden):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    return nn.Sequential(*layers, nn.Linear(n_in, 784))


class VAE(nn.Module):
    """The VAE of 25.30: encoder (dense ReLU layers), to_mu and to_log_var, decoder (logits)."""

    def __init__(self, n_latent=20, hidden=(512, 256)):
        super().__init__()
        self.n_latent, self.hidden = n_latent, tuple(hidden)
        layers, n_in = [], 784
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
            n_in = n_out
        self.encoder = nn.Sequential(*layers)
        self.to_mu, self.to_log_var = nn.Linear(n_in, n_latent), nn.Linear(n_in, n_latent)
        self.decoder = make_vae_decoder(n_latent, hidden)

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_log_var(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        return self.decoder(mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)), mu, log_var


def vae_loss(x, logits, mu, log_var, beta=1.0):
    """(loss, rec, kl): the binary cross-entropy summed over the pixels, the KL summed over the latents, both
    averaged over the batch; loss = rec + beta * kl."""
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()
    kl = 0.5 * (mu ** 2 + torch.exp(log_var) - 1 - log_var).sum(dim=1).mean()
    return rec + beta * kl, rec, kl


def vae_eval(model, x, batch_size=1000):
    """(rec, kl, kl_per_latent) of the images x: rec decoded from the means mu (no draw), kl exact, both averaged
    over the images; kl_per_latent: the mean KL of each latent, an array (n_latent,)."""
    model.eval()
    rec, kls = 0.0, []
    x = to_tensor(x)
    with torch.inference_mode():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(DEVICE)
            mu, log_var = model.encode(xb)
            rec += F.binary_cross_entropy_with_logits(model.decoder(mu), xb, reduction="sum").item()
            kls.append((0.5 * (mu ** 2 + torch.exp(log_var) - 1 - log_var)).cpu())
    per_latent = torch.cat(kls).mean(dim=0).numpy()
    return rec / len(x), float(per_latent.sum()), per_latent


def train_vae(model, x, epochs, beta=1.0, lr=1e-3, batch_size=128, seed=0, loss_fn=None, verbose=False):
    """Train the VAE `model` in place (on DEVICE) on the images x (N, 784): Adam(lr), mini-batches in an order drawn
    from `seed`, the draws of the codes from torch.manual_seed(seed), loss_fn(x, logits, mu, log_var, beta) (vae_loss
    by default). Returns the history: "loss", "rec", "kl" (means of each epoch), "seconds"."""
    torch.manual_seed(seed)
    model.to(DEVICE)
    loss_fn = loss_fn or vae_loss
    x = to_tensor(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    history = {"loss": [], "rec": [], "kl": [], "seconds": []}
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        totals = np.zeros(3)
        order = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch_size):
            xb = x[order[i:i + batch_size]].to(DEVICE)
            loss, rec, kl = loss_fn(xb, *model(xb), beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += np.array([loss.item(), rec.item(), kl.item()]) * len(xb)
        for key, value in zip(("loss", "rec", "kl"), totals / len(x)):
            history[key].append(float(value))
        history["seconds"].append(time.perf_counter() - start)
        if verbose:
            print(f"epoch {epoch:2d} · loss {history['loss'][-1]:.1f} · rec {history['rec'][-1]:.1f} · kl {history['kl'][-1]:.1f}")
    return history


def vae_images(model, z):
    """The images (N, 784) that the decoder of `model` draws from the codes z: sigmoid of the logits."""
    model.eval()
    return 1 / (1 + np.exp(-decode(model, np.asarray(z, dtype=np.float32))))


torch.manual_seed(2530)
VAE_C = VAE(20)
train_vae(VAE_C, X_MNIST[:N_FIXED], 10, seed=2530)
rec_c, kl_c, _ = vae_eval(VAE_C, XM_TEST)
print(f"VAE_C : reconstruction {rec_c:.1f} nats et KL {kl_c:.1f} nats par image de test")'''

EXPERIMENT_30 = r'''with checked("25.30"):
    EPOCHS_30 = 3 if FAST_MODE else 20
    torch.manual_seed(2530)
    mine_30 = VAE30(20)
    history_30 = train_vae(mine_30, X_MNIST, EPOCHS_30, seed=2530, loss_fn=vae_loss_30, verbose=True)
    z_30 = np.random.default_rng(30).standard_normal((10, 20))
    show_rows([XM_TEST[:10], vae_images(mine_30, encode_mu(mine_30, XM_TEST[:10])), vae_images(mine_30, z_30)],
              ["originaux", "décodés de mu", "codes tirés de N(0, I)"], f"ton VAE, {EPOCHS_30} epochs")'''

GIVEN_30_HELPER = r'''def encode_mu(model, x):
    """The means mu of the codes of the images x, for a VAE (model.encode): a NumPy array (eval mode, no graph)."""
    model.eval()
    return _batched(lambda xb: model.encode(xb)[0], x)'''

GIVEN_31 = r'''# CLF_31: a small digit classifier (an MLP trained 5 epochs on the first N_FIXED images), and 1 000 codes drawn from N(0, I)
torch.manual_seed(2531)
CLF_31 = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(DEVICE)
_opt_31 = torch.optim.Adam(CLF_31.parameters(), lr=1e-3)
_x_31, _y_31 = to_tensor(X_MNIST[:N_FIXED]), torch.from_numpy(Y_MNIST[:N_FIXED])
_g_31 = torch.Generator().manual_seed(2531)
for _ in range(5):
    _order_31 = torch.randperm(N_FIXED, generator=_g_31)
    for _i in range(0, N_FIXED, 128):
        _idx = _order_31[_i:_i + 128]
        _loss = F.cross_entropy(CLF_31(_x_31[_idx].to(DEVICE)), _y_31[_idx].to(DEVICE))
        _opt_31.zero_grad()
        _loss.backward()
        _opt_31.step()


def classify_31(images):
    """(predicted digit, probability of that digit) for each image, two NumPy arrays."""
    probs = torch.softmax(torch.from_numpy(_batched(CLF_31, images)), dim=1).numpy()
    return probs.argmax(axis=1), probs.max(axis=1)


Z_31 = np.random.default_rng(2531).standard_normal((1000, 20)).astype(np.float32)
print(f"CLF_31 : accuracy de test {np.mean(classify_31(XM_TEST)[0] == YM_TEST):.3f}")'''

TODO_31 = r'''ae_top_31 = ...    # a) the codes Z_31 decoded by AE_DEEP: does the digit the classifier predicts most often cover
                   #    more than 20 % of the 1 000 images? True or False
vae_top_31 = ...   # b) the same question for the codes decoded by VAE_C: True or False
vae_conf_31 = ...  # c) are more than half of the VAE_C images recognised with confidence (probability > 0.9)?
                   #    True or False'''

SOLUTION_31 = r'''ae_top_31, vae_top_31, vae_conf_31 = True, False, False'''

EXPERIMENT_31 = r'''with checked("25.31"):
    images_31 = {"AE_DEEP": decode(AE_DEEP, Z_31), "VAE_C": vae_images(VAE_C, Z_31)}
    for name_31, imgs_31 in images_31.items():
        pred_31, conf_31 = classify_31(imgs_31)
        counts_31 = np.bincount(pred_31, minlength=10) / len(pred_31)
        print(f"{name_31:8s} chiffre le plus prédit : {counts_31.argmax()} ({counts_31.max():.0%} des tirages) · "
              f"reconnus avec confiance (p > 0,9) : {np.mean(conf_31 > 0.9):.0%} · répartition : "
              + " ".join(f"{c:.0%}" for c in counts_31))
    codes_ae_31 = encode(AE_DEEP, XM_TEST)
    print(f"normes des codes : tirages de N(0, I) {np.linalg.norm(Z_31, axis=1).mean():.1f} · codes de AE_DEEP "
          f"{np.linalg.norm(codes_ae_31, axis=1).mean():.1f} · moyennes mu de VAE_C {np.linalg.norm(encode_mu(VAE_C, XM_TEST), axis=1).mean():.1f}")
    show_rows([images_31["AE_DEEP"][:12], images_31["VAE_C"][:12]], ["AE_DEEP", "VAE_C"], "les mêmes codes, deux décodeurs")
wb.check("25.31a", ae_top_31)
wb.check("25.31b", vae_top_31)
wb.check("25.31c", vae_conf_31)'''

RECORD_31 = r'''wb.record("25.31a", True, mistakes={"lis la part du chiffre le plus prédit pour AE_DEEP": False})
wb.record("25.31b", False, mistakes={"lis la part du chiffre le plus prédit pour VAE_C": True})
wb.record("25.31c", False, mistakes={"lis la part des tirages de VAE_C reconnus avec confiance": True})'''

TODO_32 = r'''def repeat_32(vae, x, k, seed=0):
    """k passes of ONE image x (784 pixels) through the VAE: mu, log_var = vae.encode(x), then after
    torch.manual_seed(seed), z = mu + exp(0.5 log_var) * torch.randn(k, n_latent), decoded (sigmoid of the logits),
    in eval mode and without a graph. A NumPy array (k, 784)."""
    raise NotImplementedError("repeat_32")


def relative_noise_32(model, mu, frac, seed=0):
    """The codes mu (N, d) with a noise of `frac` of their own value: mu * (1 + frac * u), u uniform in [-1, 1],
    drawn by np.random.default_rng(seed).uniform(-1, 1, size=mu.shape); a new float32 array."""
    raise NotImplementedError("relative_noise_32")'''

SOLUTION_32 = r'''def repeat_32(vae, x, k, seed=0):
    """k passes of ONE image x (784 pixels) through the VAE: mu, log_var = vae.encode(x), then after
    torch.manual_seed(seed), z = mu + exp(0.5 log_var) * torch.randn(k, n_latent), decoded (sigmoid of the logits),
    in eval mode and without a graph. A NumPy array (k, 784)."""
    vae.eval()
    with torch.inference_mode():
        mu, log_var = vae.encode(to_tensor(x).reshape(1, 784).to(DEVICE))
        torch.manual_seed(seed)
        z = mu + torch.exp(0.5 * log_var) * torch.randn(k, mu.shape[1]).to(DEVICE)
        return torch.sigmoid(vae.decoder(z)).cpu().numpy()


def relative_noise_32(model, mu, frac, seed=0):
    """The codes mu (N, d) with a noise of `frac` of their own value: mu * (1 + frac * u), u uniform in [-1, 1],
    drawn by np.random.default_rng(seed).uniform(-1, 1, size=mu.shape); a new float32 array."""
    mu = np.asarray(mu, dtype=np.float64)
    u = np.random.default_rng(seed).uniform(-1, 1, size=mu.shape)
    return (mu * (1 + frac * u)).astype(np.float32)'''

CHECK_32 = r'''with checked("25.32"):
    two_32 = XM_TEST[np.flatnonzero(YM_TEST == 2)[0]]
    rep_32 = repeat_32(VAE_C, two_32, 8, seed=5)
    if returned("25.32", "repeat_32", rep_32):
        mu_32, lv_32 = (t.detach() for t in VAE_C.encode(to_tensor(two_32)[None].to(DEVICE)))
        torch.manual_seed(5)
        want_32 = torch.sigmoid(VAE_C.decoder(mu_32 + torch.exp(0.5 * lv_32) * torch.randn(8, 20).to(DEVICE))).detach().cpu().numpy()
        same("25.32", "repeat_32(VAE_C, un 2, 8, seed=5) : huit tirages du même code", rep_32, want_32, rtol=1e-4, atol=1e-6)
    noisy_32 = relative_noise_32(VAE_C, np.array([[1.0, -2.0], [0.0, 4.0]]), 0.1, seed=3)
    if returned("25.32", "relative_noise_32", noisy_32):
        same("25.32", "relative_noise_32([[1, -2], [0, 4]], 0.1, seed=3)", noisy_32,
             np.array([[1.0, -2.0], [0.0, 4.0]]) * (1 + 0.1 * np.random.default_rng(3).uniform(-1, 1, size=(2, 2))),
             rtol=1e-6)'''

EXPERIMENT_32 = r'''with checked("25.32"):
    two_32 = XM_TEST[np.flatnonzero(YM_TEST == 2)[0]]
    rep_32 = repeat_32(VAE_C, two_32, 8, seed=32)
    print(f"huit passages du même 2 : écart-type moyen d'un pixel d'un passage à l'autre {rep_32.std(axis=0).mean():.3f}")
    show_rows([np.vstack([two_32[None], rep_32])], ["un 2, puis 8 passages"])
    mu_32 = encode_mu(VAE_C, XM_TEST[:500])
    codes_32 = encode(AE_DEEP, XM_TEST[:500])
    for frac_32 in (0.1, 0.3, 1.0):
        conf_vae_32 = np.mean(classify_31(vae_images(VAE_C, relative_noise_32(VAE_C, mu_32, frac_32, seed=32)))[1] > 0.9)
        conf_ae_32 = np.mean(classify_31(decode(AE_DEEP, relative_noise_32(AE_DEEP, codes_32, frac_32, seed=32)))[1] > 0.9)
        print(f"bruit de {frac_32:.0%} de la valeur des codes : reconnus avec confiance · VAE_C {conf_vae_32:.0%} · AE_DEEP {conf_ae_32:.0%}")
    show_rows([vae_images(VAE_C, relative_noise_32(VAE_C, mu_32[:10], f, seed=32)) for f in (0.0, 0.1, 0.3, 1.0)],
              ["0 %", "10 %", "30 %", "100 %"], "VAE_C : les codes de 10 chiffres, bruités")
    with_mylearn_32 = written(lambda: mylearn.vae.interpolate_latents([0.0], [1.0], 2))
    if with_mylearn_32:
        a_32, b_32 = encode_mu(VAE_C, XM_TEST[[np.flatnonzero(YM_TEST == 4)[0], np.flatnonzero(YM_TEST == 9)[1]]])
        show_rows([vae_images(VAE_C, mylearn.vae.interpolate_latents(a_32, b_32, 8, method="slerp"))], ["VAE_C, slerp"],
                  "d'un 4 à un 9")
    else:
        print("⏳ Ex 25.32 : l'interpolation attend ta fonction interpolate_latents (25.20).")'''

GIVEN_33 = r'''# VAE_2D: the VAE of 25.30 with a code of 2 numbers (the first N_FIXED images, 10 epochs, ≈ 15 s), and grid_codes,
# the reference version of grid_codes_18
torch.manual_seed(2533)
VAE_2D = VAE(2)
train_vae(VAE_2D, X_MNIST[:N_FIXED], 10, seed=2533)


def grid_codes(lims, n):
    """The n × n codes of a regular grid over lims = (xmin, xmax, ymin, ymax), row by row from the top (25.18)."""
    xmin, xmax, ymin, ymax = lims
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymax, ymin, n))
    return np.column_stack([gx.ravel(), gy.ravel()])


print(f"VAE_2D : reconstruction {vae_eval(VAE_2D, XM_TEST)[0]:.1f} nats par image de test")'''

TODO_33 = r'''def plot_vae_map_33(vae, x, labels, lim=3.0, n=15):
    """Two panels side by side: (left) the means mu of the codes of x (encode_mu), coloured by digit, with a legend,
    the circles of radius 1, 2 and 3 around the origin, and the limits -lim to lim on both axes; (right) the
    n × n images that vae_images draws from grid_codes((-lim, lim, -lim, lim), n), in ONE image of (28 n) × (28 n)
    pixels, the code of the top-left corner first. Ends with plt.show(). Returns the mosaic."""
    raise NotImplementedError("plot_vae_map_33")'''

SOLUTION_33 = r'''def plot_vae_map_33(vae, x, labels, lim=3.0, n=15):
    """Two panels side by side: (left) the means mu of the codes of x (encode_mu), coloured by digit, with a legend,
    the circles of radius 1, 2 and 3 around the origin, and the limits -lim to lim on both axes; (right) the
    n × n images that vae_images draws from grid_codes((-lim, lim, -lim, lim), n), in ONE image of (28 n) × (28 n)
    pixels, the code of the top-left corner first. Ends with plt.show(). Returns the mosaic."""
    mu = encode_mu(vae, x)
    images = vae_images(vae, grid_codes((-lim, lim, -lim, lim), n)).reshape(n, n, 28, 28)
    mosaic = images.transpose(0, 2, 1, 3).reshape(28 * n, 28 * n)
    fig, (ax_map, ax_grid) = plt.subplots(1, 2, figsize=(12, 5.6))
    colors = wb.plot.class_colors(10)
    for digit in range(10):
        keep = np.asarray(labels) == digit
        ax_map.scatter(mu[keep, 0], mu[keep, 1], s=3, color=colors[digit], label=str(digit))
    for radius in (1, 2, 3):
        ax_map.add_patch(plt.Circle((0, 0), radius, fill=False, color="k", lw=0.8, ls="--"))
    ax_map.set_xlim(-lim, lim)
    ax_map.set_ylim(-lim, lim)
    ax_map.set_aspect("equal")
    ax_map.legend(markerscale=4, fontsize=8, ncol=2, loc="upper right")
    ax_map.set_title("moyennes mu de 2 000 chiffres de test")
    ax_grid.imshow(mosaic, cmap="gray", extent=(-lim, lim, -lim, lim))
    ax_grid.set_title(f"la grille de -{lim:g} à {lim:g}, décodée")
    plt.tight_layout()
    plt.show()
    return mosaic'''

CHECK_33 = r'''with checked("25.33"):
    mosaic_33 = plot_vae_map_33(VAE_2D, XM_TEST, YM_TEST)
    if returned("25.33", "plot_vae_map_33", mosaic_33):
        corner_33 = vae_images(VAE_2D, [[-3.0, 3.0]]).reshape(28, 28)
        verdict("25.33", np.shape(mosaic_33) == (420, 420) and np.allclose(np.asarray(mosaic_33)[:28, :28], corner_33, atol=1e-5),
                "la mosaïque (420, 420) commence par le code (-3, 3).",
                f"attendu une mosaïque (420, 420) dont l'image en haut à gauche décode (-3, 3), reçu {shape_text(mosaic_33)}.")
    radius_33 = np.linalg.norm(encode_mu(VAE_2D, XM_TEST), axis=1)
    print(f"part des moyennes mu à moins de 1, 2 et 3 de l'origine : {np.mean(radius_33 < 1):.0%}, "
          f"{np.mean(radius_33 < 2):.0%}, {np.mean(radius_33 < 3):.0%} (pour des tirages de N(0, I) : 39 %, 86 %, 99 %)")'''


def _history_34_cell() -> str:
    runs = HISTORY["collapse34"]["runs"]
    lines = ["HISTORY_34 = {   # four trainings of the same VAE (20 latents, 20 epochs, MNIST), tools/chapters/history_ch25.py"]
    for name in sorted(runs):
        run = runs[name]
        lines.append(f'    "{name}": {{"rec": {[round(v, 2) for v in run["rec"]]},')
        lines.append(f'          "kl": {[round(v, 2) for v in run["kl"]]},')
        lines.append(f'          "active": {run["active"]}}},')
    lines.append("}")
    return "\n".join(lines) + r'''

fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for name_34, run_34 in HISTORY_34.items():
    epochs_34 = range(1, len(run_34["rec"]) + 1)
    axes[0].plot(epochs_34, run_34["rec"], label=name_34)
    axes[1].plot(epochs_34, run_34["kl"], label=name_34)
    axes[2].plot(epochs_34, run_34["active"], label=name_34)
for ax, title in zip(axes, ("reconstruction (nats par image de validation)", "KL (nats par image)",
                            "latents actifs (KL moyenne > 0,05 nat)")):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("epoch")
    ax.legend()
plt.tight_layout()
plt.show()'''


def _answers_34():
    runs = HISTORY["collapse34"]["runs"]
    by_setting = HISTORY["collapse34"]["settings"]           # {"A": "anneal", ...}
    letter = {setting: name for name, setting in by_setting.items()}
    collapsed = letter["beta100"]
    active_b20 = runs[letter["beta20"]]["active"][-1]
    anneal_kl = runs[letter["anneal"]]["kl"]
    peak = round(anneal_kl[0] - anneal_kl[-1], 1)          # the drop of the KL that starts the highest
    gap = round(runs[letter["beta20"]]["rec"][-1] - runs[letter["beta1"]]["rec"][-1], 1)
    mapping = letter["beta1"] + letter["beta20"] + letter["beta100"] + letter["anneal"]
    return collapsed, active_b20, peak, gap, mapping, letter


import numpy as np  # noqa: E402  (the builder computes the answers of 25.34 from the history)

ANSWERS_34 = _answers_34() if HISTORY["collapse34"] else ("A", 0, 1, 0.0, "ABCD", {})

TODO_34 = r'''collapsed_34 = ...  # a) the letter of the run whose posterior collapsed (KL almost 0 for all the latents)
active_34 = ...     # b) in the run that ends with 1 to 5 active latents, how many are active at the last epoch?
peak_34 = ...       # c) in the run whose KL starts the highest: its KL at the first epoch minus its KL at the
                    #    last epoch, in nats (1 decimal)
gap_34 = ...        # d) the last reconstruction of the run of b) minus the last reconstruction of the run whose
                    #    KL ends the highest, in nats (1 decimal)
mapping_34 = ...    # e) the letters of the runs for beta = 1, beta = 20, beta = 100 and beta rising from 0 to 1
                    #    over the first 10 epochs, in this order, e.g. "ABCD"'''

CHECK_34 = r'''wb.check("25.34a", collapsed_34)
wb.check("25.34b", active_34)
wb.check("25.34c", peak_34)
wb.check("25.34d", gap_34)
wb.check("25.34e", mapping_34)'''


def _record_34() -> str:
    collapsed, active, peak, gap, mapping, letter = ANSWERS_34
    runs = HISTORY["collapse34"]["runs"]
    m_a = {"ce run garde quelques latents actifs jusqu'au bout : cherche celui où il n'en reste aucun": letter["beta20"],
           "la KL de ce run reste nettement positive jusqu'au bout": letter["beta1"],
           "la KL de ce run ne finit pas près de 0 : relis la fin de sa courbe": letter["anneal"]}
    m_b = {}
    if runs[letter["beta20"]]["active"][0] != active:
        m_b["lis la dernière valeur de la courbe des latents actifs, pas la première"] = runs[letter["beta20"]]["active"][0]
    m_c = {"la première epoch moins la dernière, pas l'inverse": -peak}
    b1, b20, b100, ann = mapping
    m_e = {"les runs à β = 1 et à β montant finissent proches : regarde les premières epochs de leur KL": ann + b20 + b100 + b1,
           "relis a) et b) : lequel garde des latents actifs, lequel n'en garde aucun ?": b1 + b100 + b20 + ann}
    return (f'wb.record("25.34a", "{collapsed}", mistakes={json.dumps(m_a, ensure_ascii=False)})\n'
            f'wb.record("25.34b", {active}, mistakes={json.dumps(m_b, ensure_ascii=False)})\n'
            f'wb.record("25.34c", {peak}, decimals=1, mistakes={json.dumps(m_c, ensure_ascii=False)})\n'
            f'wb.record("25.34d", {gap}, decimals=1, mistakes={{"le run de b) moins l\'autre, pas l\'inverse": {-gap}}})\n'
            f'wb.record("25.34e", "{mapping}", mistakes={json.dumps(m_e, ensure_ascii=False)})')


TODO_35 = r'''def export_decoder_35(vae, path):
    """Save what a generator needs, with torch.save, in the file `path`: the dict {"n_latent": vae.n_latent,
    "hidden": list(vae.hidden), "state_dict": vae.decoder.state_dict()} (the decoder only)."""
    raise NotImplementedError("export_decoder_35")


def load_generator_35(path):
    """Read the file of export_decoder_35 (torch.load(path, weights_only=True)), rebuild the decoder (the layers of
    25.30: Linear(n_latent, hidden[-1]) + ReLU, ..., Linear(hidden[0], 784)), load its weights, switch it to eval
    mode, and return a function generate(n, seed=0): z = torch.randn(n, n_latent,
    generator=torch.Generator().manual_seed(seed)), images = sigmoid(decoder(z)) * 255, rounded (torch.round),
    as a NumPy uint8 array (n, 28, 28). Only torch, nn and numpy: nothing from this notebook."""
    raise NotImplementedError("load_generator_35")


def test_generator_shape_35():
    """Your test: generate(5) gives 5 images of 28 x 28 in uint8 (use GEN_PATH, the path of the exported file)."""
    raise NotImplementedError("test_generator_shape_35")


def test_generator_seed_35():
    """Your test: the same seed gives the same images, two different seeds give different images."""
    raise NotImplementedError("test_generator_seed_35")'''

SOLUTION_35 = r'''def export_decoder_35(vae, path):
    """Save what a generator needs, with torch.save, in the file `path`: the dict {"n_latent": vae.n_latent,
    "hidden": list(vae.hidden), "state_dict": vae.decoder.state_dict()} (the decoder only)."""
    torch.save({"n_latent": vae.n_latent, "hidden": list(vae.hidden), "state_dict": vae.decoder.state_dict()}, path)


def load_generator_35(path):
    """Read the file of export_decoder_35 (torch.load(path, weights_only=True)), rebuild the decoder (the layers of
    25.30: Linear(n_latent, hidden[-1]) + ReLU, ..., Linear(hidden[0], 784)), load its weights, switch it to eval
    mode, and return a function generate(n, seed=0): z = torch.randn(n, n_latent,
    generator=torch.Generator().manual_seed(seed)), images = sigmoid(decoder(z)) * 255, rounded (torch.round),
    as a NumPy uint8 array (n, 28, 28). Only torch, nn and numpy: nothing from this notebook."""
    saved = torch.load(path, weights_only=True)
    layers, n_in = [], saved["n_latent"]
    for n_out in reversed(saved["hidden"]):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    decoder = nn.Sequential(*layers, nn.Linear(n_in, 784))
    decoder.load_state_dict(saved["state_dict"])
    decoder.eval()

    def generate(n, seed=0):
        z = torch.randn(n, saved["n_latent"], generator=torch.Generator().manual_seed(seed))
        with torch.inference_mode():
            images = torch.round(torch.sigmoid(decoder(z)) * 255)
        return images.numpy().astype(np.uint8).reshape(n, 28, 28)

    return generate


def test_generator_shape_35():
    """Your test: generate(5) gives 5 images of 28 x 28 in uint8 (use GEN_PATH, the path of the exported file)."""
    images = load_generator_35(GEN_PATH)(5)
    assert images.shape == (5, 28, 28), f"expected (5, 28, 28), got {images.shape}"
    assert images.dtype == np.uint8, f"expected uint8, got {images.dtype}"


def test_generator_seed_35():
    """Your test: the same seed gives the same images, two different seeds give different images."""
    generate = load_generator_35(GEN_PATH)
    assert np.array_equal(generate(3, seed=7), generate(3, seed=7)), "the same seed must give the same images"
    assert not np.array_equal(generate(3, seed=7), generate(3, seed=8)), "two seeds must give different images"'''

CHECK_35 = r'''GEN_PATH = str(WORK_DIR / "generator_35.pt")
with checked("25.35"):
    export_decoder_35(VAE_C.cpu(), GEN_PATH)
    VAE_C.to(DEVICE)
    generate_35 = load_generator_35(GEN_PATH)
    if returned("25.35", "load_generator_35", generate_35):
        images_35 = generate_35(6, seed=1)
        z_35 = torch.randn(6, 20, generator=torch.Generator().manual_seed(1))
        want_35 = np.round(vae_images(VAE_C, z_35.numpy()) * 255).astype(np.uint8).reshape(6, 28, 28)
        verdict("25.35", isinstance(images_35, np.ndarray) and images_35.dtype == np.uint8 and images_35.shape == (6, 28, 28)
                and np.abs(images_35.astype(int) - want_35.astype(int)).max() <= 1,
                "le générateur rechargé rend les images du décodeur de VAE_C, en uint8 (6, 28, 28).",
                f"attendu un tableau uint8 (6, 28, 28) égal aux images du décodeur de VAE_C pour la graine 1, reçu "
                f"{type(images_35).__name__} {shape_text(images_35)}.")
        show_rows([generate_35(10, seed=0) / 255.0], ["generate(10)"])
    PRE_35 = f"import torch\nimport torch.nn as nn\nGEN_PATH = {GEN_PATH!r}"
    TESTS_35 = [test_generator_shape_35, test_generator_seed_35]
    import inspect
    if all("NotImplementedError" not in inspect.getsource(test) for test in TESTS_35):
        print("Tes tests, sur ton générateur (ils doivent passer) :")
        result_35 = wb.run_pytest(TESTS_35, subject=load_generator_35, preamble=PRE_35)
        verdict("25.35", result_35.ok, "tes tests passent sur ton générateur.", "tes tests doivent passer sur ton générateur.")
        for broken_35, why_35 in [(broken_float_35, "un générateur qui rend des flottants dans [0, 1]"),
                                  (broken_seed_35, "un générateur qui ignore la graine")]:
            caught_35 = not wb.run_pytest(TESTS_35, subject=broken_35, name="load_generator_35", preamble=PRE_35, quiet=True).ok
            verdict("25.35", caught_35, f"tes tests attrapent {why_35}.", f"tes tests laissent passer {why_35}.")
    else:
        print("⏳ Ex 25.35 : écris tes deux tests.")'''

GIVEN_35 = r'''# Two broken generators, for the check: your tests must catch them
def broken_float_35(path):
    saved = torch.load(path, weights_only=True)
    layers, n_in = [], saved["n_latent"]
    for n_out in reversed(saved["hidden"]):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    decoder = nn.Sequential(*layers, nn.Linear(n_in, 784))
    decoder.load_state_dict(saved["state_dict"])
    decoder.eval()

    def generate(n, seed=0):
        z = torch.randn(n, saved["n_latent"], generator=torch.Generator().manual_seed(seed))
        with torch.inference_mode():
            return torch.sigmoid(decoder(z)).numpy().reshape(n, 28, 28)          # floats in [0, 1], not uint8

    return generate


def broken_seed_35(path):
    saved = torch.load(path, weights_only=True)
    layers, n_in = [], saved["n_latent"]
    for n_out in reversed(saved["hidden"]):
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    decoder = nn.Sequential(*layers, nn.Linear(n_in, 784))
    decoder.load_state_dict(saved["state_dict"])
    decoder.eval()

    def generate(n, seed=0):
        z = torch.randn(n, saved["n_latent"], generator=torch.Generator().manual_seed(0))   # the seed is ignored
        with torch.inference_mode():
            images = torch.round(torch.sigmoid(decoder(z)) * 255)
        return images.numpy().astype(np.uint8).reshape(n, 28, 28)

    return generate'''

TODO_36 = r'''def beta_sweep_36(betas, epochs, seed=2536, x=None, n_latent=16):
    """For each beta of `betas`: torch.manual_seed(seed), model = VAE(n_latent), train_vae(model, x, epochs,
    beta=beta, seed=seed) (x = X_FASHION if None), then rec, kl, per_latent = vae_eval(model, XF_TEST).
    Returns {beta: {"rec": rec, "kl": kl, "active": the number of latents whose mean KL is > 0.05 nat,
    "model": model}}."""
    raise NotImplementedError("beta_sweep_36")'''

SOLUTION_36 = r'''def beta_sweep_36(betas, epochs, seed=2536, x=None, n_latent=16):
    """For each beta of `betas`: torch.manual_seed(seed), model = VAE(n_latent), train_vae(model, x, epochs,
    beta=beta, seed=seed) (x = X_FASHION if None), then rec, kl, per_latent = vae_eval(model, XF_TEST).
    Returns {beta: {"rec": rec, "kl": kl, "active": the number of latents whose mean KL is > 0.05 nat,
    "model": model}}."""
    x = X_FASHION if x is None else x
    results = {}
    for beta in betas:
        torch.manual_seed(seed)
        model = VAE(n_latent)
        train_vae(model, x, epochs, beta=beta, seed=seed)
        rec, kl, per_latent = vae_eval(model, XF_TEST)
        results[beta] = {"rec": rec, "kl": kl, "active": int(np.sum(per_latent > 0.05)), "model": model}
    return results'''

CHECK_36 = r'''with checked("25.36"):
    tiny_36 = beta_sweep_36([2.0], 1, seed=4, x=X_FASHION[:512], n_latent=4)
    if returned("25.36", "beta_sweep_36", tiny_36):
        torch.manual_seed(4)
        ref_36 = VAE(4)
        train_vae(ref_36, X_FASHION[:512], 1, beta=2.0, seed=4)
        rec_36, kl_36, per_36 = vae_eval(ref_36, XF_TEST)
        row_36 = tiny_36.get(2.0, {}) if isinstance(tiny_36, dict) else {}
        verdict("25.36", {"rec", "kl", "active", "model"} <= set(row_36) and abs(row_36["rec"] - rec_36) < 1e-2
                and abs(row_36["kl"] - kl_36) < 1e-3 and row_36["active"] == int(np.sum(per_36 > 0.05)),
                f"beta_sweep_36 sur un cas d'essai : rec {rec_36:.1f}, kl {kl_36:.2f}, {int(np.sum(per_36 > 0.05))} latents actifs.",
                f"attendu {{2.0: {{'rec': {rec_36:.1f}, 'kl': {kl_36:.2f}, 'active': {int(np.sum(per_36 > 0.05))}, 'model': ...}}}}, "
                f"reçu {({k: v for k, v in row_36.items() if k != 'model'})!r:.200}.")'''

EXPERIMENT_36 = r'''with checked("25.36"):
    BETAS_36 = (0.5, 1.0, 4.0, 16.0)
    EPOCHS_36 = 4 if FAST_MODE else 20
    sweep_36 = beta_sweep_36(BETAS_36, EPOCHS_36)
    for beta_36, row_36 in sweep_36.items():
        print(f"beta = {beta_36:4.1f} · reconstruction {row_36['rec']:6.1f} nats · KL {row_36['kl']:5.1f} nats · "
              f"{row_36['active']:2d} latents actifs sur 16")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot([r["kl"] for r in sweep_36.values()], [r["rec"] for r in sweep_36.values()], "o-")
    for beta_36, row_36 in sweep_36.items():
        ax.annotate(f"β = {beta_36:g}", (row_36["kl"], row_36["rec"]), fontsize=8)
    ax.set_xlabel("KL (nats)")
    ax.set_ylabel("reconstruction (nats)")
    plt.show()
    z_36 = np.random.default_rng(36).standard_normal((10, 16))
    show_rows([XF_TEST[:10]] + [vae_images(r["model"], encode_mu(r["model"], XF_TEST[:10])) for r in sweep_36.values()],
              ["originaux"] + [f"β = {b:g}" for b in BETAS_36], "reconstructions (décodées de mu)", size=0.9)
    show_rows([vae_images(r["model"], z_36) for r in sweep_36.values()], [f"β = {b:g}" for b in BETAS_36],
              "codes tirés de N(0, I)", size=0.9)'''

TODO_37 = r'''def fit_without_37(digit, epochs, seed=2537):
    """make_deep_ae(20), created just after torch.manual_seed(seed), trained by train_ae(model, x, epochs, seed=seed)
    on the images of X_MNIST that are NOT `digit` (Y_MNIST != digit). Returns the model."""
    raise NotImplementedError("fit_without_37")


def scores_37(model, x):
    """The anomaly score of each image: the MSE between the image and its reconstruction (an array of len(x))."""
    raise NotImplementedError("scores_37")


def auc_37(normal, anomalies):
    """The ROC-AUC of the scores (a larger score = more anomalous): mylearn.metrics.roc_auc with the label 0 for the
    scores of `normal` and 1 for those of `anomalies`."""
    raise NotImplementedError("auc_37")'''

SOLUTION_37 = r'''def fit_without_37(digit, epochs, seed=2537):
    """make_deep_ae(20), created just after torch.manual_seed(seed), trained by train_ae(model, x, epochs, seed=seed)
    on the images of X_MNIST that are NOT `digit` (Y_MNIST != digit). Returns the model."""
    torch.manual_seed(seed)
    model = make_deep_ae(20)
    train_ae(model, X_MNIST[Y_MNIST != digit], epochs, seed=seed)
    return model


def scores_37(model, x):
    """The anomaly score of each image: the MSE between the image and its reconstruction (an array of len(x))."""
    return mse_per_image(reconstruct(model, x), x)


def auc_37(normal, anomalies):
    """The ROC-AUC of the scores (a larger score = more anomalous): mylearn.metrics.roc_auc with the label 0 for the
    scores of `normal` and 1 for those of `anomalies`."""
    y_true = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(anomalies), dtype=int)])
    return float(mylearn.metrics.roc_auc(y_true, np.concatenate([normal, anomalies])))'''

CHECK_37 = r'''with checked("25.37"):
    from sklearn.metrics import roc_auc_score
    value_37 = auc_37(np.array([0.1, 0.2, 0.4]), np.array([0.3, 0.5]))
    if returned("25.37", "auc_37", value_37):
        same("25.37", "auc_37([0.1, 0.2, 0.4], [0.3, 0.5]) contre sklearn.metrics.roc_auc_score", [value_37],
             [roc_auc_score([0, 0, 0, 1, 1], [0.1, 0.2, 0.4, 0.3, 0.5])])
    s_37 = scores_37(AE_DEEP, XM_TEST[:4])
    if returned("25.37", "scores_37", s_37):
        same("25.37", "scores_37(AE_DEEP, 4 chiffres)", s_37, mse_per_image(reconstruct(AE_DEEP, XM_TEST[:4]), XM_TEST[:4]),
             rtol=1e-5)'''

EXPERIMENT_37 = r'''with checked("25.37"):
    EPOCHS_37 = 5 if FAST_MODE else 15
    model_37 = fit_without_37(7, EPOCHS_37)
    normal_37 = scores_37(model_37, XM_TEST[YM_TEST != 7])
    sevens_37 = scores_37(model_37, XM_TEST[YM_TEST == 7])
    clothes_37 = scores_37(model_37, XF_TEST)
    faint_37 = scores_37(model_37, 0.15 * XM_TEST[YM_TEST != 7][:200])          # digits drawn very faintly
    print(f"ROC-AUC · les 7 contre les autres chiffres {auc_37(normal_37, sevens_37):.3f} · les vêtements contre les "
          f"chiffres {auc_37(normal_37, clothes_37):.3f} · des chiffres très pâles contre les chiffres {auc_37(normal_37, faint_37):.3f}")
    fig, ax = plt.subplots(figsize=(8, 3))
    bins_37 = np.linspace(0, np.quantile(np.concatenate([normal_37, sevens_37, clothes_37]), 0.99), 60)
    for values_37, name_37 in [(normal_37, "chiffres (pas de 7)"), (sevens_37, "7"), (clothes_37, "vêtements"),
                               (faint_37, "chiffres pâles")]:
        ax.hist(values_37, bins=bins_37, alpha=0.5, density=True, label=name_37)
    ax.set_xlabel("score d'anomalie (MSE de reconstruction)")
    ax.legend()
    plt.show()'''

TODO_38 = r'''def init_numpy_vae_38(n_latent=8, hidden=128, seed=0):
    """{"enc": init_mlp([784, hidden, 2 * n_latent], rng=rng), "dec": init_mlp([n_latent, hidden, 784], rng=rng)},
    with ONE rng = np.random.default_rng(seed), the encoder first (mylearn.nn.layers.init_mlp)."""
    raise NotImplementedError("init_numpy_vae_38")


def numpy_vae_loss_38(params, x, eps):
    """One forward and backward pass of the NumPy VAE on the batch x (n, 784), with the noise eps (n, d) given.
    Returns (loss, rec, kl, grads), grads = {"enc": [(dW, db), ...], "dec": [(dW, db), ...]} like params."""
    raise NotImplementedError("numpy_vae_loss_38")


def train_numpy_vae_38(params, x, epochs, lr=1e-3, batch_size=128, seed=0):
    """Train params in place with mylearn.optim.Adam on the flat list [W1, b1, W2, b2, ...] of the encoder then the
    decoder; rng = np.random.default_rng(seed): each epoch, rng.permutation(N), then for each mini-batch
    eps = rng.standard_normal((n, d)). Returns the list of the mean losses of each epoch."""
    raise NotImplementedError("train_numpy_vae_38")'''

SOLUTION_38 = r'''def _relu_38(z):
    return np.maximum(z, 0.0)


def _relu_derivative_38(z):
    return (z > 0).astype(float)


def init_numpy_vae_38(n_latent=8, hidden=128, seed=0):
    """{"enc": init_mlp([784, hidden, 2 * n_latent], rng=rng), "dec": init_mlp([n_latent, hidden, 784], rng=rng)},
    with ONE rng = np.random.default_rng(seed), the encoder first (mylearn.nn.layers.init_mlp)."""
    rng = np.random.default_rng(seed)
    return {"enc": mylearn.nn.layers.init_mlp([784, hidden, 2 * n_latent], rng=rng),
            "dec": mylearn.nn.layers.init_mlp([n_latent, hidden, 784], rng=rng)}


def numpy_vae_loss_38(params, x, eps):
    """One forward and backward pass of the NumPy VAE on the batch x (n, 784), with the noise eps (n, d) given.
    Returns (loss, rec, kl, grads), grads = {"enc": [(dW, db), ...], "dec": [(dW, db), ...]} like params."""
    layers, backward, vae = mylearn.nn.layers, mylearn.nn.backward, mylearn.vae
    n, d = eps.shape
    h, cache_enc = layers.mlp_forward(x, params["enc"], activation=_relu_38, return_cache=True)
    mu, log_var = h[:, :d], h[:, d:]
    z = mu + np.exp(0.5 * log_var) * eps                                   # reparameterize, with eps given
    logits, cache_dec = layers.mlp_forward(z, params["dec"], activation=_relu_38, return_cache=True)
    mean_bce, dlogits = backward.binary_cross_entropy_with_logits(logits, x)
    rec, dlogits = 784 * mean_bce, 784 * dlogits                            # summed over pixels, mean over batch
    kl = float(np.mean(vae.gaussian_kl(mu, log_var)))
    grads_dec, dz = backward.mlp_backward(dlogits, cache_dec, params["dec"], activation_derivative=_relu_derivative_38,
                                          return_input_grad=True)
    dmu, dlog_var = vae.reparameterize_backward(dz, eps, log_var)           # through z
    gmu, glog_var = vae.gaussian_kl_grad(mu, log_var)                       # through the KL, averaged over the batch
    dh = np.concatenate([dmu + gmu / n, dlog_var + glog_var / n], axis=1)
    grads_enc = backward.mlp_backward(dh, cache_enc, params["enc"], activation_derivative=_relu_derivative_38)
    return rec + kl, rec, kl, {"enc": grads_enc, "dec": grads_dec}


def train_numpy_vae_38(params, x, epochs, lr=1e-3, batch_size=128, seed=0):
    """Train params in place with mylearn.optim.Adam on the flat list [W1, b1, W2, b2, ...] of the encoder then the
    decoder; rng = np.random.default_rng(seed): each epoch, rng.permutation(N), then for each mini-batch
    eps = rng.standard_normal((n, d)). Returns the list of the mean losses of each epoch."""
    flat = [p for part in ("enc", "dec") for layer in params[part] for p in layer]
    optimizer = mylearn.optim.Adam(flat, lr=lr)
    rng = np.random.default_rng(seed)
    d = params["dec"][0][0].shape[0]
    x = np.asarray(x, dtype=np.float64)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            xb = x[order[i:i + batch_size]]
            loss, _, _, grads = numpy_vae_loss_38(params, xb, rng.standard_normal((len(xb), d)))
            optimizer.step([g for part in ("enc", "dec") for layer in grads[part] for g in layer])
            total += loss * len(xb)
        losses.append(total / len(x))
    return losses'''

TODO_38 = r'''def _relu_38(z):
    return np.maximum(z, 0.0)


def _relu_derivative_38(z):
    return (z > 0).astype(float)


''' + TODO_38

CHECK_38 = r'''with checked("25.38"):
    params_38 = init_numpy_vae_38(n_latent=3, hidden=16, seed=1)
    if returned("25.38", "init_numpy_vae_38", params_38):
        shapes_38 = {k: [(W.shape, b.shape) for W, b in v] for k, v in params_38.items()}
        verdict("25.38", shapes_38 == {"enc": [((784, 16), (16,)), ((16, 6), (6,))], "dec": [((3, 16), (16,)), ((16, 784), (784,))]},
                "init_numpy_vae_38(3, 16) : les formes attendues.", f"formes reçues : {shapes_38}.")
        x_38 = X_MNIST[:5].astype(np.float64)
        eps_38 = np.random.default_rng(2).standard_normal((5, 3))
        result_38 = numpy_vae_loss_38(params_38, x_38, eps_38)
        if returned("25.38", "numpy_vae_loss_38", result_38):
            loss_38, rec_38, kl_38, grads_38 = result_38
            # the same VAE in PyTorch, float64, with copies of the weights: its autograd gives the reference
            tp_38 = {k: [(torch.tensor(W, requires_grad=True), torch.tensor(b, requires_grad=True)) for W, b in v]
                     for k, v in params_38.items()}
            def _mlp_38(a, layers):
                for i, (W, b) in enumerate(layers):
                    a = a @ W + b
                    if i < len(layers) - 1:
                        a = torch.relu(a)
                return a
            xt_38 = torch.tensor(x_38)
            h_38 = _mlp_38(xt_38, tp_38["enc"])
            mu_t38, lv_t38 = h_38[:, :3], h_38[:, 3:]
            logits_t38 = _mlp_38(mu_t38 + torch.exp(0.5 * lv_t38) * torch.tensor(eps_38), tp_38["dec"])
            rec_t38 = F.binary_cross_entropy_with_logits(logits_t38, xt_38, reduction="sum") / 5
            kl_t38 = (0.5 * (mu_t38 ** 2 + torch.exp(lv_t38) - 1 - lv_t38)).sum(dim=1).mean()
            (rec_t38 + kl_t38).backward()
            same("25.38", "(loss, rec, kl) contre le même VAE en PyTorch", [loss_38, rec_38, kl_38],
                 [float(rec_t38 + kl_t38), float(rec_t38), float(kl_t38)], rtol=1e-8)
            for part_38 in ("enc", "dec"):
                for i_38, ((dW_38, db_38), (W_t, b_t)) in enumerate(zip(grads_38[part_38], tp_38[part_38])):
                    same("25.38", f"dW de la couche {i_38} de {part_38} contre torch.autograd", dW_38, W_t.grad.numpy(),
                         rtol=1e-6, atol=1e-10)
                    same("25.38", f"db de la couche {i_38} de {part_38} contre torch.autograd", db_38, b_t.grad.numpy(),
                         rtol=1e-6, atol=1e-10)'''

EXPERIMENT_38 = r'''with checked("25.38"):
    EPOCHS_38 = 5 if FAST_MODE else 30
    params_38 = init_numpy_vae_38(n_latent=8, hidden=128, seed=2538)
    start_38 = time.perf_counter()
    losses_38 = train_numpy_vae_38(params_38, X_MNIST[:5000] if FAST_MODE else X_MNIST, EPOCHS_38, seed=2538)
    print(f"{EPOCHS_38} epochs en {time.perf_counter() - start_38:.0f} s · loss par epoch : " + " ".join(f"{v:.1f}" for v in losses_38))
    verdict("25.38", losses_38[-1] < losses_38[0] - 20, "la loss baisse : ton VAE en NumPy apprend.",
            "la loss devrait baisser d'au moins 20 nats : vérifie les gradients (cellule précédente).")
    z_38 = np.random.default_rng(38).standard_normal((12, 8))
    logits_38 = mylearn.nn.layers.mlp_forward(z_38, params_38["dec"], activation=lambda a: np.maximum(a, 0.0))
    show_rows([1 / (1 + np.exp(-logits_38))], ["N(0, I) décodés"], "ton VAE NumPy comme générateur")'''

GIVEN_39 = r'''# The challenge: the validation (public) and the test (hidden) clothes, with a fixed noise of sigma = 0.3
TEST_CLEAN_39 = XF_TEST[1000:2000]
TEST_NOISY_39 = np.clip(TEST_CLEAN_39 + SIGMA_B * np.random.default_rng(2539).standard_normal(TEST_CLEAN_39.shape),
                        0, 1).astype(np.float32)
GOAL_39 = {GOAL}                    # dB on the validation, then on the test
MAX_PARAMS_39 = 100_000
MAX_SECONDS_39 = 300 if FAST_MODE else 900
print(f"objectif : PSNR ≥ {GOAL_39} dB · au plus {MAX_PARAMS_39:,d} paramètres · au plus {MAX_SECONDS_39} s d'entraînement")'''

TODO_39 = r'''def make_denoiser_39():
    """Your denoiser: an AutoEncoder (rows of 784 pixels in, rows of 784 pixels in [0, 1] out), at most
    MAX_PARAMS_39 parameters."""
    raise NotImplementedError("make_denoiser_39")


def train_39(model, x_clean):
    """Train `model` in place to remove a gaussian noise of standard deviation SIGMA_B (clipped to [0, 1]) from the
    clean images x_clean (X_FASHION): any loop, optimizer or number of epochs, within MAX_SECONDS_39 seconds."""
    raise NotImplementedError("train_39")'''

SOLUTION_39 = r'''def make_denoiser_39():
    """Your denoiser: an AutoEncoder (rows of 784 pixels in, rows of 784 pixels in [0, 1] out), at most
    MAX_PARAMS_39 parameters."""
    c = 32                                            # no pooling: every layer keeps the 28 x 28 resolution
    layers = [nn.Unflatten(1, (1, 28, 28)), nn.Conv2d(1, c, 3, padding=1), nn.ReLU()]
    for _ in range(3):
        layers += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU()]
    return AutoEncoder(nn.Sequential(*layers), nn.Sequential(nn.Conv2d(c, 1, 3, padding=1), nn.Sigmoid(), nn.Flatten()))


def train_39(model, x_clean):
    """Train `model` in place to remove a gaussian noise of standard deviation SIGMA_B (clipped to [0, 1]) from the
    clean images x_clean (X_FASHION): any loop, optimizer or number of epochs, within MAX_SECONDS_39 seconds."""
    train_denoiser(model, x_clean, 3 if FAST_MODE else 2, lr=2e-3, seed=2539)'''

CHECK_39 = r'''with checked("25.39"):
    torch.manual_seed(2539)
    model_39 = make_denoiser_39()
    if returned("25.39", "make_denoiser_39", model_39):
        count_39 = n_params(model_39)
        if count_39 > MAX_PARAMS_39:
            print(f"❌ Ex 25.39 : {count_39:,d} paramètres, plus que les {MAX_PARAMS_39:,d} permis.")
            raise StopCheck
        start_39 = time.perf_counter()
        train_39(model_39.to(DEVICE), X_FASHION)
        seconds_39 = time.perf_counter() - start_39
        val_39 = psnr(VAL_CLEAN_B, reconstruct(model_39, VAL_NOISY_B))
        print(f"{count_39:,d} paramètres · entraînement {seconds_39:.0f} s · PSNR de validation {val_39:.2f} dB "
              f"(images bruitées : {psnr(VAL_CLEAN_B, VAL_NOISY_B):.2f} dB)")
        if seconds_39 > MAX_SECONDS_39:
            print(f"❌ Ex 25.39 : l'entraînement a duré plus de {MAX_SECONDS_39} s.")
        elif val_39 < GOAL_39:
            print(f"❌ Ex 25.39 : il faut au moins {GOAL_39} dB sur la validation avant de révéler le test.")
        else:
            REVEALED_39 = globals().setdefault("REVEALED_39", psnr(TEST_CLEAN_39, reconstruct(model_39, TEST_NOISY_39)))
            print(f"test (révélé une fois par session ; un nouvel essai affiche la même valeur) : {REVEALED_39:.2f} dB")
            verdict("25.39", REVEALED_39 >= GOAL_39, f"défi réussi : {REVEALED_39:.2f} dB sur le test caché.",
                    f"sur le test caché, {REVEALED_39:.2f} dB : moins que les {GOAL_39} dB visés.")'''

PART_C = Part("C", "Le VAE",
              "Tu écris dans ta librairie la reparamétrisation et la KL gaussienne (25.28, 25.29), puis un VAE en "
              "PyTorch (25.30). Tu prédis ce que deviennent des codes tirés au hasard (25.31), tu explores la "
              "stochasticité du VAE (25.32) et son espace latent en 2D (25.33), tu lis des courbes d'entraînement "
              "(25.34), tu exportes le décodeur comme générateur (25.35), tu règles $\\beta$ (25.36), tu détectes des "
              "anomalies (25.37), tu écris un VAE entièrement en NumPy (25.38), et tu relèves le défi de débruitage "
              "(25.39).",
              exercises=[
    Ex("25.28", "🔨", 2, 25, "La reparamétrisation en NumPy : passe avant et passe arrière",
       "écrire le tirage $\\mathbf{z} = \\boldsymbol{\\mu} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{\\varepsilon}$ et "
       "sa passe arrière dans ta librairie.",
       "∂ 25.5 · ch. 18 (passe arrière) · fiche §25.9.2", tracks="R, M, C", mylearn="vae.py",
       body=mylearn_howto() + "\n\n" + COMMON_VAE + r"""

Écris `reparameterize(mu, log_var, rng=None)` et `reparameterize_backward(dz, eps, log_var)`. **Contrat** propre à ces deux fonctions :
- `mu`, `log_var`, `dz` et `eps` sont des tableaux 2-D `(n, d)` de **même forme** (pas de diffusion : `(2, 3)` face à `(1, 3)` est une `ValueError`) ;
- `rng` accepte `None`, un entier (une graine) ou un `np.random.Generator` ; la même graine donne les mêmes tirages ; `eps` est tiré par `standard_normal` du générateur, et la fonction rend `(z, eps)` ;
- `reparameterize_backward` rend `(dmu, dlog_var)`, les gradients de la loss par rapport à `mu` et `log_var` quand on connaît `dz` (le ∂ 25.5 les a dérivés) ; `dmu` est un nouveau tableau, même s'il est égal à `dz`.

La première cellule de vérification essaie tes fonctions : `z` contre sa formule, la reproductibilité, la moyenne et la variance de 200 000 tirages, et la passe arrière contre `torch.autograd`. La seconde lance les tests (`-k "test_reparameterize_"`).

Dans tes notes : pourquoi faut-il rendre `eps` ? Pourquoi le gradient par rapport à `log_var` est-il nul quand `eps` vaut 0, et que devient-il quand `log_var` est très négatif ?""",
       todo="# Nothing to write here: reparameterize and reparameterize_backward go in mon_travail/mylearn/vae.py",
       check=RELOAD + CHECK_28, solution=CHECK_28,
       after=tests_after("25.28", "mylearn.vae.reparameterize([[0.0]], [[0.0]], rng=0)", "test_reparameterize_")),

    Ex("25.29", "🔨", 2, 20, "Divergence KL gaussienne et son gradient",
       "écrire la KL d'un code gaussien au prior $\\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ et son gradient dans ta librairie.",
       "∂ 25.7 · ch. 6 (KL) · fiche §25.9.1, §25.9.2", tracks="R, M, C", mylearn="vae.py",
       body=mylearn_howto() + "\n\n" + COMMON_VAE + r"""

Écris `gaussian_kl(mu, log_var)` et `gaussian_kl_grad(mu, log_var)`. **Contrat** propre à ces deux fonctions :
- `mu` et `log_var` sont des tableaux 2-D `(n, d)` de même forme, sans diffusion ;
- `gaussian_kl` rend un tableau `(n,)` : pour chaque exemple, la KL de $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ à $\mathcal{N}(\mathbf{0}, \mathbf{I})$, **sommée** sur les $d$ latents, en nats (la formule du ∂ 25.7) ;
- `gaussian_kl_grad` rend `(dmu, dlog_var)`, les dérivées de la KL de chaque exemple par rapport à `mu` et à `log_var`, de forme `(n, d)` ; `dmu` est un nouveau tableau, même s'il est égal à `mu` ;
- pour des `log_var` proches de 0, préfère `np.expm1(log_var)` à `np.exp(log_var) - 1` (moins d'erreur d'arrondi).

La première cellule de vérification compare ta KL à `torch.distributions.kl_divergence` et ton gradient à `torch.autograd`. La seconde lance les tests (`-k "test_gaussian_kl_"`).

Dans tes notes : en quels `mu` et `log_var` la KL est-elle minimale, et que vaut-elle là ? Pourquoi somme-t-on sur les latents au lieu de moyenner ?""",
       todo="# Nothing to write here: gaussian_kl and gaussian_kl_grad go in mon_travail/mylearn/vae.py",
       check=RELOAD + CHECK_29, solution=CHECK_29,
       after=tests_after("25.29", "mylearn.vae.gaussian_kl([[0.0]], [[0.0]])", "test_gaussian_kl_")),

    Ex("25.30", "📦", 2, 30, "Un VAE en PyTorch : reconstruction + KL",
       "écrire un VAE en PyTorch et sa loss, la reconstruction plus $\\beta$ fois la KL.",
       "Ex 25.28 · Ex 25.29 · Ex 25.16 · fiche §25.9.2, §25.10", thread="MNIST", tracks="R, M, C",
       body=r"""Ce que tu as écrit en NumPy, PyTorch le dérive tout seul. Le VAE de cet exercice suit celui de la fiche : un encodeur dense qui rend, pour chaque latent, une moyenne et une log-variance ; un tirage par la reparamétrisation ; un décodeur qui rend des **logits** (la sigmoïde est dans la loss, plus stable).

**Contrat**
- `VAE30(n_latent=20, hidden=(512, 256))` est un `nn.Module` : `encoder` = `nn.Sequential(Linear(784, hidden[0]), ReLU, Linear(hidden[0], hidden[1]), ReLU, …)` ; `to_mu` et `to_log_var`, deux `nn.Linear(hidden[-1], n_latent)` appliqués à la sortie de l'encodeur ; `decoder` = `nn.Sequential(Linear(n_latent, hidden[-1]), ReLU, …, Linear(hidden[0], 784))`, sans sigmoïde ; les attributs `n_latent` et `hidden` (un tuple) gardent les deux arguments ;
- `encode(x)` rend `(mu, log_var)` ; `forward(x)` rend `(logits, mu, log_var)`, avec `z = mu + exp(0.5 log_var) * eps` et `eps = torch.randn_like(mu)` ;
- `vae_loss_30(x, logits, mu, log_var, beta=1.0)` rend `(loss, rec, kl)`, trois tenseurs : `rec` est la cross-entropy binaire calculée **sur les logits**, **sommée** sur les 784 pixels puis moyennée sur le batch ; `kl` la KL au prior, sommée sur les latents puis moyennée sur le batch ; `loss = rec + beta * kl`.

La vérification regarde les couches, puis la sortie de `forward` (la même graine, la même sortie), et compare ta loss à un calcul fait avec `torch.distributions`. La cellule suivante donne les versions de référence (`VAE`, `vae_loss`, `train_vae`, `vae_eval`, `vae_images`) et entraîne `VAE_C`, le VAE que la suite explore ; la dernière entraîne **ton** VAE avec ta loss (3 epochs en mode rapide) et décode des codes tirés de $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

Dans tes notes : pourquoi sommer la reconstruction sur les pixels (fiche, pièges) ? Que vaut la KL de ton VAE après 3 epochs, en nats, et que dirait-elle en bits ?""",
       todo=TODO_30, check=CHECK_30, solution=SOLUTION_30 + "\n\n\n" + CHECK_30,
       after=[("code", GIVEN_VAE), ("code", GIVEN_30_HELPER), ("code", EXPERIMENT_30)]),

    Ex("25.31", "🔮", 2, 15, "Prédire : décoder des tirages N(0, I) avec un AE puis avec un VAE",
       "prédire, puis mesurer, ce que deviennent des codes tirés au hasard avec un autoencodeur classique et avec un VAE.",
       "Ex 25.30 · Ex 25.19 · fiche §25.9, §25.10", thread="MNIST", tracks="R, C",
       body=r"""La fiche le dit : des codes tirés au hasard donnent des taches avec un autoencodeur classique, et des chiffres avec un VAE. Combien ? Les mêmes 1 000 codes tirés de $\mathcal{N}(\mathbf{0}, \mathbf{I})$ (`Z_31`, 20 nombres chacun) passent dans le décodeur de `AE_DEEP` et dans celui de `VAE_C`, puis un petit classifieur de chiffres (`CLF_31`, entraîné par la cellule donnée) dit quel chiffre il voit dans chaque image, et avec quelle probabilité.

**Avant d'exécuter**, écris tes trois prédictions (`True` ou `False`) :
a) pour `AE_DEEP`, le chiffre le plus souvent prédit couvre-t-il **plus de 20 %** des 1 000 images (deux fois la part d'un chiffre sur dix) ?
b) pour `VAE_C`, la même question ;
c) **plus de la moitié** des images de `VAE_C` sont-elles reconnues avec confiance (probabilité > 0,9) ?""",
       given=GIVEN_31, todo=TODO_31, solution=SOLUTION_31, hypothesis=True, record=RECORD_31,
       after=[("code", guarded(EXPERIMENT_31, ["ae_top_31", "vae_top_31", "vae_conf_31"],
                               "⏳ Ex 25.31 : écris d'abord tes trois prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : que t'apprend la répartition des chiffres prédits sur ce que les tirages "
                     "rencontrent dans l'espace latent de `AE_DEEP` (compare les normes affichées) ? Pourquoi un "
                     "classifieur peut-il être « sûr de lui » sur une tache ? Qu'est-ce qui limite la netteté des "
                     "images du VAE ici (25.36) ?")]),

    Ex("25.32", "🔬", 2, 30, "Un VAE est stochastique : sorties répétées, bruit relatif, interpolations",
       "mesurer ce que le tirage du code change aux sorties d'un VAE, et sa tolérance au bruit face à un autoencodeur.",
       "Ex 25.30 · Ex 25.31 · fiche §25.10", thread="MNIST", tracks="C",
       body=r"""Le livre passe huit fois le même 2 dans son VAE et obtient huit sorties voisines ; il bruite les codes de 10 % puis de 30 % de leur valeur. Tu refais ces expériences, en comparant `VAE_C` à `AE_DEEP`, et tu mesures avec `classify_31` (25.31) la part des sorties qui restent des chiffres nets.

**Contrat**
- `repeat_32(vae, x, k, seed=0)` passe $k$ fois **une** image `x` (784 pixels) dans le VAE : `mu, log_var = vae.encode(x)` (une ligne), puis, après `torch.manual_seed(seed)`, `z = mu + exp(0.5 log_var) * torch.randn(k, n_latent)`, décodé (la sigmoïde des logits), en mode évaluation et sans graphe : un tableau NumPy `(k, 784)` ;
- `relative_noise_32(model, mu, frac, seed=0)` rend les codes `mu` `(N, d)` bruités de `frac` de leur propre valeur : $\mu \,(1 + \text{frac} \cdot u)$, avec $u$ uniforme dans $[-1, 1]$ tiré par `np.random.default_rng(seed).uniform(-1, 1, size=mu.shape)` ; un nouveau tableau `float32` (l'argument `model` ne sert qu'à garder la même signature pour les deux modèles).

La vérification compare tes fonctions à des calculs faits avec les mêmes graines. La cellule suivante montre huit passages d'un 2, compare `VAE_C` et `AE_DEEP` face à des bruits de 10 %, 30 % et 100 %, et interpole d'un 4 à un 9 avec ta `interpolate_latents`.

Dans tes notes : d'où vient la variété des huit sorties ? Lequel des deux modèles tolère le mieux un code bruité, et pourquoi (pense à ce que la KL impose aux codes) ? L'interpolation du VAE passe-t-elle par des chiffres à chaque étape ?""",
       todo=TODO_32, check=CHECK_32, solution=SOLUTION_32 + "\n\n\n" + CHECK_32, after=[("code", EXPERIMENT_32)]),

    Ex("25.33", "🎨", 2, 30, "Reproduire le nuage latent d'un VAE 2D et la grille −3…3 (fig. 25.67-25.68)",
       "dessiner l'espace latent d'un VAE à 2 latents et le comparer au prior.",
       "Ex 25.30 · Ex 25.18 · fiche §25.9.1, §25.10", thread="MNIST", tracks="C",
       body=r"""Le livre dessine les codes d'un VAE à 2 latents (fig. 25.67), puis décode une grille de codes de −3 à 3 sur chaque axe (fig. 25.68). La cellule donnée entraîne `VAE_2D` et donne `grid_codes`, la version de référence de ta `grid_codes_18`.

**Contrat** : `plot_vae_map_33(vae, x, labels, lim=3.0, n=15)` trace deux panneaux côte à côte : à gauche, les moyennes $\boldsymbol{\mu}$ des codes de `x` (`encode_mu`), colorées par chiffre, avec une légende, les cercles de rayon 1, 2 et 3 autour de l'origine, et les limites $-$`lim` à `lim` sur les deux axes ; à droite, les $n \times n$ images que `vae_images` tire de `grid_codes((-lim, lim, -lim, lim), n)`, en **une** image de $28n \times 28n$ pixels, le code du coin en haut à gauche en premier ; elle finit par `plt.show()` et rend la mosaïque.

La vérification compare le coin de ta mosaïque au décodage de $(-3, 3)$, puis affiche la part des $\boldsymbol{\mu}$ à moins de 1, 2 et 3 de l'origine, face à celle de tirages de $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

Dans tes notes : compare ce nuage à la carte de 25.18 : où sont les codes, reste-t-il des zones vides ? La grille se décode-t-elle en chiffres « presque partout », comme le dit la fiche ? Pourquoi les $\boldsymbol{\mu}$ seuls ne remplissent-ils pas tout à fait la gaussienne ?""",
       given=GIVEN_33, todo=TODO_33, check=CHECK_33, solution=SOLUTION_33 + "\n\n\n" + CHECK_33),

    Ex("25.34", "📈", 2, 15, "Diagnostiquer des courbes reconstruction/KL (effondrement du posterior)",
       "reconnaître, sur des courbes réelles, l'effondrement du posterior, un $\\beta$ trop fort et une montée progressive de $\\beta$.",
       "Ex 25.30 · fiche (Au-delà du livre : β-VAE, effondrement du posterior)", thread="MNIST", tracks="R, M, C",
       body=r"""`HISTORY_34`, calculé à l'avance par `tools/chapters/history_ch25.py`, contient quatre entraînements du même VAE (20 latents, 20 epochs sur 20 000 chiffres), qui ne diffèrent que par $\beta$ : l'un garde $\beta = 1$, un autre $\beta = 20$, un autre $\beta = 100$, et le dernier fait monter $\beta$ de 0 à 1 pendant les 10 premières epochs (le *KL annealing*). Tu ne sais pas lequel est lequel. Chaque run donne, epoch par epoch, la reconstruction et la KL de validation (en nats par image) et le nombre de **latents actifs** (dont la KL moyenne dépasse 0,05 nat). La cellule donnée trace les trois courbes.

Réponds en lisant les courbes, ou en calculant sur `HISTORY_34` (des listes, l'epoch 1 à l'indice 0) :
a) `collapsed_34` : la lettre du run dont le posterior s'est effondré (la KL presque nulle pour tous les latents) ;
b) `active_34` : dans le run qui finit avec 1 à 5 latents actifs, combien en a-t-il à la dernière epoch ?
c) `peak_34` : dans le run dont la KL part le plus haut, la KL de la première epoch moins celle de la dernière, en nats (1 décimale) ;
d) `gap_34` : la dernière reconstruction du run de b), moins la dernière reconstruction du run dont la KL finit la plus haute, en nats (1 décimale) ;
e) `mapping_34` : les lettres des runs à $\beta = 1$, $\beta = 20$, $\beta = 100$ et $\beta$ montant, dans cet ordre (par exemple `"ABCD"`).""",
       given=_history_34_cell() if HISTORY["collapse34"] else "HISTORY_34 = {}", todo=TODO_34, check=CHECK_34, solution=TODO_34.split("collapsed_34")[0] +
       "collapsed_34, active_34, peak_34, gap_34, mapping_34 = " + repr(ANSWERS_34[:5]).strip("()"),
       record=_record_34() if HISTORY["collapse34"] else "",
       after=[("md", "Dans tes notes : à quoi reconnais-tu l'effondrement du posterior sur les trois courbes ? Que coûte "
                     "un $\\beta$ fort en reconstruction, et que gagne-t-on en échange (25.36) ? Pourquoi la KL du run à "
                     "$\\beta$ montant part-elle si haut, et pourquoi finit-il avec à la fois une meilleure reconstruction "
                     "et plus de latents actifs que $\\beta = 1$ ? Quelle courbe surveillerais-tu en entreprise pour repérer un "
                     "effondrement tôt ?")]),

    Ex("25.35", "🛠️", 2, 20, "Exporter le décodeur comme générateur autonome et le tester",
       "exporter le décodeur d'un VAE, le recharger sans le notebook, et le couvrir par des tests qui attrapent de vrais bugs.",
       "Ex 25.30 · ch. 23 (Ex 23.22, state_dict et rechargement) · fiche §25.10", thread="MNIST", tracks="R, C",
       body=r"""Une fois le VAE entraîné, on jette l'encodeur et on garde le décodeur : c'est un **générateur**. Tu l'exportes dans un fichier qu'un autre programme peut recharger, sans rien de ce notebook, et tu écris deux tests ; la vérification s'assure qu'ils passent sur ton générateur **et** qu'ils échouent sur deux générateurs cassés (un test qui ne peut pas échouer ne vérifie rien).

**Contrat**
- `export_decoder_35(vae, path)` enregistre avec `torch.save`, dans le fichier `path`, le dictionnaire `{"n_latent": vae.n_latent, "hidden": list(vae.hidden), "state_dict": vae.decoder.state_dict()}` (le décodeur seul) ;
- `load_generator_35(path)` relit ce fichier (`torch.load(path, weights_only=True)`), reconstruit le décodeur (les couches de 25.30), charge ses poids, le passe en mode évaluation, et rend une fonction `generate(n, seed=0)` : `z = torch.randn(n, n_latent, generator=torch.Generator().manual_seed(seed))`, puis la sigmoïde des logits, × 255, arrondie (`torch.round`), en tableau NumPy `uint8` de forme `(n, 28, 28)` ; elle n'utilise que `torch`, `nn` et `numpy` ;
- tes tests `test_generator_shape_35()` (5 images de 28 × 28 en `uint8`) et `test_generator_seed_35()` (la même graine donne les mêmes images, deux graines différentes des images différentes) lisent le fichier par `GEN_PATH` ; ils tournent dans un autre processus, avec `wb.run_pytest`.

La vérification exporte le décodeur de `VAE_C`, recharge ton générateur et compare ses images à celles du décodeur, puis lance tes tests sur ton générateur et sur les deux générateurs cassés de la cellule donnée.

Dans tes notes : que faudrait-il encore livrer avec ce fichier pour qu'une autre équipe s'en serve (pense à la fiche modèle du ch. 23) ? Pourquoi `weights_only=True` ?""",
       given=GIVEN_35, todo=TODO_35, check=CHECK_35, solution=SOLUTION_35 + "\n\n\n" + CHECK_35),

    Ex("25.36", "🔬", 3, 40, "β-VAE : échanger la netteté contre la régularité",
       "mesurer le compromis entre reconstruction et KL quand $\\beta$ varie, et ce qu'il change aux images générées.",
       "Ex 25.30 · Ex 25.34 · fiche (Au-delà du livre : β-VAE)", thread="Fashion-MNIST", tracks="M, C",
       body=r"""Le $\beta$ de la loss règle le poids de la KL. Tu entraînes le même VAE (16 latents) sur Fashion-MNIST avec $\beta$ = 0,5, 1, 4 et 16, et tu compares la reconstruction, la KL, le nombre de latents actifs, les reconstructions et les images générées.

**Contrat** : `beta_sweep_36(betas, epochs, seed=2536, x=None, n_latent=16)` : pour chaque $\beta$ de `betas`, `torch.manual_seed(seed)`, `model = VAE(n_latent)`, `train_vae(model, x, epochs, beta=beta, seed=seed)` (avec `x = X_FASHION` si `x` vaut `None`), puis `rec, kl, per_latent = vae_eval(model, XF_TEST)` ; elle rend `{beta: {"rec": ..., "kl": ..., "active": le nombre de latents dont la KL moyenne dépasse 0,05 nat, "model": model}}`.

La vérification compare ta fonction à un entraînement fait à la main sur un cas d'essai. La cellule suivante lance les quatre entraînements (4 epochs en mode rapide, une trentaine de secondes), trace la reconstruction en fonction de la KL, et montre les reconstructions et des tirages pour chaque $\beta$.

Dans tes notes : comment la reconstruction et la KL évoluent-elles avec $\beta$ ? Que deviennent les images générées, et les reconstructions ? Combien de latents restent actifs à $\beta = 16$, et qu'est-ce que cela dit du code ? Quel $\beta$ choisirais-tu pour un générateur, et pour un détecteur d'anomalies ?""",
       todo=TODO_36, check=CHECK_36, solution=SOLUTION_36 + "\n\n\n" + CHECK_36, after=[("code", EXPERIMENT_36)]),

    Ex("25.37", "🔬", 3, 45, "Détecter des anomalies par l'erreur de reconstruction",
       "utiliser l'erreur de reconstruction comme score d'anomalie, la mesurer par la ROC-AUC, et en voir la limite.",
       "Ex 25.24 · Ex 25.17 · ch. 3 (ROC-AUC, `mylearn.metrics`) · fiche (Au-delà du livre : anomalies)",
       thread="MNIST", tracks="R, C",
       body=r"""Un autoencodeur entraîné sur des données « normales » reconstruit mal ce qu'il n'a jamais vu : son erreur de reconstruction est un **score d'anomalie**. Tu entraînes `make_deep_ae(20)` sur tous les chiffres **sauf les 7**, puis tu mesures s'il repère les 7, les vêtements, et des chiffres très pâles.

**Contrat**
- `fit_without_37(digit, epochs, seed=2537)` crée `make_deep_ae(20)` juste après `torch.manual_seed(seed)` et l'entraîne par `train_ae(model, x, epochs, seed=seed)` sur les images de `X_MNIST` qui ne sont **pas** `digit` ; elle rend le modèle ;
- `scores_37(model, x)` rend le score de chaque image : la MSE entre l'image et sa reconstruction ;
- `auc_37(normal, anomalies)` rend la ROC-AUC des scores (un grand score = plus anormal) avec **ta** `mylearn.metrics.roc_auc` (ch. 3), le label 0 pour les scores de `normal` et 1 pour ceux de `anomalies`.

La vérification compare ta ROC-AUC à celle de scikit-learn et tes scores à un calcul direct. La cellule suivante entraîne le modèle sans les 7 (5 epochs en mode rapide), puis calcule trois ROC-AUC et trace les histogrammes des scores.

Dans tes notes : les 7 sont-ils faciles à repérer, et les vêtements ? Que se passe-t-il avec les chiffres pâles, et pourquoi est-ce une limite de toute détection par reconstruction (25.Q11) ? Comment choisirais-tu un seuil en pratique, et que mesurerais-tu (ch. 3) ?""",
       todo=TODO_37, check=CHECK_37, solution=SOLUTION_37 + "\n\n\n" + CHECK_37, after=[("code", EXPERIMENT_37)]),

    Ex("25.38", "🔨", 4, 150, "Un VAE entièrement en NumPy avec mylearn",
       "assembler un VAE complet avec ta librairie (couches, passe arrière, Adam, `vae.py`) et vérifier sa passe arrière.",
       "Ex 25.28 · Ex 25.29 · ch. 18 (Ex 18.14, `mlp_backward`) · ch. 19 (Adam) · fiche §25.9.2", thread="MNIST",
       tracks="M, C",
       body=r"""Tout ce qu'il faut pour un VAE est déjà dans ta librairie : `init_mlp` et `mlp_forward` (ch. 16), `binary_cross_entropy_with_logits` et `mlp_backward` (ch. 18), `Adam` (ch. 19), et `vae.py`. Tu l'assembles **sans nouvelle fonction de librairie** : un encodeur `784 → hidden → 2d` (ReLU au milieu ; les $d$ premières sorties sont $\boldsymbol{\mu}$, les $d$ suivantes $\log\boldsymbol{\sigma}^2$), un décodeur `d → hidden → 784` (des logits).

**Contrat**
- `init_numpy_vae_38(n_latent=8, hidden=128, seed=0)` rend `{"enc": init_mlp([784, hidden, 2 * n_latent], rng=rng), "dec": init_mlp([n_latent, hidden, 784], rng=rng)}`, avec **un seul** `rng = np.random.default_rng(seed)`, l'encodeur d'abord ;
- `numpy_vae_loss_38(params, x, eps)` fait une passe avant et une passe arrière sur le batch `x` `(n, 784)`, avec le bruit `eps` `(n, d)` **donné** : `z = mu + exp(0.5 log_var) * eps` ; `rec` = la cross-entropy binaire des logits **sommée** sur les pixels et moyennée sur le batch ; `kl` = la moyenne sur le batch de `gaussian_kl` ; `loss = rec + kl` ; elle rend `(loss, rec, kl, grads)`, avec `grads = {"enc": [(dW, db), ...], "dec": [(dW, db), ...]}` comme `params` ; la ReLU et sa dérivée sont `_relu_38` et `_relu_derivative_38` (données dans la cellule) ;
- `train_numpy_vae_38(params, x, epochs, lr=1e-3, batch_size=128, seed=0)` entraîne `params` sur place avec `mylearn.optim.Adam` sur la liste à plat `[W1, b1, W2, b2, ...]` de l'encodeur puis du décodeur ; `rng = np.random.default_rng(seed)` : à chaque epoch `rng.permutation(N)`, puis pour chaque mini-batch `eps = rng.standard_normal((n, d))` ; elle rend la liste des losses moyennes de chaque epoch.

La vérification construit le même petit VAE en PyTorch (`float64`, poids copiés, même `eps`) et compare ta loss et **tous** tes gradients à `torch.autograd`, avant tout entraînement. La cellule suivante entraîne ton VAE NumPy (5 epochs sur 5 000 chiffres en mode rapide) et décode des codes tirés de $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

Dans tes notes : par où le gradient arrive-t-il sur $\boldsymbol{\mu}$ et sur $\log\boldsymbol{\sigma}^2$ (deux chemins) ? Pourquoi la vérification contre `torch.autograd` passe-t-elle **avant** l'entraînement ? Combien de temps prend une epoch, face au VAE PyTorch ?""",
       todo=TODO_38, check=CHECK_38, solution=SOLUTION_38 + "\n\n\n" + CHECK_38, after=[("code", EXPERIMENT_38)]),

    Ex("25.39", "🏆", 4, 120, "Défi débruitage Fashion-MNIST : le meilleur PSNR sous contrainte",
       "concevoir et entraîner un débruiteur de vêtements qui atteint un PSNR fixé, sous contraintes de taille et de temps.",
       "Ex 25.26 · Ex 25.27 · fiche §25.8", thread="Fashion-MNIST", tracks="C",
       body=r"""Le défi : débruiter des vêtements abîmés par un bruit gaussien d'écart-type 0,3 (écrêté dans $[0, 1]$), avec un PSNR moyen d'au moins `GOAL_39` dB, d'abord sur les 1 000 vêtements de validation de la partie B, puis sur 1 000 **autres** vêtements de test, cachés.

**Règles**
- `make_denoiser_39()` rend un `AutoEncoder` (des lignes de 784 pixels en entrée, des lignes de 784 pixels dans $[0, 1]$ en sortie) d'au plus 100 000 paramètres ;
- `train_39(model, x_clean)` l'entraîne sur les vêtements propres `X_FASHION` (tous, ou une partie) ; tout est permis : architecture, boucle (`train_denoiser` ou la tienne, avec le bruit de l'énoncé), optimiseur, nombre d'epochs, augmentation ; l'entraînement doit tenir en 300 s en mode rapide (900 s en mode complet) ;
- règles d'honneur : n'utilise ni `XF_TEST` ni `VAL_NOISY_B` pour entraîner, et ne règle rien sur le test caché ; il n'est révélé qu'**une fois par session**, et le résultat est figé.

La vérification compte les paramètres, chronomètre `train_39`, mesure le PSNR de validation, puis, si l'objectif est atteint, révèle le test. Le débruiteur de 25.26 (32 filtres, 3 epochs sur 5 000 images) n'y arrive pas : à toi de trouver ce qui compte.

Dans tes notes : qu'as-tu essayé, dans quel ordre, et qu'est-ce qui a le plus aidé (largeur, nombre d'images, epochs, learning rate, architecture sans pooling) ? Combien de secondes ton entraînement laisse-t-il de marge ?""",
       given=GIVEN_39.replace("{GOAL}", repr(HISTORY["calib"]["goal_39"])), todo=TODO_39, check=CHECK_39,
       solution=SOLUTION_39 + "\n\n\n" + CHECK_39),
])


PARTS: list = [PART_A, PART_B, PART_C]
NEXT_SESSION: list = []
# PARTS_END


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 25.Q1 à 25.Q12, 25.R1 à 25.R3, 25.1 à 25.4, 25.6, 25.7 | vérifier tes réponses courtes | "
            "🧠 🔁 ✏️ ∂ | ★ à ★★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# 25 · Autoencodeurs et VAE — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch25_autoencodeurs/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 25`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 25 · Autoencodeurs et VAE — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Construire et entraîner en PyTorch des autoencodeurs denses, convolutifs et débruiteurs, et "
               "explorer leur espace latent (carte 2D, bruit, interpolation).\n"
               "- Relier l'autoencodeur linéaire à l'ACP ; mesurer un débruitage (PSNR).\n"
               "- Écrire en NumPy le cœur du VAE (`mylearn.vae`), entraîner un VAE, régler β, générer, détecter "
               "des anomalies.\n\n"
               "**Rappel express.** $\\mathbf{z} = f_\\theta(\\mathbf{x})$, $\\hat{\\mathbf{x}} = g_\\phi(\\mathbf{z})$, "
               "la cible est l'entrée. Convolution transposée : $H_{out} = (H_{in} - 1)s - 2p + k + p_{out}$. "
               "PSNR $= 10 \\log_{10}(1/\\text{MSE})$ pour des pixels dans $[0, 1]$. VAE : "
               "$\\mathbf{z} = \\boldsymbol{\\mu} + e^{\\frac{1}{2}\\log\\boldsymbol{\\sigma}^2} \\odot "
               "\\boldsymbol{\\varepsilon}$, loss = reconstruction + $\\beta$ KL, "
               "$\\mathrm{KL} = \\tfrac{1}{2}\\sum_j (\\mu_j^2 + \\sigma_j^2 - 1 - \\ln\\sigma_j^2)$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 25.14 à 25.39 (parties A à C : les autoencodeurs denses, convolutifs et "
             "débruiteurs, puis le VAE, ta librairie `vae.py` et le défi) seront ajoutés à la prochaine session de "
             "génération ; relance alors `python tools/start_chapter.py 25` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu expliquer le rôle du goulot, et pourquoi un autoencodeur entraîné sur une seule image ne "
               "comprime rien ?\n"
               "2. Sais-tu dire ce que l'autoencodeur linéaire a en commun avec l'ACP, et ce qu'il ne retrouve pas ?\n"
               "3. Sais-tu écrire la reparamétrisation et la KL gaussienne, et expliquer pourquoi un VAE sait "
               "générer ?\n\n"
               "**Pour aller plus loin** : l'introduction aux VAE de Kingma et Welling (2019), citée dans la fiche. "
               "La suite : le ch. 26, l'apprentissage par renforcement, puis les GAN au ch. 27." + later)]


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
