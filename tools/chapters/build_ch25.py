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
Parts A to C (exercises 25.14 to 25.39) come at the second generation session: dense autoencoders (one image,
MNIST, the latent space, interpolation, the bottleneck, linear autoencoder and PCA), convolutional and denoising
autoencoders, and the VAE (the learner's mylearn.vae, a PyTorch VAE, β, anomalies, a VAE in NumPy, the
denoising challenge).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Ex, Paper, Part, badge, check_unique_keys, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402,F401

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
          "3 minutes, c'est 180 secondes": 0.048}'''),
    ]),
    Paper("25.Q3", "Mélange de contenu ou mélange de paramètres ?", [
        ("a", LETTER, '"A"', r'''mistakes={"on mélange les pixels, pas ce qui décrit les objets": "B"}'''),
        ("b", WORD, 'letter_mix("LUNE", "SOIR")',
         r'''mistakes={"quand il y a deux lettres au milieu, la règle prend la plus proche du début de l'alphabet": "PRLL",
          "la lettre du milieu, pas la première lettre de chaque paire": "LOIE"}'''),
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
         r'''mistakes={"un biais par neurone de la couche de sortie, qui a un neurone par pixel": 10}'''),
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
          "une seule condition : le livre en pose deux": "C",
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
          "il manque une exigence : relis le §25.9.2": "AB",
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
          "h_n a trois axes : couches, exemples, état": "(16, 32)"}'''),
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
          "c'est la valeur en bits": 0.21}'''),
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
         r'''decimals=2, mistakes={"ce pixel n'est que dans A : son poids est 1 − t": 0.25,
          "ce pixel est-il aussi dans B ?": 0.85}'''),
        ("e", DEC2, "0.25 * B_25_2['g']",
         r'''decimals=2, mistakes={"ce pixel n'est que dans B : son poids est t, sa valeur l'intensité de B": 0.3,
          "multiplie aussi par l'intensité de B": 0.25}'''),
        ("f", WHOLE, "2", r'''mistakes={"dans un mélange de contenu, chaque disque garde sa place": 1}'''),
        ("g", DEC1, "math.pi * 2.5 ** 2",
         r'''decimals=1, mistakes={"l'aire d'un disque est π r², pas 2 π r": 2 * math.pi * 2.5,
          "relis la question : l'aire du disque mélangé, pas la moyenne des aires": (math.pi + 16 * math.pi) / 2}'''),
        ("h", DEC1, "(math.pi * 1 ** 2 + math.pi * 4 ** 2) / 2",
         r'''decimals=1, mistakes={"relis la question : la moyenne des aires des deux disques": math.pi * 2.5 ** 2,
          "π r² pour chaque disque, pas 2 π r": (2 * math.pi + 8 * math.pi) / 2}'''),
    ]),
    Paper("25.3", "Suivre les formes dans l'autoencodeur convolutif", [
        ("a", SHAPE, '"(64, 16, 14, 14)"',
         r'''mistakes={"le pooling divise la hauteur et la largeur par deux": "(64, 16, 28, 28)",
          "PyTorch range les canaux avant la hauteur et la largeur": "(64, 14, 14, 16)",
          "l'axe des exemples reste en premier": "(16, 14, 14)"}'''),
        ("b", SHAPE, '"(64, 8, 7, 7)"',
         r'''mistakes={"un pooling ne change pas le nombre de canaux : d'où viennent-ils ?": "(64, 7, 7, 7)",
          "les canaux sont ceux de la deuxième convolution": "(64, 16, 7, 7)",
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
          "le code a 3 canaux de 7 × 7": 16.0}'''),
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
          "un biais par canal de sortie (16), pas d'entrée": 32 * 16 * 3 * 3 + 32}'''),
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
         r'''mistakes={"chaque coordonnée de m a pour variance 1/4 + 1/4": 20,
          "les deux tirages sont indépendants : leurs variances s'ajoutent avant la division par 4": 5}'''),
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
          "n'oublie pas le facteur ½": 2 * gaussian_kl(0, 0.01)}'''),
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

# ---------------------------------------------------------------------------
# Parts A to C (exercises 25.14 to 25.39): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "25.14 à 25.23", "des autoencodeurs denses : une seule image, MNIST, l'espace latent, l'interpolation, "
                           "la taille du goulot, l'autoencodeur linéaire face à l'ACP"),
    ("B", "25.24 à 25.27", "des autoencodeurs convolutifs et débruiteurs : 147 latents, un 🐛 de débruiteur, "
                           "sur-échantillonnage ou convolutions à pas"),
    ("C", "25.28 à 25.39", "le VAE : ta librairie `vae.py`, un VAE PyTorch, générer, β, les anomalies, un VAE en "
                           "NumPy, le défi de débruitage"),
]


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
