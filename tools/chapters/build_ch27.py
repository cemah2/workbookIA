#!/usr/bin/env python
"""Build the two notebooks of chapter 27 from a single source (used by Claude).

    python tools/chapters/history_ch27.py          # the multi-seed runs shown by 27.13, 27.14 and the calibrations
    python tools/chapters/build_ch27.py
    python tools/run_all_notebooks.py chapitres/ch27_gan/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch27_gan/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q10, of the recalls R1 to R3 and of the ✏️ paper exercises 27.1
to 27.4 (02_exercices.md). The derivations ∂ 27.5 and 27.6, the estimate 🧮 27.7, the reflection (🗣️ 27.8, ⚖️ 27.9,
📄 27.10) and the interview are corrected with 05_solutions.md.
Parts A and B (exercises 27.11 to 27.28): toy GANs (the learner's wasserstein_1d, a 1-D gaussian, the book's 2-D cloud,
a buggy loop, a too strong discriminator, the saturating loss, the book's round), then DCGANs on MNIST (Radford's
rules, the 2016 architecture of the book, the learner's Fréchet distance on the features of a classifier, mode
collapse on a ring of gaussians, a conditional GAN, the latent space, a full checkpoint, the Fashion-MNIST challenge).
The multi-seed results shown by the predictions and the calibration of the challenge come from
tools/chapters/history_ch27.py (history_ch27.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, bug_letters_mistakes, check_unique_keys, code, guarded, md,  # noqa: E402,F401
                         paper_cells, part_cells, setup_cell, write_notebook)

CHAPTER = "27"
FOLDER = "chapitres/ch27_gan"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math


def dense(n_in, n_out, bias=True):
    """Parameters of nn.Linear(n_in, n_out)."""
    return n_in * n_out + (n_out if bias else 0)


def conv_params(c_in, c_out, k, bias=True):
    """Parameters of nn.Conv2d(c_in, c_out, k) or nn.ConvTranspose2d(c_in, c_out, k)."""
    return c_out * c_in * k * k + (c_out if bias else 0)


def conv_out(h, k, s=1, p=0):
    """Height after nn.Conv2d(kernel_size=k, stride=s, padding=p)."""
    return (h + 2 * p - k) // s + 1


def tconv_out(h, k, s=1, p=0, op=0):
    """Height after nn.ConvTranspose2d(kernel_size=k, stride=s, padding=p, output_padding=op)."""
    return (h - 1) * s - 2 * p + k + op


GAINS_27_2 = [[3, -1, 2], [1, 0, 4], [-2, 5, -3]]          # rows A, B, C; columns 1, 2, 3
MAXIMIN_27_2 = max(min(row) for row in GAINS_27_2)
MINIMAX_27_2 = min(max(col) for col in zip(*GAINS_27_2))
D_REAL_27_3, D_FAKE_27_3 = [0.8, 0.6], [0.3, 0.1]
REAL_TERM_27_3 = -sum(math.log(d) for d in D_REAL_27_3) / 2
FAKE_TERM_27_3 = -sum(math.log(1 - d) for d in D_FAKE_27_3) / 2
'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "F"'
LETTERS = 'the letters, e.g. "AF"'
WHOLE = "a whole number"
NUMBER = "a number"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
ORDER = 'the letters in order, e.g. "ABBA"'


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("27.Q1", "À quoi bon générer des données ?", [
        ("a", LETTERS, '"AB"', r'''mistakes={"il manque l'un des deux usages du livre : relis le §27.1": "A",
          "un seul usage : le livre en donne deux": "B",
          "compresser sans perte n'est pas générer : relis le §27.1": "ABC",
          "retrouver une photo existante, ce n'est pas en créer une nouvelle": "ABD"}'''),
        ("b", TF, "False", r'''mistakes={"un générateur crée de nouveaux échantillons : copierait-il ses données ?": True}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "le discriminateur ne sert qu'à entraîner : que garde-t-on pour fabriquer des données ?",
                                           "C": "relis la fin du §27.4.2 : on jette souvent l'un des deux"})),
        ("d", TF, "False", r'''mistakes={"un générateur peut-il apprendre plus que ce que contenaient ses données ? (encadré ⚠️ du §27.1)": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "c'est le mot du livre, corrigé dans la fiche (§27.3)",
                                           "C": "un GAN n'a pas d'autoencodeur : relis le titre de l'article de 2014"})),
    ]),
    Paper("27.Q2", "Le faussaire, l'inspectrice et les quatre verdicts", [
        ("a", LETTER, '"B"', letters("B", {"A": "le billet est vrai, mais Dawn a-t-elle répondu « vrai » ?",
                                           "C": "un faux positif est un faux pris pour un vrai",
                                           "D": "le billet est vrai : ce n'est pas un négatif correct"})),
        ("b", LETTER, '"C"', letters("C", {"A": "le billet est faux : la réponse « vrai » est-elle correcte ?",
                                           "B": "un faux négatif est un vrai pris pour un faux",
                                           "D": "un vrai négatif est un faux bien repéré"})),
        ("c", LETTER, '"D"', letters("D", {"A": "quand tout va bien, le livre dit que personne n'apprend",
                                           "B": "un vrai pris pour un faux : c'est Dawn qui doit apprendre",
                                           "C": "un faux qui passe : c'est Dawn qui s'est fait duper"})),
        ("d", WHOLE, "1", r'''mistakes={"un seul cas laisse tout le monde tranquille dans la version du livre": 0,
          "compte les cas où quelqu'un apprend, puis retire-les des quatre": 3}'''),
        ("e", TF, "True", r'''mistakes={"−ln 0,9 vaut-il zéro ? (encadré ⚠️ du §27.2.3)": False}'''),
    ]),
    Paper("27.Q3", "Pourquoi le générateur ne voit-il jamais de vrais billets ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "Glenn n'a que des rectangles de papier blanc : il ne voit jamais de vrai billet",
                                           "C": "le générateur ne reçoit aucun label",
                                           "D": "la sortie de D sert à calculer la loss, pas d'entrée au générateur"})),
        ("b", LETTER, '"B"', letters("B", {"A": "il finit pourtant par dessiner des chiffres : par où l'information passe-t-elle ?",
                                           "C": "il ne voit jamais de vraie image, même à la fin"})),
        ("c", TF, "False", r'''mistakes={"le générateur apprend par un gradient : faut-il pouvoir le dériver ?": True}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "le bruit ne dit pas quelle classe produire (sauf dans un GAN conditionnel)",
                                           "C": "le bruit est l'entrée du générateur, pas une régularisation"})),
        ("e", TF, "False", r'''mistakes={"un GAN a-t-il un encodeur ? relis le §27.4.2": True}'''),
    ]),
    Paper("27.Q4", "Le round d'apprentissage du livre en quatre temps", [
        ("a", ORDER, '"BCAC"',
         r'''mistakes={"le round commence par un vrai billet": "CBAC",
          "le générateur apprend deux fois par round, la seconde fois à la fin": "BCCA",
          "les deux étapes du discriminateur ne se suivent pas": "BACC",
          "un faux qui dupe D sert à D, pas à G : quelle lettre ?": "BCCC"}'''),
        ("b", WHOLE, "2", r'''mistakes={"l'étape du générateur est répétée": 1,
          "compte seulement les étapes où G peut apprendre": 4}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "si D apprenait aussi, il apprendrait à se laisser duper",
                                           "C": "sans D, comment le générateur saurait-il qu'il est repéré ?"})),
        ("d", TF, "True", r'''mistakes={"relis l'algorithme 1 de Goodfellow et coll. (encadré ⚠️ du §27.2.3)": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "detach ne libère pas de mémoire du générateur",
                                           "C": "detach s'applique aux faux, pas au discriminateur"})),
    ]),
    Paper("27.Q5", "Coopératifs ou antagonistes : somme nulle, minimax et équilibre de Nash", [
        ("a", TF, "True", r'''mistakes={"relis la formule min sur G, max sur D (fiche §27.4.4)": False}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "c'est la définition du livre, corrigée dans la fiche",
                                           "C": "aux échecs, somme nulle, il y a pourtant un gagnant"})),
        ("c", TF, "True", r'''mistakes={"chaque jeton gagné par l'un est-il perdu par un autre ?": False}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "égalité des losses n'est pas la définition",
                                           "C": "une loss nulle voudrait dire que D ne se trompe jamais"})),
        ("e", NUMBER, "0.5", r'''decimals=2, mistakes={"à l'équilibre, D ne sait plus distinguer : il n'a aucune raison de dire « faux »": 0.0,
          "D ne sait plus distinguer : il n'a aucune raison de dire « vrai » à coup sûr": 1.0,
          "c'est la valeur de la loss, pas la sortie de D": 0.693}'''),
    ]),
    Paper("27.Q6", "Trois modèles, des couches partagées et un discriminateur gelé", [
        ("a", WHOLE, "3", r'''mistakes={"il y a aussi le modèle combiné du livre": 2}'''),
        ("b", TF, "False", r'''mistakes={"relis le §27.5 : copies ou mêmes couches ?": True}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "un seul optimiseur ferait avancer D pendant l'étape de G",
                                           "C": "pas besoin de copie : PyTorch construit le graphe à la volée"})),
        ("d", TF, "True", r'''mistakes={"backward() remplit le .grad de tous les poids qui ont servi au calcul, D compris": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "le gradient traverse D : ses poids reçoivent bien un .grad",
                                           "C": "sous torch.no_grad(), le générateur ne recevrait aucun gradient"})),
    ]),
    Paper("27.Q7", "Pourquoi juger des mini-batchs plutôt qu'un point isolé", [
        ("a", WHOLE, "math.ceil(10_000 / 32)", r'''mistakes={"le dernier mini-batch, incomplet, compte aussi": 312,
          "un pas par mini-batch, pas un pas par point": 10_000,
          "le discriminateur fait un pas par mini-batch (le générateur aussi, mais on ne compte que D)": 626}'''),
        ("b", TF, "False", r'''mistakes={"combien de valeurs reçoit l'entrée du discriminateur du livre ?": True}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "le dropout ne donne aucune information sur les autres points",
                                           "C": "un learning rate ne change pas ce que voit D"})),
        ("d", LETTER, '"B"', letters("B", {"A": "68 % est la règle d'une seule dimension (encadré ⚠️ du §27.5)",
                                           "C": "95 % correspond à deux écarts-types, en une dimension"})),
        ("e", WHOLE, "(4 + 1) * 16 + (16 + 1) * 2", r'''mistakes={"chaque neurone a aussi un biais": 4 * 16 + 16 * 2,
          "18 est le nombre de neurones, pas de paramètres": 18,
          "la seconde couche reçoit les 16 sorties de la première": (4 + 1) * 16 + (4 + 1) * 2}'''),
    ]),
    Paper("27.Q8", "Les règles de Radford : vrai ou faux", [
        ("a", TF, "True", r'''mistakes={"relis la première règle du §27.6.1": False}'''),
        ("b", TF, "False", r'''mistakes={"ce sont les valeurs du livre : l'article utilise le même learning rate pour les deux": True}'''),
        ("c", TF, "False", r'''mistakes={"c'est la coquille du livre : dans quel réseau va la Leaky ReLU ?": True}'''),
        ("d", TF, "True", r'''mistakes={"relis la règle de la batchnorm dans la fiche §27.6.1": False}'''),
        ("e", TF, "True", r'''mistakes={"une tanh rend des valeurs entre −1 et 1 : les vrais pixels doivent être dans la même plage": False}'''),
    ]),
    Paper("27.Q9", "Grandes images et croissance progressive", [
        ("a", WHOLE, "1000 * 1000 * 3", r'''mistakes={"une photo couleur a trois canaux": 1_000_000,
          "1 000 × 1 000, pas 1 000 + 1 000": 6_000}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "le format de fichier n'y est pour rien",
                                           "C": "la taille du bruit n'est pas le problème des grandes images"})),
        ("c", LETTER, '"B"', letters("B", {"A": "on commence petit, pour que l'essentiel du calcul se fasse sur de petites images"})),
        ("d", TF, "False", r'''mistakes={"relis l'encadré ⚠️ du §27.7.1 : comment arrivent les nouvelles couches ?": True}'''),
        ("e", WHOLE, "len([4, 8, 16, 32, 64, 128, 256, 512, 1024])", r'''mistakes={"compte aussi 4 × 4 et 1 024 × 1 024": 7,
          "compte aussi l'une des deux extrémités": 8,
          "en doublant le côté, pas en le multipliant par 4": 5}'''),
    ]),
    Paper("27.Q10", "Effondrement de mode total ou partiel", [
        ("a", LETTER, '"B"', letters("B", {"A": "c'est l'explication du livre, corrigée dans la fiche (§27.7.2)",
                                           "C": "rien à voir avec train() ou eval()"})),
        ("b", LETTER, '"B"', letters("B", {"A": "il reste quelques 7 : le générateur produit encore plusieurs choses"})),
        ("c", TF, "True", r'''mistakes={"si ce chiffre trompe le discriminateur pendant un temps, que vaut −ln D(G(z)) ?": False}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "le discriminateur, lui, continue d'apprendre : que fait-il de ce mode ?"})),
        ("e", LETTERS, '"ABD"', r'''mistakes={"entraîner plus longtemps ne change pas ce que voit le discriminateur": "ABCD",
          "il manque la loss de Wasserstein avec pénalité": "AB",
          "il manque la couche d'écart-type par mini-batch": "BD",
          "il manque la discrimination par mini-batch": "AD"}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("27.R1", "Deux agents qui jouent l'un contre l'autre (self-play)", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis le ch. 26 : en auto-apprentissage, pas besoin d'humain",
                                           "C": "un adversaire aléatoire, c'est l'entraînement ordinaire du ch. 26"})),
        ("b", TF, "False", r'''mistakes={"un GAN a deux réseaux aux rôles différents : partagent-ils leurs poids ?": True}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "deux joueurs parfaits : les croix peuvent-elles forcer la victoire ?",
                                           "C": "deux joueurs parfaits : les ronds peuvent-ils forcer la victoire ?"})),
        ("d", TF, "True", r'''mistakes={"l'adversaire reste-t-il le même pendant l'entraînement ?": False}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "un agent de RL reçoit déjà une récompense",
                                           "C": "relis le §27.4.3 : par où passe le signal qui corrige G ?"})),
    ]),
    Paper("27.R2", "Precision et recall pour juger un générateur (fidélité contre diversité)", [
        ("a", DEC3, "80 / (80 + 30)", r'''decimals=3, mistakes={"precision = TP / (TP + FP), pas TP / (TP + FN)": 0.8,
          "les 30 faux pris pour des vrais sont les faux positifs": 30 / 110,
          "les vrais négatifs ne comptent pas dans la precision": 150 / 200}'''),
        ("b", DEC2, "80 / 100", r'''decimals=2, mistakes={"recall = TP / (TP + FN), pas TP / (TP + FP)": 0.73,
          "le recall regarde les vrais, pas les faux": 0.3}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "c'est le recall qui mesure la diversité"})),
        ("d", LETTER, '"A"', letters("A", {"B": "un seul chiffre parfait : chaque échantillon ressemble à une vraie donnée",
                                           "C": "un seul chiffre : couvre-t-il toute la variété des données ?"})),
        ("e", TF, "True", r'''mistakes={"le FID compare moyennes et covariances : un manque de diversité le fait-il monter ?": False}'''),
    ]),
    Paper("27.R3", "Batchnorm, où la placer, train() ou eval()", [
        ("a", LETTER, '"A"', letters("A", {"B": "les moyennes glissantes servent en évaluation"})),
        ("b", LETTER, '"B"', letters("B", {"A": "en eval(), la sortie ne doit pas dépendre des autres exemples du batch"})),
        ("c", LETTER, '"A"', letters("A", {"B": "relis le §27.6 : le livre la place avant l'activation",
                                           "C": "une batchnorm normalise la sortie d'une couche"})),
        ("d", TF, "True", r'''mistakes={"en eval(), quelle statistique utilise la batchnorm ?": False}'''),
        ("e", WHOLE, "2 * 64", r'''mistakes={"γ et β : deux paramètres appris par canal": 64,
          "la moyenne et la variance glissantes ne sont pas apprises par descente de gradient": 4 * 64}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("27.1", "Dimensions d'un DCGAN pour MNIST : 6 272 neurones, tailles 7 → 14 → 28 et retour", [
        ("a", WHOLE, "7 * 7 * 128", r'''mistakes={"7 × 7 pixels et 128 canaux : multiplie les trois": 7 * 7 + 128}'''),
        ("b", WHOLE, "dense(1024, 6272)", r'''mistakes={"n'oublie pas les biais": 1024 * 6272,
          "un biais par neurone de sortie, pas par entrée": 1024 * 6272 + 1024}'''),
        ("c", WHOLE, "conv_out(28, 4, 2, 1)", r'''mistakes={"relis la formule : le padding compte deux fois (2p), et il y a un + 1": 13,
          "le pas de 2 divise la taille : (H + 2p − k) / s": 27}'''),
        ("d", WHOLE, "conv_out(conv_out(28, 4, 2, 1), 4, 2, 1)", r'''mistakes={"c'est la taille après une seule convolution": 14}'''),
        ("e", WHOLE, "tconv_out(7, 4, 2, 1)", r'''mistakes={"(H − 1)s : la formule part de H − 1": 7 * 2 - 2 + 4,
          "−2p, pas −p": (7 - 1) * 2 - 1 + 4}'''),
        ("f", WHOLE, "conv_params(128, 64, 4, bias=False)", r'''mistakes={"un noyau 4 × 4 a 16 poids par couple de canaux": 128 * 64 * 4,
          "relis l'énoncé : sans biais": conv_params(128, 64, 4)}'''),
        ("g", WHOLE, "conv_params(128, 64, 5)", r'''mistakes={"n'oublie pas les biais, un par canal de sortie": 128 * 64 * 25,
          "le noyau est de 5 × 5": conv_params(128, 64, 4)}'''),
    ]),
    Paper("27.2", "Valeur minimax d'une petite matrice de gains", [
        ("a", LETTER, '"B"', letters("B", {"A": "le pire gain de A est −1 : y a-t-il mieux ?",
                                           "C": "C peut rapporter 5, mais son pire gain est −3"})),
        ("b", NUMBER, "MAXIMIN_27_2", r'''mistakes={"c'est le pire gain de A, pas le meilleur des pires": -1,
          "c'est le meilleur gain possible, pas le gain garanti": 5}'''),
        ("c", NUMBER, "MINIMAX_27_2", r'''mistakes={"Colonne regarde le plus grand gain de Ligne dans chaque colonne, puis prend le plus petit": 0,
          "c'est la pire colonne pour Colonne, pas la meilleure": 5,
          "le plus grand gain de la colonne 3 est 4 : y a-t-il une colonne qui laisse moins ?": 4}'''),
        ("d", TF, "False", r'''mistakes={"compare les valeurs de b) et de c)": True}'''),
        ("e", TF, "False", r'''mistakes={"maximin et minimax diffèrent même ici ; relis la fin de l'encadré 🧮 du §27.4.4": True}'''),
    ]),
    Paper("27.3", "Losses BCE du discriminateur et du générateur à partir de sorties données", [
        ("a", DEC3, "REAL_TERM_27_3", r'''decimals=3, mistakes={"le terme des vrais est une moyenne : divise par 2": 2 * REAL_TERM_27_3,
          "−ln : la loss est positive": -REAL_TERM_27_3,
          "les vrais ont le label 1 : −ln D(x), pas −ln(1 − D(x))": -(math.log(0.2) + math.log(0.4)) / 2}'''),
        ("b", DEC3, "FAKE_TERM_27_3", r'''decimals=3, mistakes={"les faux ont le label 0 : −ln(1 − D(G(z))), pas −ln D(G(z))": -(math.log(0.3) + math.log(0.1)) / 2,
          "le terme des faux est une moyenne : divise par 2": 2 * FAKE_TERM_27_3}'''),
        ("c", DEC3, "REAL_TERM_27_3 + FAKE_TERM_27_3", r'''decimals=3, mistakes={"la loss de D est la somme des deux termes, pas leur moyenne": (REAL_TERM_27_3 + FAKE_TERM_27_3) / 2}'''),
        ("d", DEC3, "-(math.log(0.3) + math.log(0.1)) / 2", r'''decimals=3, mistakes={"la loss non saturante prend −ln D(G(z)), pas −ln(1 − D(G(z)))": FAKE_TERM_27_3,
          "une moyenne sur les deux faux : divise par 2": -(math.log(0.3) + math.log(0.1))}'''),
        ("e", DEC3, "-FAKE_TERM_27_3", r'''decimals=3, mistakes={"le générateur minimax minimise ln(1 − D(G(z))) : une valeur négative": FAKE_TERM_27_3}'''),
        ("f", DEC2, "4 / 4", r'''decimals=2, mistakes={"compare chaque sortie à 0,5 : D dit « vrai » au-dessus, « faux » en dessous": 0.75,
          "les faux sont bien classés quand D(G(z)) < 0,5": 0.5}'''),
    ]),
    Paper("27.4", "À l'équilibre, la loss vaut ln 2 et non 0,5", [
        ("a", DEC3, "2 * math.log(2)", r'''decimals=3, mistakes={"la loss de D additionne le terme des vrais et celui des faux": 0.693,
          "c'est la sortie de D, pas sa loss": 0.5,
          "log de base 10 ? PyTorch utilise le logarithme népérien": 2 * math.log10(2)}'''),
        ("b", DEC3, "math.log(2)", r'''decimals=3, mistakes={"un seul terme pour le générateur": 1.386,
          "c'est la sortie de D, pas la loss du générateur": 0.5}'''),
        ("c", DEC3, "math.exp(-0.5)", r'''decimals=3, mistakes={"résous −ln p = 0,5 : p = e^(−0,5)": 0.5,
          "e^(−0,5), pas e^(0,5)": 1.649,
          "−ln p = 0,5, pas −log₁₀ p": 0.316}'''),
        ("d", DEC3, "-math.log(4)", r'''decimals=3, mistakes={"V est la somme de deux ln(1/2), qui sont négatifs": 1.386,
          "deux espérances, chacune égale à ln(1/2)": -0.693}'''),
        ("e", TF, "True", r'''mistakes={"0,69 ≈ ln 2 : compare avec b)": False}'''),
    ]),
]

for _paper in PAPER:                      # a dict literal with a repeated message would silently lose one
    for _sub in _paper.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_paper.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 27.1 à 27.4)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"F\"`) ; "
               "plusieurs lettres : entre guillemets (`\"AF\"`), dans l'ordre demandé quand l'énoncé en impose un. "
               "Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore "
               "remplies affichent ⏳. Les questions « dans ta copie », les dérivations ∂ 27.5 et 27.6, "
               "l'estimation 🧮 27.7, la réflexion (🗣️ 27.8, ⚖️ 27.9, 📄 27.10) et l'entretien se corrigent avec "
               "`05_solutions.md`.")

# PART0_END

# ---------------------------------------------------------------------------
# Notebook parts A and B (27.11 to 27.28): shared tools
# ---------------------------------------------------------------------------
_HISTORY_FILE = Path(__file__).resolve().parent / "history_ch27.json"
HISTORY = (json.loads(_HISTORY_FILE.read_text(encoding="utf-8")) if _HISTORY_FILE.exists()
           else {"runs13": {}, "curves14": {}, "calib": {}})


def _packed(data) -> str:
    """JSON data packed (zlib + base64) so that a code cell does not show the numbers before the prediction."""
    import base64
    import zlib
    return base64.b64encode(zlib.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")


TOOLS = r'''# Tools for the notebook exercises (parts A and B)
import copy
import json
import math
import re
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from wb.nbtools import StopCheck, as_numpy, checked, filled, fr, refuses, returned, run_pytest_group, same, verdict, written

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                       # a computer without PyTorch (Mac Intel): this chapter is done on Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : ce chapitre se fait sur Colab (00_setup/INSTALL_LOCAL.md). Les "
          "cellules qui utilisent PyTorch affichent ⏳ au lieu de s'exécuter ; les tests de ta librairie tournent "
          "sans lui : python -m pytest tests/test_ch27_gen_metrics.py -q")

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

TEST_GEN = "tests/test_ch27_gen_metrics.py"            # the tests of your library gen_metrics.py
WORK_DIR = Path(tempfile.mkdtemp(prefix="wb_ch27_"))   # a temporary folder for the files the exercises write
LN2 = math.log(2)
N_THREADS = max(1, min(4, torch.get_num_threads())) if torch is not None else 1   # for the DCGANs of part B
# Every training of this notebook runs on the CPU (the GANs are small): the results then do not depend on a GPU.


def pct(x, digits=1):
    """A share written the French way: 0.734 -> '73,4 %'."""
    return f"{100 * x:.{digits}f} %".replace(".", ",")


def unpack(text):
    """Data packed by the builder (zlib + base64 of a JSON text)."""
    import base64
    import zlib
    return json.loads(zlib.decompress(base64.b64decode(text)).decode("utf-8"))


def n_params(model):
    """The number of parameters (learned numbers) of a torch module."""
    return sum(p.numel() for p in model.parameters())


def mlp(sizes, slope=0.2):
    """Dense layers sizes[0] -> sizes[1] -> ... -> sizes[-1], a LeakyReLU(slope) between two of them, nothing at the end."""
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            layers.append(nn.LeakyReLU(slope))
    return nn.Sequential(*layers)


def make_toy_gan(dim=1, nz=4, hidden=16, seed=0):
    """A small generator G: nz -> hidden -> hidden -> dim and a discriminator D: dim -> hidden -> hidden -> 1 (a
    LOGIT: no sigmoid at the end), LeakyReLU(0.2) inside; the weights are drawn with torch.manual_seed(seed)."""
    torch.manual_seed(seed)
    return mlp([nz, hidden, hidden, dim]), mlp([dim, hidden, hidden, 1])


def latent_dim(G):
    """The size of the noise a generator expects: G.nz if it has one, else the inputs of its first Linear layer."""
    if hasattr(G, "nz"):
        return int(G.nz)
    first = next(m for m in G.modules() if isinstance(m, nn.Linear))
    return first.in_features


def sample_noise(n, nz, generator=None, kind="normal"):
    """n noise vectors of size nz: N(0, 1) ("normal") or uniform on [0, 1] ("uniform"), drawn from `generator`."""
    if kind == "normal":
        return torch.randn(n, nz, generator=generator)
    if kind == "uniform":
        return torch.rand(n, nz, generator=generator)
    raise ValueError(f"unknown noise kind {kind!r}")


def generate(G, z):
    """G(z) in eval mode and without a graph, as a float64 NumPy array (G is put back in train mode)."""
    was_training = G.training
    G.eval()
    with torch.no_grad():
        out = G(torch.as_tensor(z, dtype=torch.float32))
    G.train(was_training)
    return out.double().numpy()


def mean_epochs(history, key, steps_per_epoch):
    """The mean of history[key] over each epoch (history: one value per step)."""
    values = np.asarray(history[key], dtype=float)
    n = len(values) // steps_per_epoch
    return values[:n * steps_per_epoch].reshape(n, steps_per_epoch).mean(axis=1)


def add_ellipse(ax, points, color, **kw):
    """Draw the ellipse of the mean ± one standard deviation (along x and along y) of 2-D points."""
    points = np.asarray(points)
    m, s = points.mean(axis=0), points.std(axis=0)
    ax.add_patch(Ellipse(m, 2 * s[0], 2 * s[1], fill=False, color=color, lw=1.5, **kw))
    ax.plot(*m, "+", color=color, ms=8)


def show_digits(images, n_cols=10, title=None, size=0.8):
    """Draw images in [-1, 1] (a tensor or an array of shape (N, 1, 28, 28), (N, 28, 28) or (N, 784)) in a grid."""
    images = np.asarray(as_numpy(images), dtype=np.float32).reshape(-1, 28, 28)
    n_rows = math.ceil(len(images) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size * n_cols, size * n_rows + (0.35 if title else 0)),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image in zip(axes.flat, images):
        ax.imshow(image, cmap="gray", vmin=-1, vmax=1)
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


print(f"tools ready · torch {torch.__version__ if torch is not None else 'absent'} · {torch.get_num_threads() if torch is not None else 0} "
      f"threads · tests: {TEST_GEN}")'''

RELOAD = 'mylearn = wb.load_mylearn("learner", missing_ok=True, fallback="ref", chapter="27")   # reload your saved files\n'


def ref_check(check: str) -> str:
    """The solutions version of a check cell: the tests run on the reference."""
    return check.replace('run_pytest_group(TEST_GEN, "', 'run_pytest_group(TEST_GEN, impl="ref", keyword="')


def tests_after(ex_id: str, probe: str, keyword: str) -> list:
    cell = (f"if written(lambda: {probe}):\n"
            f'    run_pytest_group(TEST_GEN, "{keyword}")\n'
            "else:\n"
            f'    print("⏳ Ex {ex_id} : les tests attendent tes fonctions.")')
    return [("check", cell), ("solution", ref_check(cell))]


def mylearn_howto() -> str:
    return ("> **Mode d'emploi mylearn** (comme aux chapitres précédents) : ouvre `mon_travail/mylearn/gen_metrics.py` "
            "(créé par `python tools/start_chapter.py 27`), lis la docstring, remplace chaque `raise "
            "NotImplementedError(...)` de l'exercice par ton code et **enregistre**. NumPy et la bibliothèque "
            "standard seulement (pas de SciPy : il sert d'oracle aux tests). La première cellule de vérification "
            "recharge ta librairie et l'essaie sur quelques cas ; la seconde lance les tests de l'exercice. "
            "`python -m pytest tests/test_ch27_gen_metrics.py -q` les lance tous dans un terminal.")


COMMON_GEN = ("**Contrat commun de `gen_metrics.py`** (les docstrings, plus ces précisions que vérifient les tests) : "
              "les listes (d'entiers compris) et les tableaux sont acceptés, et jamais modifiés (`np.sort(x)`, pas "
              "`x.sort()`) ; chaque fonction rend un `float` Python positif ou nul (`float(...)`) ; une entrée "
              "invalide lève `ValueError`, jamais une autre erreur : un NaN ou un infini, une forme qui ne convient "
              "pas.")

# ---------------------------------------------------------------------------
# Part A: toy GANs (27.11 to 27.19)
# ---------------------------------------------------------------------------
GIVEN_A = r'''# The data of part A: 10 000 numbers drawn from N(4, 1.25²) (wb.synth.gaussian_1d), as a tensor (10 000, 1)
torch.set_num_threads(1)      # one thread is faster for these tiny networks (part B gives the others back)
DATA_1D = torch.tensor(wb.synth.gaussian_1d(n=10_000, mu=4.0, sigma=1.25, seed=0), dtype=torch.float32).reshape(-1, 1)
Z_FIXED_1D = sample_noise(2000, 4, torch.Generator().manual_seed(1))   # a fixed noise to look at a 1-D generator
STEPS_1D = math.ceil(len(DATA_1D) / 64)                                # mini-batches of 64: steps per epoch


def stats_1d(G):
    """The mean and the standard deviation of the numbers G makes from Z_FIXED_1D."""
    x = generate(G, Z_FIXED_1D)[:, 0]
    return float(x.mean()), float(x.std())


print(f"DATA_1D {tuple(DATA_1D.shape)} · moyenne {fr(DATA_1D.mean().item(), 3)} · écart-type {fr(DATA_1D.std().item(), 3)}")'''

TODO_11 = r'''# Nothing to write here: wasserstein_1d goes in mon_travail/mylearn/gen_metrics.py'''

CHECK_11 = r'''with checked("27.11"):
    M = mylearn.gen_metrics
    sample_11 = np.random.default_rng(11).normal(0.0, 1.0, 50)
    same("27.11", "wasserstein_1d([3, 1, 2], [2, 6, 4])", M.wasserstein_1d([3, 1, 2], [2, 6, 4]), 2.0)
    same("27.11", "deux échantillons égaux à l'ordre près", M.wasserstein_1d([5.0, -1.0, 2.5], [2.5, 5.0, -1.0]), 0.0)
    same("27.11", "tous les points décalés de 0,5", M.wasserstein_1d(sample_11, sample_11 + 0.5), 0.5)
    same("27.11", "tailles différentes : [0, 2] et [1]", M.wasserstein_1d([0, 2], [1]), 1.0)
    refuses("27.11", "un échantillon vide", lambda: M.wasserstein_1d([], [1.0]))
    refuses("27.11", "un échantillon 2-D", lambda: M.wasserstein_1d([[1.0, 2.0]], [1.0, 2.0]))'''

EXPERIMENT_11 = r'''with checked("27.11"):
    # What W1 sees: the data against four samples of 2 000 numbers
    rng_11 = np.random.default_rng(2711)
    data_11 = DATA_1D[:2000, 0].numpy()
    for label_11, other_11 in [("N(4 ; 1,25²), la même loi", rng_11.normal(4.0, 1.25, 2000)),
                               ("N(5 ; 1,25²), décalée de 1", rng_11.normal(5.0, 1.25, 2000)),
                               ("N(4 ; 2,5²), deux fois plus étalée", rng_11.normal(4.0, 2.5, 2000)),
                               ("2 000 fois le nombre 4 (effondrement)", np.full(2000, 4.0))]:
        print(f"{label_11:40s} W1 = {fr(mylearn.gen_metrics.wasserstein_1d(data_11, other_11), 3)}")'''

TODO_12 = r'''def gan_step_12(G, D, opt_g, opt_d, real, z):
    """One iteration of the standard GAN loop on a mini-batch (D returns LOGITS).
    1) step of D: the reals labelled 1, the fakes G(z) labelled 0, the fakes detached from G;
    2) step of G: the same fakes through the updated D, labelled 1 (non-saturating loss); only opt_g steps.
    Returns a dict of Python floats: "loss_d" (the real term + the fake term, two BCE means, as computed for the step
    of D), "loss_g", "d_real" and "d_fake" (the mean probabilities sigmoid(D(.)) of the reals and of the fakes, seen
    by D BEFORE its step)."""
    raise NotImplementedError("gan_step_12")


def train_gan_12(G, D, data, n_epochs, batch_size=64, lr_g=1e-3, lr_d=1e-3, noise="normal", seed=0, step=None,
                 on_epoch=None):
    """Train G and D in place on `data` (a tensor (N, ...)), with two Adam optimizers created here (learning rates
    lr_g and lr_d, betas (0.5, 0.999)). gen = torch.Generator().manual_seed(seed); at each epoch,
    perm = torch.randperm(N, generator=gen), then the mini-batches real = data[perm[i:i + batch_size]] in order (the
    last one may be smaller); for each one, z = sample_noise(len(real), latent_dim(G), gen, noise), then
    step(G, D, opt_g, opt_d, real, z) (step: gan_step_12 when None). After each epoch, on_epoch(epoch, G) if given
    (epochs counted from 1). Returns the history: a dict of lists, one value per step, with the keys of the dicts
    returned by step."""
    raise NotImplementedError("train_gan_12")'''

SOLUTION_12 = r'''def gan_step_12(G, D, opt_g, opt_d, real, z):
    """One iteration of the standard GAN loop on a mini-batch (D returns LOGITS).
    1) step of D: the reals labelled 1, the fakes G(z) labelled 0, the fakes detached from G;
    2) step of G: the same fakes through the updated D, labelled 1 (non-saturating loss); only opt_g steps.
    Returns a dict of Python floats: "loss_d" (the real term + the fake term, two BCE means, as computed for the step
    of D), "loss_g", "d_real" and "d_fake" (the mean probabilities sigmoid(D(.)) of the reals and of the fakes, seen
    by D BEFORE its step)."""
    fake = G(z)
    logits_real, logits_fake = D(real), D(fake.detach())          # detach: the loss of D does not reach G
    loss_d = (F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
              + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake)))
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()
    logits = D(fake)                                              # the updated D; the gradient flows into G
    loss_g = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    opt_g.zero_grad()
    loss_g.backward()                                             # also fills D's .grad: opt_d.zero_grad() clears it
    opt_g.step()                                                  # only G moves
    return {"loss_d": loss_d.item(), "loss_g": loss_g.item(),
            "d_real": torch.sigmoid(logits_real).mean().item(), "d_fake": torch.sigmoid(logits_fake).mean().item()}


def train_gan_12(G, D, data, n_epochs, batch_size=64, lr_g=1e-3, lr_d=1e-3, noise="normal", seed=0, step=None,
                 on_epoch=None):
    """Train G and D in place on `data` (a tensor (N, ...)), with two Adam optimizers created here (learning rates
    lr_g and lr_d, betas (0.5, 0.999)). gen = torch.Generator().manual_seed(seed); at each epoch,
    perm = torch.randperm(N, generator=gen), then the mini-batches real = data[perm[i:i + batch_size]] in order (the
    last one may be smaller); for each one, z = sample_noise(len(real), latent_dim(G), gen, noise), then
    step(G, D, opt_g, opt_d, real, z) (step: gan_step_12 when None). After each epoch, on_epoch(epoch, G) if given
    (epochs counted from 1). Returns the history: a dict of lists, one value per step, with the keys of the dicts
    returned by step."""
    step = gan_step_12 if step is None else step
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(seed)
    nz = latent_dim(G)
    history = {}
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(len(data), generator=gen)
        for i in range(0, len(data), batch_size):
            real = data[perm[i:i + batch_size]]
            z = sample_noise(len(real), nz, gen, noise)
            for key, value in step(G, D, opt_g, opt_d, real, z).items():
                history.setdefault(key, []).append(value)
        if on_epoch is not None:
            on_epoch(epoch, G)
    return history'''

CHECK_12 = r'''with checked("27.12"):
    # One step on a tiny GAN, compared with what the step must do
    G_c, D_c = make_toy_gan(dim=1, seed=12)
    G_0, D_0 = copy.deepcopy(G_c), copy.deepcopy(D_c)
    opt_g_c = torch.optim.Adam(G_c.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d_c = torch.optim.Adam(D_c.parameters(), lr=1e-3, betas=(0.5, 0.999))
    real_c, z_c = DATA_1D[:64], sample_noise(64, 4, torch.Generator().manual_seed(12))
    out_c = gan_step_12(G_c, D_c, opt_g_c, opt_d_c, real_c, z_c)
    if returned("27.12", "gan_step_12", out_c):
        missing_c = sorted({"loss_d", "loss_g", "d_real", "d_fake"} - set(out_c))
        if missing_c:
            print(f"❌ Ex 27.12 : il manque les clés {missing_c} dans le dictionnaire rendu.")
            raise StopCheck
        with torch.no_grad():
            lr0_c, fake0_c = D_0(real_c), G_0(z_c)
            lf0_c = D_0(fake0_c)
            same("27.12", "loss_d (calculée avant le pas de D)", out_c["loss_d"],
                 (F.binary_cross_entropy_with_logits(lr0_c, torch.ones_like(lr0_c))
                  + F.binary_cross_entropy_with_logits(lf0_c, torch.zeros_like(lf0_c))).item(), rtol=1e-5)
            same("27.12", "d_real, la moyenne de sigmoid(D(x)) avant le pas", out_c["d_real"], torch.sigmoid(lr0_c).mean().item(), rtol=1e-5)
            same("27.12", "d_fake, la moyenne de sigmoid(D(G(z))) avant le pas", out_c["d_fake"], torch.sigmoid(lf0_c).mean().item(), rtol=1e-5)
            after_d_c = D_c(fake0_c)                                  # D after its step (G's step must not move D)
            same("27.12", "loss_g (les faux d'avant le pas de G, jugés par D après son pas, label 1)", out_c["loss_g"],
                 F.binary_cross_entropy_with_logits(after_d_c, torch.ones_like(after_d_c)).item(), rtol=1e-5)
            lr1_c = D_c(real_c)
            d_learns_c = bool((F.binary_cross_entropy_with_logits(lr1_c, torch.ones_like(lr1_c))
                               + F.binary_cross_entropy_with_logits(after_d_c, torch.zeros_like(after_d_c))).item()
                              < out_c["loss_d"])
            g_learns_c = bool(D_c(G_c(z_c)).mean() > after_d_c.mean())
        verdict("27.12", d_learns_c, "après le pas de D, sa loss sur le même mini-batch a baissé : il distingue mieux vrais et faux.",
                "après ton pas, D ne distingue pas mieux les vrais des faux : vérifie les labels (1 pour les vrais, 0 "
                "pour les faux) et l'ordre zero_grad, backward, step.")
        verdict("27.12", g_learns_c, "le pas de G fait monter D(G(z)) : les nouveaux faux trompent mieux D.",
                "après ton pas, les faux trompent moins D : le générateur vise le label 1, et seul opt_g avance.")'''

EXPERIMENT_12 = r'''with checked("27.12"):
    # Train the toy GAN for 10 epochs on DATA_1D (about 10 s), then look at what it makes
    STATS_12 = {0: (None, None)}

    def record_12(epoch, G):
        STATS_12[epoch] = stats_1d(G)

    G_12, D_12 = make_toy_gan(dim=1, seed=0)
    STATS_12[0] = stats_1d(G_12)
    start_12 = time.perf_counter()
    history_12 = train_gan_12(G_12, D_12, DATA_1D, 10, batch_size=64, on_epoch=record_12)
    mean_12, std_12 = STATS_12[10]
    print(f"10 epochs en {time.perf_counter() - start_12:.0f} s · générés : moyenne {fr(mean_12, 2)}, écart-type "
          f"{fr(std_12, 2)} (cible 4 et 1,25)")
    if written(lambda: mylearn.gen_metrics.wasserstein_1d([0.0], [1.0])):
        print(f"W1 entre 2 000 nombres générés et 2 000 données : "
              f"{fr(mylearn.gen_metrics.wasserstein_1d(generate(G_12, Z_FIXED_1D)[:, 0], DATA_1D[:2000, 0].numpy()), 3)}")
    fig_12, (left_12, right_12) = plt.subplots(1, 2, figsize=(10, 3.2))
    left_12.hist(DATA_1D[:, 0].numpy(), bins=60, density=True, alpha=0.5, label="données", color=wb.plot.CATEGORICAL[3])
    left_12.hist(generate(G_12, Z_FIXED_1D)[:, 0], bins=60, density=True, alpha=0.5, label="générés", color=wb.plot.CATEGORICAL[0])
    left_12.legend()
    for key_12 in ("loss_d", "loss_g", "d_real", "d_fake"):
        right_12.plot(np.arange(1, 11), mean_epochs(history_12, key_12, STEPS_1D), marker="o", label=key_12)
    right_12.axhline(2 * LN2, color="gray", ls=":", lw=1)
    right_12.axhline(LN2, color="gray", ls=":", lw=1)
    right_12.axhline(0.5, color="gray", ls="--", lw=1)
    right_12.set_xlabel("epoch")
    right_12.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
    ok_12 = (len(history_12.get("loss_d", [])) == 10 * STEPS_1D and abs(mean_12 - 4) < 0.5 and 0.6 < std_12 < 2.2)
    verdict("27.12", ok_12, "le générateur imite la gaussienne (moyenne à moins de 0,5 de 4, écart-type entre 0,6 et 2,2).",
            f"après 10 epochs : {len(history_12.get('loss_d', []))} pas enregistrés (attendu {10 * STEPS_1D}), moyenne "
            f"{fr(mean_12, 2)}, écart-type {fr(std_12, 2)} : relis l'ordre des étapes de train_gan_12.")'''


def _runs13() -> dict:
    return HISTORY.get("runs13") or {"runs": []}


GIVEN_13 = r'''# The trainings of 27.12 on five seeds (tools/chapters/history_ch27.py, the solution of 27.12): for each seed, after
# each epoch (0 = before training), the mean and the standard deviation of 2 000 numbers made from Z_FIXED_1D, and the
# epoch means of loss_d, loss_g, d_real and d_fake
RUNS_13 = unpack("{RUNS13}")   # packed, so that the numbers stay hidden until the experiment cell'''

TODO_13 = r'''overshoot_13 = ...   # a) the generated mean after epoch 1: "A" below 3.5 (still on its way), "B" between 3.5 and
                     #    4.5, "C" above 4.5 (beyond the target)
steady_13 = ...      # b) True or False: from epoch 2 to epoch 10, the standard deviation stays between 1.0 and 1.5 at
                     #    every epoch, for the five seeds
loss_d_13 = ...      # c) the mean loss_d of the last epoch: "A" about 0.5, "B" about 0.69, "C" about 1.39'''

SOLUTION_13 = r'''overshoot_13, steady_13, loss_d_13 = "C", False, "C"'''

EXPERIMENT_13 = r'''print(f"{'graine':>6s} {'moyenne après l’epoch 1':>24s} {'écart-type, epochs 2 à 10':>28s} {'loss_d, epoch 10':>17s}")
for run_13 in RUNS_13["runs"]:
    print(f"{run_13['seed']:6d} {fr(run_13['mean'][1], 2):>24s} {fr(min(run_13['std'][2:]), 2) + ' à ' + fr(max(run_13['std'][2:]), 2):>28s} "
          f"{fr(run_13['loss_d'][-1], 3):>17s}")
fig_13, axes_13 = plt.subplots(1, 2, figsize=(10, 3))
for run_13 in RUNS_13["runs"]:
    axes_13[0].plot(run_13["mean"], marker=".", label=f"graine {run_13['seed']}")
    axes_13[1].plot(run_13["std"], marker=".")
axes_13[0].axhline(4.0, color="gray", ls="--")
axes_13[1].axhline(1.25, color="gray", ls="--")
axes_13[0].set_title("moyenne générée (cible 4)", fontsize=10)
axes_13[1].set_title("écart-type généré (cible 1,25)", fontsize=10)
for ax_13 in axes_13:
    ax_13.set_xlabel("epoch")
axes_13[0].legend(fontsize=7)
plt.tight_layout()
plt.show()
wb.check("27.13a", overshoot_13)
wb.check("27.13b", steady_13)
wb.check("27.13c", loss_d_13)'''

RECORD_13 = r'''wb.record("27.13a", "C", mistakes={"regarde la colonne « moyenne après l'epoch 1 » : le générateur est-il encore en chemin ?": "A",
                                   "la moyenne après l'epoch 1 tombe-t-elle entre 3,5 et 4,5 ?": "B"})
wb.record("27.13b", False, mistakes={"regarde la colonne des écarts-types : restent-ils entre 1,0 et 1,5 ?": True})
wb.record("27.13c", "C", mistakes={"la loss de D additionne deux termes de cross-entropy, chacun ln 2 à l'équilibre": "B",
                                   "0,5 est la sortie de D à l'équilibre, pas sa loss": "A"})'''

GIVEN_15 = r'''# The book's experiment (§27.5): 10 000 points of a 2-D gaussian centred at (5, 5), standard deviation 1
DATA_2D = torch.tensor(np.random.default_rng(2715).normal(5.0, 1.0, size=(10_000, 2)), dtype=torch.float32)
EPOCHS_15 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13)                    # the panels of fig. 27.20
Z_FIXED_15 = sample_noise(500, 4, torch.Generator().manual_seed(15), "uniform")


def make_book_gan_15(seed=0):
    """The networks of the book (§27.5): G: 4 -> 16 -> 2 (LeakyReLU of slope 0.1, nothing at the end); D: 2 -> 16 ->
    16 -> 1 (LeakyReLU 0.1, a logit at the end: BCEWithLogitsLoss puts the sigmoid in the loss)."""
    torch.manual_seed(seed)
    return mlp([4, 16, 2], slope=0.1), mlp([2, 16, 16, 1], slope=0.1)'''

TODO_15 = r'''def train_book_15(seed=0, n_epochs=13):
    """Train make_book_gan_15(seed) on DATA_2D with your train_gan_12 (mini-batches of 32, uniform noise, learning
    rates 1e-3, training seed `seed`). Returns (snapshots, history): snapshots[e] = generate(G, Z_FIXED_15) after
    epoch e, for each e of EPOCHS_15 not above n_epochs (e = 0: before training)."""
    raise NotImplementedError("train_book_15")


def plot_clouds_15(snapshots, data):
    """One panel per epoch of `snapshots` (in increasing order, 4 per row): 500 points of `data` in one color and the
    generated points in another, each with its ellipse (add_ellipse), the same axes [0, 14] x [0, 14] everywhere, the
    epoch in the title. Returns an array (number of panels, 4): for each panel, the mean (x, y) and the standard
    deviation (x, y) of the generated points."""
    raise NotImplementedError("plot_clouds_15")


def plot_losses_15(history, steps_per_epoch):
    """The mean loss of D and of G over each epoch (mean_epochs), with horizontal lines at ln 2 and 2 ln 2.
    Returns the two arrays of epoch means (loss_d, loss_g)."""
    raise NotImplementedError("plot_losses_15")'''

SOLUTION_15 = r'''def train_book_15(seed=0, n_epochs=13):
    """Train make_book_gan_15(seed) on DATA_2D with your train_gan_12 (mini-batches of 32, uniform noise, learning
    rates 1e-3, training seed `seed`). Returns (snapshots, history): snapshots[e] = generate(G, Z_FIXED_15) after
    epoch e, for each e of EPOCHS_15 not above n_epochs (e = 0: before training)."""
    G, D = make_book_gan_15(seed)
    snapshots = {0: generate(G, Z_FIXED_15)}

    def keep(epoch, G):
        if epoch in EPOCHS_15:
            snapshots[epoch] = generate(G, Z_FIXED_15)

    history = train_gan_12(G, D, DATA_2D, n_epochs, batch_size=32, lr_g=1e-3, lr_d=1e-3, noise="uniform", seed=seed,
                           on_epoch=keep)
    return snapshots, history


def plot_clouds_15(snapshots, data):
    """One panel per epoch of `snapshots` (in increasing order, 4 per row): 500 points of `data` in one color and the
    generated points in another, each with its ellipse (add_ellipse), the same axes [0, 14] x [0, 14] everywhere, the
    epoch in the title. Returns an array (number of panels, 4): for each panel, the mean (x, y) and the standard
    deviation (x, y) of the generated points."""
    epochs = sorted(snapshots)
    real = np.asarray(data[:500])
    n_rows = math.ceil(len(epochs) / 4)
    fig, axes = plt.subplots(n_rows, 4, figsize=(10, 2.6 * n_rows), squeeze=False)
    stats = []
    for ax in axes.flat:
        ax.axis("off")
    for ax, epoch in zip(axes.flat, epochs):
        fake = np.asarray(snapshots[epoch])
        ax.axis("on")
        ax.scatter(*real.T, s=3, alpha=0.4, color=wb.plot.CATEGORICAL[0])
        ax.scatter(*fake.T, s=3, alpha=0.4, color=wb.plot.CATEGORICAL[1])
        add_ellipse(ax, real, wb.plot.CATEGORICAL[0])
        add_ellipse(ax, fake, wb.plot.CATEGORICAL[1])
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 14)
        ax.set_aspect("equal")
        ax.set_title(f"epoch {epoch}", fontsize=9)
        stats.append([*fake.mean(axis=0), *fake.std(axis=0)])
    plt.tight_layout()
    plt.show()
    return np.array(stats)


def plot_losses_15(history, steps_per_epoch):
    """The mean loss of D and of G over each epoch (mean_epochs), with horizontal lines at ln 2 and 2 ln 2.
    Returns the two arrays of epoch means (loss_d, loss_g)."""
    loss_d = mean_epochs(history, "loss_d", steps_per_epoch)
    loss_g = mean_epochs(history, "loss_g", steps_per_epoch)
    epochs = np.arange(1, len(loss_d) + 1)
    plt.figure(figsize=(6, 3))
    plt.plot(epochs, loss_d, marker="o", label="loss de D (vrais + faux)")
    plt.plot(epochs, loss_g, marker="o", label="loss de G")
    plt.axhline(2 * LN2, color="gray", ls=":", label="2 ln 2")
    plt.axhline(LN2, color="gray", ls="--", label="ln 2")
    plt.xlabel("epoch")
    plt.legend(fontsize=8)
    plt.show()
    return loss_d, loss_g'''

CHECK_15 = r'''with checked("27.15"):
    start_15 = time.perf_counter()
    snapshots_15, history_15 = train_book_15(seed=0, n_epochs=13)
    print(f"13 epochs en {time.perf_counter() - start_15:.0f} s")
    if returned("27.15", "train_book_15", snapshots_15):
        if sorted(snapshots_15) != list(EPOCHS_15):
            print(f"❌ Ex 27.15 : snapshots a les epochs {sorted(snapshots_15)}, attendu {list(EPOCHS_15)}.")
            raise StopCheck
        stats_15 = plot_clouds_15(snapshots_15, DATA_2D.numpy())
        if returned("27.15", "plot_clouds_15", stats_15):
            want_15 = np.array([[*np.asarray(snapshots_15[e]).mean(0), *np.asarray(snapshots_15[e]).std(0)] for e in EPOCHS_15])
            same("27.15", "les moyennes et les écarts-types de chaque panneau", stats_15, want_15, rtol=1e-6)
        curves_15 = plot_losses_15(history_15, math.ceil(len(DATA_2D) / 32))
        if returned("27.15", "plot_losses_15", curves_15):
            same("27.15", "les losses moyennes de D par epoch", curves_15[0], mean_epochs(history_15, "loss_d", 313), rtol=1e-6)
        final_15 = np.asarray(snapshots_15[13])
        ok_15 = np.all(np.abs(final_15.mean(0) - 5) < 0.6) and np.all((final_15.std(0) > 0.6) & (final_15.std(0) < 1.6))
        verdict("27.15", ok_15, f"à l'epoch 13, le nuage généré a rejoint la cible : centre ({fr(final_15.mean(0)[0], 2)} ; "
                f"{fr(final_15.mean(0)[1], 2)}), écarts-types {fr(final_15.std(0)[0], 2)} et {fr(final_15.std(0)[1], 2)}.",
                f"à l'epoch 13, centre ({fr(final_15.mean(0)[0], 2)} ; {fr(final_15.mean(0)[1], 2)}) et écarts-types "
                f"{fr(final_15.std(0)[0], 2)} et {fr(final_15.std(0)[1], 2)} : loin de (5 ; 5) et de 1. Vérifie le bruit "
                "uniforme et les mini-batchs de 32.")'''

BUGGY_16 = r'''def train_buggy_16(G, D, data, n_epochs, batch_size=64, lr=1e-3, seed=0):
    """A GAN training loop (D returns logits): the code to debug."""
    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))                    # (A)
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))                    # (B)
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(n_epochs):
        perm = torch.randperm(len(data), generator=gen)                                     # (C)
        for i in range(0, len(data), batch_size):
            real = data[perm[i:i + batch_size]]                                             # (D)
            z = torch.randn(len(real), latent_dim(G), generator=gen)                        # (E)
            opt_g.zero_grad()                                                               # (F)
            fake = G(z)                                                                     # (G)
            logits_real, logits_fake = D(real), D(fake)                                     # (H)
            loss_d = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake))  # (I)
            loss_d.backward()                                                               # (J)
            opt_d.step()                                                                    # (K)
            logits = D(G(z))                                                                # (L)
            loss_g = bce(logits, torch.zeros_like(logits))                                  # (M)
            loss_g.backward()                                                               # (N)
            opt_g.step()                                                                    # (O)'''

GIVEN_16 = r'''# The code to debug (do not change this cell: copy the function into the next one to fix it)
''' + BUGGY_16 + r'''


class SpyAdam(torch.optim.Adam):
    """Adam that notices gradients never reset between two of its steps: at a step, a parameter whose .grad is the
    very tensor it had at the previous step has not been reset (zero_grad() sets the gradients to None)."""
    created = []

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.kept, self._last = 0, {}
        SpyAdam.created.append(self)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and self._last.get(id(p)) is p.grad:
                    self.kept += 1
                self._last[id(p)] = p.grad
        return super().step(closure)


class SpyGenerator(nn.Module):
    """A generator that counts the backward passes that reach its output."""

    def __init__(self, G):
        super().__init__()
        self.G, self.backwards = G, 0

    def forward(self, z):
        out = self.G(z)
        if out.requires_grad:
            out.register_hook(self._count)
        return out

    def _count(self, grad):
        self.backwards += 1


def diagnose_16(train_fn):
    """The three diagnostics of 27.16, in order: the step of D must not reach G, the gradients of D must be reset at
    each step, the generator must move towards the data. Prints ✅ or ❌ for each one, stops at the first ❌."""
    G, D = make_toy_gan(dim=1, seed=16)
    spy_g, adam = SpyGenerator(G), torch.optim.Adam
    SpyAdam.created = []
    torch.optim.Adam = SpyAdam                       # the optimizers created by train_fn are spied on
    try:
        train_fn(spy_g, D, DATA_1D[:640], 1, batch_size=64, seed=16)        # 10 iterations
    finally:
        torch.optim.Adam = adam
    d_ids = {id(p) for p in D.parameters()}
    kept_d = sum(opt.kept for opt in SpyAdam.created
                 if any(id(p) in d_ids for group in opt.param_groups for p in group["params"]))
    G2, D2 = make_toy_gan(dim=1, seed=16)
    train_fn(G2, D2, DATA_1D, 3, batch_size=64, seed=16)
    mean2, std2 = stats_1d(G2)
    points = [
        ("le pas de D ne remonte pas dans le générateur", spy_g.backwards <= 10,
         f"en 10 itérations, {spy_g.backwards} passes arrière ont atteint la sortie du générateur, au lieu de 10 : la "
         "loss de D remonte jusqu'à G."),
        ("les gradients de D sont remis à zéro avant chacun de ses pas", kept_d == 0,
         "au moment de ses pas, l'optimiseur de D trouve les gradients du pas précédent : ils s'accumulent."),
        ("le générateur rejoint les données", abs(mean2 - 4) < 1.0,
         f"après 3 epochs, les nombres générés ont une moyenne de {fr(mean2, 2)} (cible 4) : le générateur ne va pas "
         "vers les données."),
    ]
    for k, (name, ok, why) in enumerate(points, start=1):
        if ok:
            print(f"✅ Ex 27.16 : point {k}, {name}.")
        else:
            print(f"❌ Ex 27.16 : point {k} : {why}")
            for later in range(k + 1, len(points) + 1):
                print(f"⏳ Ex 27.16 : point {later} : corrige d'abord le point {k}.")
            return False
    print(f"   après 3 epochs : moyenne {fr(mean2, 2)}, écart-type {fr(std2, 2)} (cible 4 et 1,25)")
    return True


print("Le code fautif, essayé par les diagnostics :")
_ = diagnose_16(train_buggy_16)'''


TODO_16 = r'''faulty_16 = ...   # the letters of the faulty lines, e.g. "AB"


def train_fixed_16(G, D, data, n_epochs, batch_size=64, lr=1e-3, seed=0):
    """The fixed version of train_buggy_16: same arguments, same draws from gen (one randperm per epoch, then one
    torch.randn per mini-batch)."""
    raise NotImplementedError("train_fixed_16")'''

SOLUTION_16 = r'''faulty_16 = "FHM"


def train_fixed_16(G, D, data, n_epochs, batch_size=64, lr=1e-3, seed=0):
    """The fixed version of train_buggy_16: same arguments, same draws from gen (one randperm per epoch, then one
    torch.randn per mini-batch)."""
    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(n_epochs):
        perm = torch.randperm(len(data), generator=gen)
        for i in range(0, len(data), batch_size):
            real = data[perm[i:i + batch_size]]
            z = torch.randn(len(real), latent_dim(G), generator=gen)
            opt_d.zero_grad()                                       # (F) reset D's gradients too
            opt_g.zero_grad()
            fake = G(z)
            logits_real, logits_fake = D(real), D(fake.detach())    # (H) the loss of D must not reach G
            loss_d = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake))
            loss_d.backward()
            opt_d.step()
            logits = D(G(z))
            loss_g = bce(logits, torch.ones_like(logits))           # (M) G wants its fakes taken for reals: label 1
            loss_g.backward()
            opt_g.step()'''

CHECK_16 = r'''wb.check("27.16", faulty_16)
with checked("27.16"):
    if written(lambda: train_fixed_16(*make_toy_gan(dim=1, seed=0), DATA_1D[:8], 0)):
        if diagnose_16(train_fixed_16) and written(lambda: train_gan_12(*make_toy_gan(dim=1, seed=0), DATA_1D[:8], 0)):
            G_a, D_a = make_toy_gan(dim=1, seed=161)
            G_b, D_b = copy.deepcopy(G_a), copy.deepcopy(D_a)
            train_fixed_16(G_a, D_a, DATA_1D[:1280], 1, seed=161)
            train_gan_12(G_b, D_b, DATA_1D[:1280], 1, seed=161)
            twins_16 = all(torch.allclose(p, q, atol=1e-6) for p, q in zip(G_a.parameters(), G_b.parameters()))
            print(("   ta version corrigée donne exactement le générateur de ton train_gan_12 (même graine)." if twins_16 else
                   "   ta version corrigée et ton train_gan_12 donnent des générateurs différents pour la même graine : "
                   "l'un des deux fait autre chose (ordre des tirages, labels, learning rates)."))
    else:
        print("⏳ Ex 27.16 : les diagnostics attendent ta fonction corrigée.")'''

RECORD_16 = ('wb.record("27.16", "FHM", mistakes=' + bug_letters_mistakes("FHM", "ABCDEGIJKLNO", {
    "F": "un des diagnostics regarde si les gradients de D sont remis à zéro avant chacun de ses pas",
    "H": "un des diagnostics compte les passes arrière qui atteignent le générateur",
    "M": "le générateur doit-il viser le label des vrais ou celui des faux ?"}) + ")")

TODO_17 = r'''d_fake_17 = ...   # a) with the strong D, the mean D(G(z)) over the FIRST epoch: "A" below 0.1, "B" from 0.1 to 0.4,
                  #    "C" above 0.4
reach_17 = ...    # b) True or False: after 10 epochs, the generated mean is within 0.3 of 4
steady_17 = ...   # c) True or False: from epoch 6 to epoch 10, the generated standard deviation varies LESS (max - min)
                  #    than in the training of 27.12'''

SOLUTION_17 = r'''d_fake_17, reach_17, steady_17 = "B", True, True'''

EXPERIMENT_17 = r'''with checked("27.17"):
    def record_17(store):
        return lambda epoch, G: store.__setitem__(epoch, stats_1d(G))

    if "STATS_12" not in globals() or 10 not in STATS_12:              # the training of 27.12, if it has not run
        STATS_12 = {}
        G_b17, D_b17 = make_toy_gan(dim=1, seed=0)
        train_gan_12(G_b17, D_b17, DATA_1D, 10, on_epoch=record_17(STATS_12))
    STATS_17 = {}
    G_17, D_17 = make_toy_gan(dim=1, seed=0)
    history_17 = train_gan_12(G_17, D_17, DATA_1D, 10, lr_d=1e-2, on_epoch=record_17(STATS_17))
    print(f"{'epoch':>5s} {'27.12 : moyenne / écart-type':>30s} {'D fort : moyenne / écart-type':>31s}")
    for e_17 in range(1, 11):
        print(f"{e_17:5d} {fr(STATS_12[e_17][0], 2) + ' / ' + fr(STATS_12[e_17][1], 2):>30s} "
              f"{fr(STATS_17[e_17][0], 2) + ' / ' + fr(STATS_17[e_17][1], 2):>31s}")
    d_fake_epochs_17 = mean_epochs(history_17, "d_fake", STEPS_1D)
    print(f"D fort : D(G(z)) moyen à l'epoch 1 = {fr(d_fake_epochs_17[0], 3)}, à l'epoch 10 = {fr(d_fake_epochs_17[-1], 3)}")
    spread_17 = [max(s[e][1] for e in range(6, 11)) - min(s[e][1] for e in range(6, 11)) for s in (STATS_12, STATS_17)]
    print(f"variation de l'écart-type, epochs 6 à 10 : {fr(spread_17[0], 2)} (27.12) contre {fr(spread_17[1], 2)} (D fort)")
wb.check("27.17a", d_fake_17)
wb.check("27.17b", reach_17)
wb.check("27.17c", steady_17)'''

RECORD_17 = r'''wb.record("27.17a", "B", mistakes={"regarde D(G(z)) moyen à l'epoch 1 : le discriminateur fort rejette-t-il presque tous les faux ?": "A",
                                   "regarde D(G(z)) moyen à l'epoch 1 : est-il au-dessus de 0,4 ?": "C"})
wb.record("27.17b", True, mistakes={"regarde la moyenne de la colonne « D fort » à l'epoch 10": False})
wb.record("27.17c", True, mistakes={"compare les deux variations de l'écart-type affichées sous le tableau": False})'''

TODO_18 = r'''def saturating_step_18(G, D, opt_g, opt_d, real, z):
    """Like gan_step_12, but the generator minimizes the minimax loss: the mean of ln(1 - sigmoid(D(G(z)))) through
    the updated D. Returns the same dict ("loss_g": that mean, a negative number)."""
    raise NotImplementedError("saturating_step_18")


def generator_grad_norm_18(G, D, z, saturating):
    """The norm of the gradient of the generator's loss (saturating=True: mean ln(1 - sigmoid(D(G(z)))); False: the
    non-saturating mean -ln sigmoid(D(G(z)))) with respect to ALL the parameters of G, put end to end in one vector.
    No optimizer step; the .grad of G and D are left at None. Returns a Python float."""
    raise NotImplementedError("generator_grad_norm_18")'''

SOLUTION_18 = r'''def saturating_step_18(G, D, opt_g, opt_d, real, z):
    """Like gan_step_12, but the generator minimizes the minimax loss: the mean of ln(1 - sigmoid(D(G(z)))) through
    the updated D. Returns the same dict ("loss_g": that mean, a negative number)."""
    fake = G(z)
    logits_real, logits_fake = D(real), D(fake.detach())
    loss_d = (F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
              + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake)))
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()
    logits = D(fake)
    loss_g = -F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))   # mean ln(1 - sigmoid(a))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return {"loss_d": loss_d.item(), "loss_g": loss_g.item(),
            "d_real": torch.sigmoid(logits_real).mean().item(), "d_fake": torch.sigmoid(logits_fake).mean().item()}


def generator_grad_norm_18(G, D, z, saturating):
    """The norm of the gradient of the generator's loss (saturating=True: mean ln(1 - sigmoid(D(G(z)))); False: the
    non-saturating mean -ln sigmoid(D(G(z)))) with respect to ALL the parameters of G, put end to end in one vector.
    No optimizer step; the .grad of G and D are left at None. Returns a Python float."""
    logits = D(G(z))
    if saturating:
        loss = -F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    else:
        loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    grads = torch.autograd.grad(loss, list(G.parameters()))         # fills no .grad at all
    return torch.sqrt(sum((g ** 2).sum() for g in grads)).item()'''

CHECK_18 = r'''with checked("27.18"):
    G_c, D_c = make_toy_gan(dim=1, seed=18)
    G_0, D_0 = copy.deepcopy(G_c), copy.deepcopy(D_c)
    opt_g_c = torch.optim.Adam(G_c.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d_c = torch.optim.Adam(D_c.parameters(), lr=1e-3, betas=(0.5, 0.999))
    real_c, z_c = DATA_1D[:64], sample_noise(64, 4, torch.Generator().manual_seed(18))
    out_c = saturating_step_18(G_c, D_c, opt_g_c, opt_d_c, real_c, z_c)
    if returned("27.18", "saturating_step_18", out_c):
        with torch.no_grad():
            after_d_c = D_c(G_0(z_c))
            same("27.18", "loss_g, la moyenne de ln(1 − sigmoid(D(G(z)))) après le pas de D", out_c["loss_g"],
                 torch.log1p(-torch.sigmoid(after_d_c)).mean().item(), rtol=1e-5)
            verdict("27.18", bool(D_c(G_c(z_c)).mean() > after_d_c.mean()), "le pas saturant fait lui aussi monter D(G(z)).",
                    "après ton pas, les faux trompent moins D : le générateur minimise ln(1 − D(G(z))).")
    for saturating_c in (True, False):
        G_n, D_n = make_toy_gan(dim=1, seed=181)
        logits_n = D_n(G_n(z_c))
        loss_n = (torch.log1p(-torch.sigmoid(logits_n)).mean() if saturating_c else -F.logsigmoid(logits_n).mean())
        want_n = torch.sqrt(sum((g ** 2).sum() for g in torch.autograd.grad(loss_n, list(G_n.parameters())))).item()
        got_n = generator_grad_norm_18(G_n, D_n, z_c, saturating_c)
        same("27.18", f"generator_grad_norm_18(..., saturating={saturating_c})", got_n, want_n, rtol=1e-5)
        verdict("27.18", all(p.grad is None for p in [*G_n.parameters(), *D_n.parameters()]),
                "aucun .grad n'est laissé dans G ni dans D.",
                "des .grad restent remplis dans G ou dans D : utilise torch.autograd.grad, ou remets-les à None.")'''

GIVEN_18 = r'''def pretrain_d_18(seed=18, steps=500):
    """A fresh toy GAN (make_toy_gan(dim=1, seed=seed)) whose discriminator is trained ALONE for `steps` steps (Adam
    1e-3, random mini-batches of 64 reals and 64 fakes of the untrained generator): D then easily spots the fakes.
    Returns (G, D)."""
    G, D = make_toy_gan(dim=1, seed=seed)
    opt = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        real = DATA_1D[torch.randint(0, len(DATA_1D), (64,), generator=gen)]
        with torch.no_grad():
            fake = G(sample_noise(64, 4, gen))
        logits_real, logits_fake = D(real), D(fake)
        loss = (F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
                + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return G, D


def run_18(G, D, step, g_optimizer, n_steps=600, seed=18):
    """Train COPIES of G and D for n_steps random mini-batches of 64 of DATA_1D with step(G, D, opt_g, opt_d, real, z);
    the optimizer of G is "adam" (lr 1e-3, betas (0.5, 0.999)) or "sgd" (lr 0.01, no momentum), D's is always Adam 1e-3.
    Returns the mean of the numbers generated from Z_FIXED_1D every 50 steps."""
    G, D = copy.deepcopy(G), copy.deepcopy(D)
    opt_g = (torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999)) if g_optimizer == "adam"
             else torch.optim.SGD(G.parameters(), lr=0.01))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(seed + 100)
    means = []
    for t in range(1, n_steps + 1):
        real = DATA_1D[torch.randint(0, len(DATA_1D), (64,), generator=gen)]
        step(G, D, opt_g, opt_d, real, sample_noise(64, 4, gen))
        if t % 50 == 0:
            means.append(stats_1d(G)[0])
    return np.array(means)'''

EXPERIMENT_18 = r'''with checked("27.18"):
    print(f"{'graine':>6s} {'D(G(z))':>8s} {'‖∇‖ saturante':>14s} {'‖∇‖ non saturante':>18s} {'rapport':>8s}")
    curves_18 = {}
    for seed_18 in (0, 1, 2):
        G_18, D_18 = pretrain_d_18(seed=seed_18)
        z_18 = sample_noise(256, 4, torch.Generator().manual_seed(seed_18))
        with torch.no_grad():
            d_18 = torch.sigmoid(D_18(G_18(z_18))).mean().item()
        sat_18, ns_18 = (generator_grad_norm_18(G_18, D_18, z_18, s) for s in (True, False))
        print(f"{seed_18:6d} {fr(d_18, 4):>8s} {fr(sat_18, 4):>14s} {fr(ns_18, 3):>18s} {ns_18 / sat_18:8.0f}")
        for opt_18 in ("sgd", "adam"):
            for name_18, step_18 in (("saturante", saturating_step_18), ("non saturante", gan_step_12)):
                curves_18.setdefault((opt_18, name_18), []).append(run_18(G_18, D_18, step_18, opt_18, seed=seed_18))
    fig_18, axes_18 = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax_18, opt_18 in zip(axes_18, ("sgd", "adam")):
        for k_18, name_18 in enumerate(("saturante", "non saturante")):
            for curve_18 in curves_18[(opt_18, name_18)]:
                ax_18.plot(np.arange(50, 601, 50), curve_18, color=wb.plot.CATEGORICAL[k_18], alpha=0.8,
                           label=name_18 if curve_18 is curves_18[(opt_18, name_18)][0] else None)
        ax_18.axhline(4.0, color="gray", ls="--")
        ax_18.set_title(f"générateur entraîné avec {'SGD (lr 0,01)' if opt_18 == 'sgd' else 'Adam (lr 0,001)'}", fontsize=10)
        ax_18.set_xlabel("pas")
        ax_18.legend(fontsize=8)
    axes_18[0].set_ylabel("moyenne générée (cible 4)")
    plt.tight_layout()
    plt.show()'''

TODO_19 = r'''def book_round_19(G, D, opt_g, opt_d, real, z_b, z_c, z_d, errors_only=True):
    """One learning round of the book (§27.2.3), each step on a mini-batch, with the D of that moment:
    (a) the reals go to D: D learns (label 1) from those it calls fake (logit < 0);
    (b) G(z_b) goes to D: G learns (label 1, through D, only opt_g steps) from the fakes D calls fake (logit < 0);
    (c) G(z_c), detached, goes to D: D learns (label 0) from the fakes it calls real (logit >= 0);
    (d) as (b), with z_d.
    The loss of a step is the BCE summed over the selected samples and divided by the size of the mini-batch; a step
    that selects no sample does nothing (no zero_grad, no optimizer step). With errors_only=False, each step selects
    all its samples (the plain BCE means). Returns a dict of ints, the number of samples selected by each step:
    {"a": ..., "b": ..., "c": ..., "d": ...}."""
    raise NotImplementedError("book_round_19")'''

SOLUTION_19 = r'''def book_round_19(G, D, opt_g, opt_d, real, z_b, z_c, z_d, errors_only=True):
    """One learning round of the book (§27.2.3), each step on a mini-batch, with the D of that moment:
    (a) the reals go to D: D learns (label 1) from those it calls fake (logit < 0);
    (b) G(z_b) goes to D: G learns (label 1, through D, only opt_g steps) from the fakes D calls fake (logit < 0);
    (c) G(z_c), detached, goes to D: D learns (label 0) from the fakes it calls real (logit >= 0);
    (d) as (b), with z_d.
    The loss of a step is the BCE summed over the selected samples and divided by the size of the mini-batch; a step
    that selects no sample does nothing (no zero_grad, no optimizer step). With errors_only=False, each step selects
    all its samples (the plain BCE means). Returns a dict of ints, the number of samples selected by each step:
    {"a": ..., "b": ..., "c": ..., "d": ...}."""
    def learn(logits, target, selected, optimizer):
        if not errors_only:
            selected = torch.ones_like(selected)
        count = int(selected.sum())
        if count:
            losses = F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target), reduction="none")
            loss = (losses * selected).sum() / len(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return count

    counts = {}
    logits = D(real)
    counts["a"] = learn(logits, 1.0, (logits < 0).float(), opt_d)                 # false negatives: D learns
    for key, z in (("b", z_b), ("c", z_c), ("d", z_d)):
        if key == "c":
            logits = D(G(z).detach())
            counts["c"] = learn(logits, 0.0, (logits >= 0).float(), opt_d)       # false positives: D learns
        else:
            logits = D(G(z))
            counts[key] = learn(logits, 1.0, (logits < 0).float().detach(), opt_g)   # true negatives: G learns
    return counts'''

CHECK_19 = r'''with checked("27.19"):
    real_c = DATA_2D[:32]
    zs_c = [sample_noise(32, 4, torch.Generator().manual_seed(190 + k), "uniform") for k in range(3)]
    for bias_c, want_c, why_c in ((20.0, {"a": 0, "b": 0, "c": 32, "d": 0}, "D répond « vrai » à tout"),
                                  (-20.0, {"a": 32, "b": 32, "c": 0, "d": 32}, "D répond « faux » à tout")):
        G_c, D_c = make_book_gan_15(seed=19)
        with torch.no_grad():
            D_c[-1].bias.fill_(bias_c)
        G_0 = copy.deepcopy(G_c)
        opt_g_c = torch.optim.Adam(G_c.parameters(), lr=1e-3, betas=(0.5, 0.999))
        opt_d_c = torch.optim.Adam(D_c.parameters(), lr=1e-3, betas=(0.5, 0.999))
        counts_c = book_round_19(G_c, D_c, opt_g_c, opt_d_c, real_c, *zs_c)
        if not returned("27.19", "book_round_19", counts_c):
            raise StopCheck
        same("27.19", f"les nombres d'échantillons retenus quand {why_c}", [counts_c.get(k) for k in "abcd"],
             [want_c[k] for k in "abcd"])
        moved_c = any(not torch.equal(p, q) for p, q in zip(G_c.parameters(), G_0.parameters()))
        verdict("27.19", moved_c == (want_c["b"] > 0), f"quand {why_c}, le générateur {'apprend' if want_c['b'] else 'ne bouge pas'}.",
                f"quand {why_c}, le générateur {'devrait apprendre' if want_c['b'] else 'ne devrait pas bouger'}.")
    G_c, D_c = make_book_gan_15(seed=19)
    counts_c = book_round_19(G_c, D_c, torch.optim.Adam(G_c.parameters()), torch.optim.Adam(D_c.parameters()),
                             real_c, *zs_c, errors_only=False)
    same("27.19", "errors_only=False : tous les échantillons à chaque étape", [counts_c.get(k) for k in "abcd"], [32] * 4)'''

GIVEN_19 = r'''def run_19(kind, seed=0, n_epochs=10):
    """Train make_book_gan_15(seed) on DATA_2D (mini-batches of 32, uniform noise, Adam 1e-3 for both) with
    kind = "standard" (your gan_step_12), "book" (your book_round_19, errors only) or "book_all" (book_round_19 with
    errors_only=False). Returns (w1, cloud): after each epoch, the mean over x and y of the W1 distances (your
    wasserstein_1d) between 2 000 data and 2 000 generated points, and the last generated points (Z_FIXED_15)."""
    G, D = make_book_gan_15(seed)
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(seed)
    z_eval = sample_noise(2000, 4, torch.Generator().manual_seed(1), "uniform")
    data_eval = DATA_2D[:2000].numpy()
    w1 = []
    for epoch in range(n_epochs):
        perm = torch.randperm(len(DATA_2D), generator=gen)
        for i in range(0, len(DATA_2D), 32):
            real = DATA_2D[perm[i:i + 32]]
            if kind == "standard":
                gan_step_12(G, D, opt_g, opt_d, real, sample_noise(len(real), 4, gen, "uniform"))
            else:
                zs = [sample_noise(len(real), 4, gen, "uniform") for _ in range(3)]
                book_round_19(G, D, opt_g, opt_d, real, *zs, errors_only=(kind == "book"))
        fake = generate(G, z_eval)
        w1.append(np.mean([mylearn.gen_metrics.wasserstein_1d(fake[:, j], data_eval[:, j]) for j in (0, 1)]))
    return np.array(w1), generate(G, Z_FIXED_15)'''

EXPERIMENT_19 = r'''with checked("27.19"):
    KINDS_19 = {"standard": "boucle standard", "book": "round du livre (erreurs seules)",
                "book_all": "round du livre (tous les échantillons)"}
    results_19 = {}
    for kind_19 in KINDS_19:
        start_19 = time.perf_counter()
        results_19[kind_19] = run_19(kind_19, seed=0)
        print(f"{KINDS_19[kind_19]:40s} {time.perf_counter() - start_19:4.0f} s · W1 final {fr(results_19[kind_19][0][-1], 3)}")
    fig_19, axes_19 = plt.subplots(1, 4, figsize=(13, 3))
    for k_19, (kind_19, (w1_19, cloud_19)) in enumerate(results_19.items()):
        axes_19[0].plot(np.arange(1, 11), w1_19, marker="o", label=KINDS_19[kind_19])
        ax_19 = axes_19[k_19 + 1]
        ax_19.scatter(*DATA_2D[:500].numpy().T, s=3, alpha=0.3, color=wb.plot.CATEGORICAL[0])
        ax_19.scatter(*cloud_19.T, s=3, alpha=0.5, color=wb.plot.CATEGORICAL[1])
        ax_19.set_xlim(-4, 14)
        ax_19.set_ylim(-4, 14)
        ax_19.set_title(KINDS_19[kind_19], fontsize=8)
    axes_19[0].set_yscale("log")
    axes_19[0].set_xlabel("epoch")
    axes_19[0].set_ylabel("W1 moyen (x et y)")
    axes_19[0].legend(fontsize=6)
    plt.tight_layout()
    plt.show()'''

# ---------------------------------------------------------------------------
# Part B: DCGANs on MNIST, evaluation and challenges (27.20 to 27.28)
# ---------------------------------------------------------------------------
GIVEN_B = r'''# The data of part B: MNIST in [-1, 1] (the range of a tanh), N_TRAIN training images (10 000 in fast mode) in a
# fixed order, and 2 000 test images drawn at random; a small CNN classifier gives the features of the Fréchet distance
torch.set_num_threads(N_THREADS)
N_TRAIN = 10_000 if FAST_MODE else 60_000
NZ = 64                                                     # the size of the noise of every generator of part B


def load_images(loader, n_train, seed):
    """Training images (n_train, 1, 28, 28) and test images (4 000) in [-1, 1], with their labels (int64 tensors)."""
    x, y = loader("train", flatten=False, normalize=True, seed=None)
    order = np.random.default_rng(seed).permutation(len(y))[:n_train]
    xt, yt = loader("test", flatten=False, normalize=True, seed=None)
    pick = np.random.default_rng(seed + 1).permutation(len(yt))[:4000]
    to_images = lambda a: torch.tensor(a, dtype=torch.float32).reshape(-1, 1, 28, 28) * 2 - 1
    return to_images(x[order]), torch.tensor(y[order]).long(), to_images(xt[pick]), torch.tensor(yt[pick]).long()


X_MNIST, Y_MNIST, XM_TEST, YM_TEST = load_images(wb.datasets.load_mnist, N_TRAIN, 2700)


class Classifier(nn.Module):
    """A small CNN: features(x) is its second-to-last layer (64 values per image), forward(x) the 10 logits."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU())
        self.head = nn.Linear(64, 10)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


def train_classifier(x, y, n_epochs=3, seed=0):
    """A Classifier trained on (x, y) (Adam 1e-3, mini-batches of 64), returned in eval mode."""
    torch.manual_seed(seed)
    clf = Classifier()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_epochs):
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(clf(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return clf.eval()


def features(clf, images, batch_size=500):
    """The features of the classifier for images in [-1, 1]: a float64 NumPy array (N, 64)."""
    images = torch.as_tensor(np.asarray(as_numpy(images), dtype=np.float32)).reshape(-1, 1, 28, 28)
    with torch.no_grad():
        return torch.cat([clf.features(images[i:i + batch_size]) for i in range(0, len(images), batch_size)]).double().numpy()


def predict(clf, images):
    """The class predicted for each image (an int NumPy array)."""
    images = torch.as_tensor(np.asarray(as_numpy(images), dtype=np.float32)).reshape(-1, 1, 28, 28)
    with torch.no_grad():
        return clf(images).argmax(dim=1).numpy()


def sample_images(G, n, seed=0):
    """n images made by G (eval mode, no graph) from the noise sample_noise(n, latent_dim(G), Generator(seed))."""
    return torch.as_tensor(generate(G, sample_noise(n, latent_dim(G), torch.Generator().manual_seed(seed))),
                           dtype=torch.float32)


def need(ex_id, *names):
    """True when the results of earlier exercises exist (G_20, SNAPSHOTS_20...); otherwise False, with a ⏳ line."""
    missing = [name for name in names if name not in globals()]
    if missing:
        print(f"⏳ Ex {ex_id} : cette cellule attend {', '.join(missing)}, le résultat d'un exercice précédent.")
    return not missing


def digit_report(G, clf, n=1000, seed=0, title=None):
    """Print how many of n generated images the classifier puts in each class, and its mean confidence."""
    images = sample_images(G, n, seed)
    with torch.no_grad():
        probs = torch.softmax(clf(images), dim=1)
    counts = np.bincount(probs.argmax(dim=1).numpy(), minlength=10)
    print(f"{title or 'classes prédites'} : {' '.join(f'{k}:{c}' for k, c in enumerate(counts))} · confiance moyenne "
          f"{pct(probs.max(dim=1).values.mean().item())}")
    return counts


start_b = time.perf_counter()
CLF_MNIST = train_classifier(X_MNIST[:10_000], Y_MNIST[:10_000])
REAL_FEATS = features(CLF_MNIST, XM_TEST[:2000])          # the reference of the Fréchet distances of MNIST
print(f"X_MNIST {tuple(X_MNIST.shape)} · classifieur entraîné en {time.perf_counter() - start_b:.0f} s · accuracy sur "
      f"2 000 images de test {pct(np.mean(predict(CLF_MNIST, XM_TEST[2000:]) == YM_TEST[2000:].numpy()))}")'''

TODO_20 = r'''def make_generator_20(nz=NZ, width=32):
    """A DCGAN generator for 28 x 28 images (Radford's rules): (N, nz) -> (N, 1, 28, 28), in [-1, 1].
    See the contract of the statement (layers, shapes, number of parameters)."""
    raise NotImplementedError("make_generator_20")


def make_discriminator_20(width=32, in_channels=1):
    """A DCGAN discriminator: (N, in_channels, 28, 28) -> (N, 1) LOGITS. See the contract of the statement."""
    raise NotImplementedError("make_discriminator_20")


def init_weights_20(module):
    """For G.apply(init_weights_20): the weights of Linear, Conv2d and ConvTranspose2d ~ N(0, 0.02²) (biases
    untouched), the weights of BatchNorm1d and BatchNorm2d ~ N(1, 0.02²) and their biases at 0 (Radford)."""
    raise NotImplementedError("init_weights_20")'''

SOLUTION_20 = r'''def make_generator_20(nz=NZ, width=32):
    """A DCGAN generator for 28 x 28 images (Radford's rules): (N, nz) -> (N, 1, 28, 28), in [-1, 1].
    See the contract of the statement (layers, shapes, number of parameters)."""
    return nn.Sequential(
        nn.Linear(nz, 2 * width * 7 * 7, bias=False),            # the only dense layer: a projection, reshaped
        nn.BatchNorm1d(2 * width * 7 * 7), nn.ReLU(),
        nn.Unflatten(1, (2 * width, 7, 7)),
        nn.ConvTranspose2d(2 * width, width, 4, stride=2, padding=1, bias=False),   # 7 -> 14
        nn.BatchNorm2d(width), nn.ReLU(),
        nn.ConvTranspose2d(width, 1, 4, stride=2, padding=1),                       # 14 -> 28, no batchnorm
        nn.Tanh())


def make_discriminator_20(width=32, in_channels=1):
    """A DCGAN discriminator: (N, in_channels, 28, 28) -> (N, 1) LOGITS. See the contract of the statement."""
    return nn.Sequential(
        nn.Conv2d(in_channels, width, 4, stride=2, padding=1), nn.LeakyReLU(0.2),  # 28 -> 14, no batchnorm
        nn.Conv2d(width, 2 * width, 4, stride=2, padding=1, bias=False),          # 14 -> 7
        nn.BatchNorm2d(2 * width), nn.LeakyReLU(0.2),
        nn.Flatten(), nn.Linear(2 * width * 7 * 7, 1))                            # one logit


def init_weights_20(module):
    """For G.apply(init_weights_20): the weights of Linear, Conv2d and ConvTranspose2d ~ N(0, 0.02²) (biases
    untouched), the weights of BatchNorm1d and BatchNorm2d ~ N(1, 0.02²) and their biases at 0 (Radford)."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)'''

CHECK_20 = r'''with checked("27.20"):
    G_c, D_c = make_generator_20(), make_discriminator_20()
    if returned("27.20", "make_generator_20", G_c) and returned("27.20", "make_discriminator_20", D_c):
        G_c.eval()
        with torch.no_grad():
            out_c, logits_c = G_c(torch.randn(5, NZ)), D_c(X_MNIST[:5])
        same("27.20", "la forme de G(z)", list(out_c.shape), [5, 1, 28, 28])
        same("27.20", "la forme de D(x)", list(logits_c.shape), [5, 1])
        same("27.20", "le nombre de paramètres de G (nz = 64, width = 32)", n_params(G_c), 240_321)
        same("27.20", "le nombre de paramètres de D (width = 32)", n_params(D_c), 36_577)
        kinds_g = [type(m).__name__ for m in G_c.modules()]
        kinds_d = [type(m).__name__ for m in D_c.modules() if not isinstance(m, nn.Sequential)]
        rules_c = [("ni pooling ni sur-échantillonnage", not any(k in "".join(kinds_g + kinds_d) for k in ("Pool", "Upsample"))),
                   ("des convolutions transposées dans G", "ConvTranspose2d" in kinds_g),
                   ("G finit par une tanh", kinds_g[-1] == "Tanh"),
                   ("des ReLU dans G, pas de Leaky ReLU", "ReLU" in kinds_g and "LeakyReLU" not in kinds_g),
                   ("des Leaky ReLU de pente 0,2 dans D", all(m.negative_slope == 0.2 for m in D_c.modules() if isinstance(m, nn.LeakyReLU))
                    and "LeakyReLU" in kinds_d),
                   ("pas de batchnorm juste après la première couche de D", kinds_d[1] != "BatchNorm2d"),
                   ("D rend un logit, pas une probabilité", kinds_d[-1] == "Linear")]
        for rule_c, ok_c in rules_c:
            verdict("27.20", ok_c, f"règle respectée : {rule_c}.", f"règle non respectée : {rule_c}.")
        G_c.apply(init_weights_20)
        weights_c = torch.cat([m.weight.flatten() for m in G_c.modules() if isinstance(m, (nn.Linear, nn.ConvTranspose2d))])
        bn_c = torch.cat([m.weight.flatten() for m in G_c.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))])
        verdict("27.20", abs(weights_c.std().item() - 0.02) < 0.003 and abs(bn_c.mean().item() - 1) < 0.01,
                "les poids suivent N(0 ; 0,02²) et ceux des batchnorms N(1 ; 0,02²).",
                f"écart-type des poids {fr(weights_c.std().item(), 4)} (attendu 0,02), moyenne des poids des batchnorms "
                f"{fr(bn_c.mean().item(), 3)} (attendu 1).")'''

EXPERIMENT_20 = r'''with checked("27.20"):
    # The DCGAN on MNIST: EPOCHS_20 epochs (3 in fast mode, about 30 s), a copy of G after each epoch
    EPOCHS_20 = 3 if FAST_MODE else 15
    torch.manual_seed(20)
    G_20, D_20 = make_generator_20(), make_discriminator_20()
    G_20.apply(init_weights_20)
    D_20.apply(init_weights_20)
    SNAPSHOTS_20 = {}
    start_20 = time.perf_counter()
    history_20 = train_gan_12(G_20, D_20, X_MNIST, EPOCHS_20, batch_size=64, lr_g=2e-4, lr_d=2e-4, seed=20,
                              on_epoch=lambda epoch, G: SNAPSHOTS_20.__setitem__(epoch, copy.deepcopy(G.state_dict())))
    SECONDS_20 = (time.perf_counter() - start_20) / EPOCHS_20
    print(f"{EPOCHS_20} epochs, {fr(SECONDS_20, 1)} s par epoch")
    G_first_20 = make_generator_20()
    G_first_20.load_state_dict(SNAPSHOTS_20[1])
    show_digits(sample_images(G_first_20, 30, seed=0), title="après 1 epoch")
    show_digits(sample_images(G_20, 30, seed=0), title=f"après {EPOCHS_20} epochs")
    steps_20 = math.ceil(len(X_MNIST) / 64)
    print("losses moyennes par epoch : D " + " · ".join(fr(v, 3) for v in mean_epochs(history_20, "loss_d", steps_20))
          + " | G " + " · ".join(fr(v, 3) for v in mean_epochs(history_20, "loss_g", steps_20)))
    _ = digit_report(G_20, CLF_MNIST, title=f"G_20 après {EPOCHS_20} epochs")'''

TODO_21 = r'''def make_gildenblat_21(nz=NZ, width=16, hidden=256):
    """The book's DCGAN (§27.6, from Gildenblat, 2016), scaled down: returns (G, D), G: (N, nz) -> (N, 1, 28, 28) in
    [-1, 1], D: (N, 1, 28, 28) -> (N, 1) logits. See the contract of the statement."""
    raise NotImplementedError("make_gildenblat_21")'''

SOLUTION_21 = r'''def make_gildenblat_21(nz=NZ, width=16, hidden=256):
    """The book's DCGAN (§27.6, from Gildenblat, 2016), scaled down: returns (G, D), G: (N, nz) -> (N, 1, 28, 28) in
    [-1, 1], D: (N, 1, 28, 28) -> (N, 1) logits. See the contract of the statement."""
    G = nn.Sequential(
        nn.Linear(nz, hidden), nn.Tanh(),
        nn.Linear(hidden, 2 * width * 7 * 7), nn.BatchNorm1d(2 * width * 7 * 7), nn.Tanh(),
        nn.Unflatten(1, (2 * width, 7, 7)),
        nn.Upsample(scale_factor=2), nn.Conv2d(2 * width, width, 5, padding=2), nn.Tanh(),     # 7 -> 14
        nn.Upsample(scale_factor=2), nn.Conv2d(width, 1, 5, padding=2), nn.Tanh())             # 14 -> 28
    D = nn.Sequential(
        nn.Conv2d(1, width, 5, padding=2), nn.Tanh(), nn.MaxPool2d(2),                          # 28 -> 14
        nn.Conv2d(width, 2 * width, 5, padding=2), nn.Tanh(), nn.MaxPool2d(2),                  # 14 -> 7
        nn.Flatten(), nn.Linear(2 * width * 7 * 7, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    return G, D'''

CHECK_21 = r'''with checked("27.21"):
    pair_c = make_gildenblat_21()
    if returned("27.21", "make_gildenblat_21", pair_c):
        G_c, D_c = pair_c
        G_c.eval()
        with torch.no_grad():
            same("27.21", "la forme de G(z)", list(G_c(torch.randn(3, NZ)).shape), [3, 1, 28, 28])
            same("27.21", "la forme de D(x)", list(D_c(X_MNIST[:3]).shape), [3, 1])
        same("27.21", "le nombre de paramètres de G (nz = 64, width = 16, hidden = 256)", n_params(G_c), 435_969)
        same("27.21", "le nombre de paramètres de D", n_params(D_c), 415_169)
        kinds_c = [type(m).__name__ for m in [*G_c.modules(), *D_c.modules()]]
        verdict("27.21", "Upsample" in kinds_c and "MaxPool2d" in kinds_c and "ConvTranspose2d" not in kinds_c,
                "sur-échantillonnage dans G, max-pooling dans D, comme dans le livre.",
                "le modèle du livre agrandit par Upsample et réduit par MaxPool2d, sans convolution transposée.")'''

EXPERIMENT_21 = r'''with checked("27.21"):
    # The book's model, trained exactly like G_20 (Adam 2e-4, β1 = 0.5, same epochs, same seed)
    EPOCHS_20 = 3 if FAST_MODE else 15
    torch.manual_seed(20)
    G_21, D_21 = make_gildenblat_21()
    start_21 = time.perf_counter()
    history_21 = train_gan_12(G_21, D_21, X_MNIST, EPOCHS_20, batch_size=64, lr_g=2e-4, lr_d=2e-4, seed=20)
    SECONDS_21 = (time.perf_counter() - start_21) / EPOCHS_20
    show_digits(sample_images(G_21, 30, seed=0), title=f"modèle du livre, après {EPOCHS_20} epochs")
    _ = digit_report(G_21, CLF_MNIST, title="livre   ")
if need("27.21", "G_20", "G_21"):
    print(f"{'':28s} {'paramètres G + D':>17s} {'s par epoch':>12s}")
    for name_21, G_x, D_x, sec_x in (("Radford (27.20)", G_20, D_20, SECONDS_20), ("livre (Gildenblat)", G_21, D_21, SECONDS_21)):
        print(f"{name_21:28s} {n_params(G_x) + n_params(D_x):17,d} {fr(sec_x, 1):>12s}")
    _ = digit_report(G_20, CLF_MNIST, title="Radford")'''

TODO_22 = r'''# Nothing to write here: frechet_distance and frechet_distance_from_features go in mon_travail/mylearn/gen_metrics.py'''

CHECK_22 = r'''with checked("27.22"):
    import scipy.linalg
    M = mylearn.gen_metrics
    same("27.22", "covariances diagonales : diag(4, 9) contre diag(1, 1), mêmes moyennes",
         M.frechet_distance([1, 2], np.diag([4.0, 9.0]), [1, 2], np.eye(2)), 5.0)
    same("27.22", "en dimension 1 : N(0 ; 4) contre N(1 ; 1)", M.frechet_distance([0.0], [[4.0]], [1.0], [[1.0]]), 2.0)
    rng_22 = np.random.default_rng(22)
    A_22, B_22 = rng_22.normal(size=(4, 4)), rng_22.normal(size=(4, 4))
    s1_22, s2_22 = A_22 @ A_22.T, B_22 @ B_22.T + np.eye(4)
    m1_22, m2_22 = rng_22.normal(size=4), rng_22.normal(size=4)
    want_22 = float(np.sum((m1_22 - m2_22) ** 2) + np.trace(s1_22 + s2_22 - 2 * scipy.linalg.sqrtm(s1_22 @ s2_22).real))
    same("27.22", "deux gaussiennes 4-D quelconques (oracle : scipy.linalg.sqrtm)", M.frechet_distance(m1_22, s1_22, m2_22, s2_22),
         want_22, rtol=1e-6)
    same("27.22", "une gaussienne contre elle-même", M.frechet_distance(m1_22, s1_22, m1_22, s1_22), 0.0, atol=1e-8)
    refuses("27.22", "des moyennes de tailles différentes", lambda: M.frechet_distance([0, 0], np.eye(2), [0, 0, 0], np.eye(3)))
    feats_22 = rng_22.normal(size=(300, 3))
    same("27.22", "frechet_distance_from_features : des features décalées de 1 sur chaque axe",
         M.frechet_distance_from_features(feats_22, feats_22 + 1.0), 3.0, rtol=1e-6)
    refuses("27.22", "frechet_distance_from_features : des dimensions différentes",
            lambda: M.frechet_distance_from_features(feats_22, feats_22[:, :2]))'''

TODO_23 = r'''def fid_23(G, clf, real_feats, n=2000, seed=0):
    """The Fréchet distance (your frechet_distance_from_features) between the features (features(clf, .)) of n images
    made by G (sample_images(G, n, seed)) and the features real_feats. Returns a Python float."""
    raise NotImplementedError("fid_23")'''

SOLUTION_23 = r'''def fid_23(G, clf, real_feats, n=2000, seed=0):
    """The Fréchet distance (your frechet_distance_from_features) between the features (features(clf, .)) of n images
    made by G (sample_images(G, n, seed)) and the features real_feats. Returns a Python float."""
    fake_feats = features(clf, sample_images(G, n, seed))
    return float(mylearn.gen_metrics.frechet_distance_from_features(fake_feats, real_feats))'''

CHECK_23 = r'''with checked("27.23"):
    G_c = make_toy_gan(dim=784, nz=NZ, hidden=32, seed=23)[0]
    G_c = nn.Sequential(G_c, nn.Tanh(), nn.Unflatten(1, (1, 28, 28)))
    G_c.nz = NZ
    got_c = fid_23(G_c, CLF_MNIST, REAL_FEATS, n=500, seed=3)
    if returned("27.23", "fid_23", got_c):
        same("27.23", "fid_23 d'un générateur non entraîné (500 images)", got_c,
             mylearn.gen_metrics.frechet_distance_from_features(features(CLF_MNIST, sample_images(G_c, 500, 3)), REAL_FEATS),
             rtol=1e-6)'''

EXPERIMENT_23 = r'''with checked("27.23"):
    if not need("27.23", "SNAPSHOTS_20", "G_20"):
        raise StopCheck
    G_first_23 = make_generator_20()
    G_first_23.load_state_dict(SNAPSHOTS_20[1])
    noise_23 = nn.Sequential(nn.Linear(NZ, 784), nn.Tanh(), nn.Unflatten(1, (1, 28, 28)))   # random weights: noise
    rows_23 = [("2 000 vraies images d'entraînement", mylearn.gen_metrics.frechet_distance_from_features(
                    features(CLF_MNIST, X_MNIST[:2000]), REAL_FEATS)),
               ("un réseau non entraîné (du bruit)", fid_23(noise_23, CLF_MNIST, REAL_FEATS)),
               ("G_20 après 1 epoch", fid_23(G_first_23, CLF_MNIST, REAL_FEATS)),
               (f"G_20 après {EPOCHS_20} epochs", fid_23(G_20, CLF_MNIST, REAL_FEATS))]
    if "G_21" in globals():
        rows_23.append((f"G_21, le modèle du livre, après {EPOCHS_20} epochs", fid_23(G_21, CLF_MNIST, REAL_FEATS)))
    for name_23, value_23 in rows_23:
        print(f"{name_23:48s} distance de Fréchet {value_23:9.2f}")
    print("\nLa même mesure de G_20 avec moins d'images générées (et toujours les 2 000 images de référence) :")
    for n_23 in (200, 500, 1000, 2000):
        print(f"   {n_23:5d} images : {fid_23(G_20, CLF_MNIST, REAL_FEATS, n=n_23):8.2f}")'''

GIVEN_24 = r'''class MinibatchStd(nn.Module):
    """Adds to each sample one feature: the mean, over the features, of their standard deviation over the mini-batch
    (the layer of ProGAN, Karras et al., 2018): D can see that a mini-batch is not varied enough."""

    def forward(self, x):
        return torch.cat([x, x.std(dim=0, keepdim=True).mean().expand(len(x), 1)], dim=1)


def make_ring_gan_24(minibatch_std=False, seed=0):
    """G: 2 -> 128 -> 128 -> 2 (ReLU); D: 2 -> 128 -> 128 -> 1 (LeakyReLU 0.2, a logit), with a MinibatchStd layer
    in front if minibatch_std (its first layer then takes 3 values)."""
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
    D = mlp([3 if minibatch_std else 2, 128, 128, 1])
    return G, (nn.Sequential(MinibatchStd(), D) if minibatch_std else D)


def train_ring_24(lr_g, lr_d, minibatch_std=False, steps=2000, seed=0):
    """Train make_ring_gan_24 with your gan_step_12 (Adam, betas (0.5, 0.999)) on fresh mini-batches of 256 points of
    your ring_24 at each step (rng = np.random.default_rng(seed)). Returns (G, trace): trace lists, every 500 steps,
    your modes_covered_24 for 2 000 points made from a fixed noise."""
    G, D = make_ring_gan_24(minibatch_std, seed)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    z_eval = sample_noise(2000, 2, torch.Generator().manual_seed(24))
    trace = []
    for t in range(1, steps + 1):
        real = torch.as_tensor(ring_24(256, rng), dtype=torch.float32)
        gan_step_12(G, D, opt_g, opt_d, real, sample_noise(256, 2, gen))
        if t % 500 == 0:
            trace.append(modes_covered_24(generate(G, z_eval)))
    return G, trace


RING_CENTERS_24 = np.stack([2.0 * np.cos(2 * np.pi * np.arange(8) / 8), 2.0 * np.sin(2 * np.pi * np.arange(8) / 8)], axis=1)'''

TODO_24 = r'''def ring_24(n, rng, k=8, radius=2.0, std=0.05):
    """n points of a mixture of k gaussians whose centres are evenly spaced on a circle of radius `radius` (centre j
    at the angle 2*pi*j/k), each of standard deviation std: first the components, j = rng.integers(k, size=n), then
    the noise, rng.normal(0, std, size=(n, 2)). Returns a float32 array (n, 2)."""
    raise NotImplementedError("ring_24")


def modes_covered_24(points, k=8, radius=2.0, threshold=0.15, min_share=0.01):
    """(number of covered modes, share of good points). A point is good when its distance to the nearest centre of the
    ring is below threshold; a mode is covered when at least min_share of ALL the points are good points nearest to
    its centre. Returns (int, float)."""
    raise NotImplementedError("modes_covered_24")'''

SOLUTION_24 = r'''def ring_24(n, rng, k=8, radius=2.0, std=0.05):
    """n points of a mixture of k gaussians whose centres are evenly spaced on a circle of radius `radius` (centre j
    at the angle 2*pi*j/k), each of standard deviation std: first the components, j = rng.integers(k, size=n), then
    the noise, rng.normal(0, std, size=(n, 2)). Returns a float32 array (n, 2)."""
    j = rng.integers(k, size=n)
    angle = 2 * np.pi * j / k
    centers = radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    return (centers + rng.normal(0.0, std, size=(n, 2))).astype(np.float32)


def modes_covered_24(points, k=8, radius=2.0, threshold=0.15, min_share=0.01):
    """(number of covered modes, share of good points). A point is good when its distance to the nearest centre of the
    ring is below threshold; a mode is covered when at least min_share of ALL the points are good points nearest to
    its centre. Returns (int, float)."""
    points = np.asarray(points, dtype=float)
    angles = 2 * np.pi * np.arange(k) / k
    centers = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)      # (n, k)
    nearest, good = dist.argmin(axis=1), dist.min(axis=1) < threshold
    counts = np.bincount(nearest[good], minlength=k)
    return int(np.sum(counts >= min_share * len(points))), float(good.mean())'''

CHECK_24 = r'''with checked("27.24"):
    pts_c = ring_24(5, np.random.default_rng(24))
    rng_c = np.random.default_rng(24)
    j_c = rng_c.integers(8, size=5)
    want_c = RING_CENTERS_24[j_c] + rng_c.normal(0.0, 0.05, size=(5, 2))
    if returned("27.24", "ring_24", pts_c):
        same("27.24", "ring_24(5, np.random.default_rng(24))", pts_c, want_c, rtol=1e-6, atol=1e-6)
        verdict("27.24", np.asarray(pts_c).dtype == np.float32, "les points sont en float32.",
                f"les points sont en {np.asarray(pts_c).dtype}, attendu float32 (le type des tenseurs de PyTorch).")
    # 100 points: 50 on the centre of mode 0, 30 near mode 2 (0.1 away), 1 point near mode 5, 19 at the origin
    test_c = np.concatenate([np.repeat(RING_CENTERS_24[[0]], 50, 0), np.repeat(RING_CENTERS_24[[2]] + [0.1, 0.0], 30, 0),
                             RING_CENTERS_24[[5]], np.zeros((19, 2))])
    same("27.24", "modes_covered_24 : 81 bons points, 3 modes, dont un d'un seul point", list(modes_covered_24(test_c)), [3, 0.81])
    same("27.24", "modes_covered_24 avec min_share = 0,05 (le mode d'un seul point ne compte plus)",
         list(modes_covered_24(test_c, min_share=0.05)), [2, 0.81])'''

EXPERIMENT_24 = r'''with checked("27.24"):
    SETTINGS_24 = [("équilibré (lr 1e-3 et 1e-3)", 1e-3, 1e-3, False),
                   ("D 30 fois plus rapide (lr 1e-4 et 3e-3)", 1e-4, 3e-3, False),
                   ("le même + écart-type par mini-batch", 1e-4, 3e-3, True)]
    fig_24, axes_24 = plt.subplots(1, 4, figsize=(13, 3.2))
    axes_24[0].scatter(*ring_24(2000, np.random.default_rng(0)).T, s=2, alpha=0.4)
    axes_24[0].set_title("données", fontsize=9)
    z_show_24 = sample_noise(2000, 2, torch.Generator().manual_seed(24))
    for ax_24, (name_24, lr_g_24, lr_d_24, mb_24) in zip(axes_24[1:], SETTINGS_24):
        start_24 = time.perf_counter()
        G_24, trace_24 = train_ring_24(lr_g_24, lr_d_24, minibatch_std=mb_24, seed=0)
        print(f"{name_24:42s} {time.perf_counter() - start_24:3.0f} s · modes couverts tous les 500 pas : "
              + " · ".join(f"{m} ({pct(g, 0)} bons)" for m, g in trace_24))
        ax_24.scatter(*generate(G_24, z_show_24).T, s=2, alpha=0.4, color=wb.plot.CATEGORICAL[1])
        ax_24.scatter(*RING_CENTERS_24.T, marker="x", color="black", s=20)
        ax_24.set_title(name_24, fontsize=8)
    for ax_24 in axes_24:
        ax_24.set_xlim(-3, 3)
        ax_24.set_ylim(-3, 3)
        ax_24.set_aspect("equal")
    plt.tight_layout()
    plt.show()'''

TODO_25 = r'''class CondGenerator25(nn.Module):
    """(z (N, nz), labels (N,) int64) -> images (N, 1, 28, 28) in [-1, 1]: an nn.Embedding(10, emb_dim) of the label
    (left with PyTorch's initialization), concatenated after z, then make_generator_20(nz + emb_dim, width)."""

    def __init__(self, nz=NZ, emb_dim=10, width=32):
        super().__init__()
        self.nz = nz
        raise NotImplementedError("CondGenerator25")

    def forward(self, z, labels):
        raise NotImplementedError("CondGenerator25.forward")


class CondDiscriminator25(nn.Module):
    """(images (N, 1, 28, 28), labels (N,)) -> logits (N, 1): an nn.Embedding(10, 28 * 28) of the label, reshaped into
    a second channel of 28 x 28 put after the image, then make_discriminator_20(width, in_channels=2)."""

    def __init__(self, width=32):
        super().__init__()
        raise NotImplementedError("CondDiscriminator25")

    def forward(self, images, labels):
        raise NotImplementedError("CondDiscriminator25.forward")


def cgan_step_25(G, D, opt_g, opt_d, real, labels, z, fake_labels):
    """gan_step_12 with labels: D judges (real, labels) as real and (G(z, fake_labels), fake_labels), detached, as fake;
    then G wants (G(z, fake_labels), fake_labels) judged real by the updated D. Returns the same dict as gan_step_12."""
    raise NotImplementedError("cgan_step_25")'''

SOLUTION_25 = r'''class CondGenerator25(nn.Module):
    """(z (N, nz), labels (N,) int64) -> images (N, 1, 28, 28) in [-1, 1]: an nn.Embedding(10, emb_dim) of the label
    (left with PyTorch's initialization), concatenated after z, then make_generator_20(nz + emb_dim, width)."""

    def __init__(self, nz=NZ, emb_dim=10, width=32):
        super().__init__()
        self.nz = nz
        self.embed = nn.Embedding(10, emb_dim)
        self.net = make_generator_20(nz + emb_dim, width)

    def forward(self, z, labels):
        return self.net(torch.cat([z, self.embed(labels)], dim=1))


class CondDiscriminator25(nn.Module):
    """(images (N, 1, 28, 28), labels (N,)) -> logits (N, 1): an nn.Embedding(10, 28 * 28) of the label, reshaped into
    a second channel of 28 x 28 put after the image, then make_discriminator_20(width, in_channels=2)."""

    def __init__(self, width=32):
        super().__init__()
        self.embed = nn.Embedding(10, 28 * 28)
        self.net = make_discriminator_20(width, in_channels=2)

    def forward(self, images, labels):
        return self.net(torch.cat([images, self.embed(labels).view(-1, 1, 28, 28)], dim=1))


def cgan_step_25(G, D, opt_g, opt_d, real, labels, z, fake_labels):
    """gan_step_12 with labels: D judges (real, labels) as real and (G(z, fake_labels), fake_labels), detached, as fake;
    then G wants (G(z, fake_labels), fake_labels) judged real by the updated D. Returns the same dict as gan_step_12."""
    fake = G(z, fake_labels)
    logits_real, logits_fake = D(real, labels), D(fake.detach(), fake_labels)
    loss_d = (F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
              + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake)))
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()
    logits = D(fake, fake_labels)
    loss_g = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    opt_g.zero_grad()
    loss_g.backward()
    opt_g.step()
    return {"loss_d": loss_d.item(), "loss_g": loss_g.item(),
            "d_real": torch.sigmoid(logits_real).mean().item(), "d_fake": torch.sigmoid(logits_fake).mean().item()}'''

CHECK_25 = r'''with checked("27.25"):
    torch.manual_seed(25)
    G_c, D_c = CondGenerator25(), CondDiscriminator25()
    G_c.eval()
    z_c, y_c = torch.randn(4, NZ), torch.tensor([0, 3, 7, 9])
    with torch.no_grad():
        out_c = G_c(z_c, y_c)
        same("27.25", "la forme de G(z, labels)", list(out_c.shape), [4, 1, 28, 28])
        same("27.25", "la forme de D(images, labels)", list(D_c(X_MNIST[:4], y_c).shape), [4, 1])
        same("27.25", "le nombre de paramètres de G", n_params(G_c), 10 * 10 + 240_321 + 10 * 3136)
        same("27.25", "le nombre de paramètres de D", n_params(D_c), 10 * 784 + 36_577 + 32 * 16)
        moved_c = (G_c(z_c, torch.tensor([1, 1, 1, 1])) - out_c).abs().mean().item()
        verdict("27.25", moved_c > 1e-4, "changer le label change l'image générée.", "le label ne change rien à l'image : est-il bien utilisé ?")
    G_c.train()
    G_0, D_0 = copy.deepcopy(G_c), copy.deepcopy(D_c)
    opt_g_c = torch.optim.Adam(G_c.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d_c = torch.optim.Adam(D_c.parameters(), lr=2e-4, betas=(0.5, 0.999))
    real_c, lab_c, fl_c = X_MNIST[:8], Y_MNIST[:8], torch.arange(8) % 10
    zz_c = torch.randn(8, NZ, generator=torch.Generator().manual_seed(25))
    res_c = cgan_step_25(G_c, D_c, opt_g_c, opt_d_c, real_c, lab_c, zz_c, fl_c)
    if returned("27.25", "cgan_step_25", res_c):
        with torch.no_grad():
            lr0_c, lf0_c = D_0(real_c, lab_c), D_0(G_0(zz_c, fl_c), fl_c)
            same("27.25", "loss_d (avant le pas de D)", res_c["loss_d"],
                 (F.binary_cross_entropy_with_logits(lr0_c, torch.ones_like(lr0_c))
                  + F.binary_cross_entropy_with_logits(lf0_c, torch.zeros_like(lf0_c))).item(), rtol=1e-4)'''

GIVEN_25 = r'''def train_cgan_25(G, D, n_epochs, seed=0):
    """Train a conditional GAN on X_MNIST, Y_MNIST with your cgan_step_25 (Adam 2e-4, β1 = 0.5, mini-batches of 64):
    the weights initialized with init_weights_20 (the embeddings keep their own initialization), the labels of the
    fakes drawn at random."""
    G.apply(init_weights_20)
    D.apply(init_weights_20)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_epochs):
        perm = torch.randperm(len(X_MNIST), generator=gen)
        for i in range(0, len(X_MNIST), 64):
            idx = perm[i:i + 64]
            z = sample_noise(len(idx), G.nz, gen)
            cgan_step_25(G, D, opt_g, opt_d, X_MNIST[idx], Y_MNIST[idx], z, torch.randint(0, 10, (len(idx),), generator=gen))


def conditional_images(G, labels, seed=0):
    """Images made by a conditional generator for the given labels (eval mode, no graph)."""
    labels = torch.as_tensor(labels).long()
    G.eval()
    with torch.no_grad():
        out = G(sample_noise(len(labels), G.nz, torch.Generator().manual_seed(seed)), labels)
    G.train()
    return out'''

EXPERIMENT_25 = r'''with checked("27.25"):
    torch.manual_seed(0)
    G_25, D_25 = CondGenerator25(), CondDiscriminator25()
    start_25 = time.perf_counter()
    train_cgan_25(G_25, D_25, EPOCHS_20 if "EPOCHS_20" in globals() else 3, seed=0)
    print(f"entraînement en {time.perf_counter() - start_25:.0f} s")
    show_digits(conditional_images(G_25, np.repeat(np.arange(10), 4).reshape(10, 4).T.flatten(), seed=1),
                n_cols=10, title="chaque colonne : le chiffre demandé, de 0 à 9")
    asked_25 = np.arange(2000) % 10
    got_25 = predict(CLF_MNIST, conditional_images(G_25, asked_25, seed=2))
    per_digit_25 = [np.mean(got_25[asked_25 == d] == d) for d in range(10)]
    print("le classifieur reconnaît le chiffre demandé : " + " · ".join(f"{d}:{pct(a, 0)}" for d, a in enumerate(per_digit_25))
          + f" · en moyenne {pct(np.mean(got_25 == asked_25))}")

    def spread_25(images):
        """The mean absolute pixel difference between two images of the set: 0 when they are all the same."""
        x = np.asarray(as_numpy(images)).reshape(len(images), -1)
        return float(np.mean(np.abs(x[:, None, :] - x[None, :, :])))

    print("variété à l'intérieur d'un chiffre (écart moyen entre deux images du même chiffre) : générées "
          + " · ".join(fr(spread_25(conditional_images(G_25, [d] * 40, seed=3)), 2) for d in range(10))
          + " | vraies " + " · ".join(fr(spread_25(X_MNIST[Y_MNIST == d][:40]), 2) for d in range(10)))
    verdict("27.25", np.mean(got_25 == asked_25) >= 0.5, "le label commande le chiffre dessiné (plus de la moitié reconnus, contre 10 % au hasard).",
            "le classifieur reconnaît moins de la moitié des chiffres demandés : le label est-il donné aux deux réseaux ?")'''

TODO_26 = r'''def latent_path_26(G, z0, z1, n_steps, method="linear"):
    """The images G makes along a path from z0 to z1 (both of shape (nz,)): the points of
    mylearn.vae.interpolate_latents (ch. 25) with this method, then generate. Returns a NumPy array (n_steps, 1, 28, 28)."""
    raise NotImplementedError("latent_path_26")


def path_report_26(clf, images):
    """(labels, jumps): the class predicted for each image (predict), and for each pair of consecutive images the RMS of
    their pixel differences, sqrt(mean((x_{k+1} - x_k)**2)): an array of len(images) - 1 values."""
    raise NotImplementedError("path_report_26")'''

SOLUTION_26 = r'''def latent_path_26(G, z0, z1, n_steps, method="linear"):
    """The images G makes along a path from z0 to z1 (both of shape (nz,)): the points of
    mylearn.vae.interpolate_latents (ch. 25) with this method, then generate. Returns a NumPy array (n_steps, 1, 28, 28)."""
    points = mylearn.vae.interpolate_latents(np.asarray(z0, dtype=float), np.asarray(z1, dtype=float), n_steps, method=method)
    return generate(G, points)


def path_report_26(clf, images):
    """(labels, jumps): the class predicted for each image (predict), and for each pair of consecutive images the RMS of
    their pixel differences, sqrt(mean((x_{k+1} - x_k)**2)): an array of len(images) - 1 values."""
    images = np.asarray(as_numpy(images), dtype=float).reshape(len(images), -1)
    jumps = np.sqrt(np.mean(np.diff(images, axis=0) ** 2, axis=1))
    return predict(clf, images), jumps'''

CHECK_26 = r'''with checked("27.26"):
    G_c = make_generator_20(width=8)
    z0_c, z1_c = torch.randn(NZ, generator=torch.Generator().manual_seed(26)), torch.randn(NZ, generator=torch.Generator().manual_seed(62))
    path_c = latent_path_26(G_c, z0_c, z1_c, 5, method="slerp")
    if returned("27.26", "latent_path_26", path_c):
        want_c = generate(G_c, mylearn.vae.interpolate_latents(z0_c.numpy().astype(float), z1_c.numpy().astype(float), 5, method="slerp"))
        same("27.26", "latent_path_26(..., 5, method=\"slerp\")", path_c, want_c, rtol=1e-5, atol=1e-6)
        labels_c, jumps_c = path_report_26(CLF_MNIST, path_c)
        same("27.26", "les classes prédites", labels_c, predict(CLF_MNIST, path_c))
        same("27.26", "les sauts entre images voisines", jumps_c,
             np.sqrt(((want_c[1:] - want_c[:-1]) ** 2).reshape(4, -1).mean(axis=1)), rtol=1e-5)'''

EXPERIMENT_26 = r'''with checked("27.26"):
    if not need("27.26", "G_20"):
        raise StopCheck
    rng_26 = torch.Generator().manual_seed(2626)
    pairs_26 = [(torch.randn(NZ, generator=rng_26), torch.randn(NZ, generator=rng_26)) for _ in range(3)]
    rows_26 = []
    for z0_26, z1_26 in pairs_26:
        for method_26 in ("linear", "slerp"):
            images_26 = latent_path_26(G_20, z0_26, z1_26, 10, method=method_26)
            labels_26, jumps_26 = path_report_26(CLF_MNIST, images_26)
            norms_26 = np.linalg.norm(mylearn.vae.interpolate_latents(z0_26.numpy().astype(float), z1_26.numpy().astype(float),
                                                                       10, method=method_26), axis=1)
            rows_26.append(images_26)
            print(f"{method_26:7s} classes {' '.join(map(str, labels_26))} · plus grand saut {fr(jumps_26.max(), 3)} · "
                  f"norme de z : de {fr(norms_26.max(), 1)} à {fr(norms_26.min(), 1)}")
    show_digits(np.concatenate(rows_26), n_cols=10, title="trois chemins, en ligne droite (linear) puis sur le grand cercle (slerp)")'''

GIVEN_27 = r'''DATA_27 = X_MNIST[:1280]                                  # 20 mini-batches of 64: a short training is enough here


def fresh_27(seed):
    """A small DCGAN of 27.20 (width 16, init_weights_20, drawn with torch.manual_seed(seed)) and its two Adam
    optimizers (2e-4, betas (0.5, 0.999)): (G, D, opt_g, opt_d)."""
    torch.manual_seed(seed)
    G, D = make_generator_20(width=16), make_discriminator_20(width=16)
    G.apply(init_weights_20)
    D.apply(init_weights_20)
    return (G, D, torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999)),
            torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999)))


def run_epochs_27(G, D, opt_g, opt_d, gen, data, n_epochs):
    """n_epochs of your gan_step_12 on `data`: mini-batches of 64 in the order of torch.randperm(len(data),
    generator=gen), the noise drawn from gen."""
    for _ in range(n_epochs):
        perm = torch.randperm(len(data), generator=gen)
        for i in range(0, len(data), 64):
            real = data[perm[i:i + 64]]
            gan_step_12(G, D, opt_g, opt_d, real, sample_noise(len(real), latent_dim(G), gen))


def all_states_27(G, D):
    """Every tensor of the two state_dicts (weights AND batchnorm running statistics), in order."""
    return [*G.state_dict().values(), *D.state_dict().values()]'''

TODO_27 = r'''def save_gan_checkpoint_27(path, G, D, opt_g, opt_d, epoch, gen):
    """Save in ONE file (torch.save) everything needed to resume exactly: the state_dicts of G, D, opt_g and opt_d,
    the epoch, the state of the torch.Generator gen (gen.get_state()) and PyTorch's global random state
    (torch.get_rng_state()). Only tensors, numbers, strings and dicts or lists of them: the file must reload with
    torch.load(path, weights_only=True)."""
    raise NotImplementedError("save_gan_checkpoint_27")


def load_gan_checkpoint_27(path, G, D, opt_g, opt_d, gen):
    """Restore into the given objects what save_gan_checkpoint_27 saved (gen.set_state, torch.set_rng_state);
    returns the saved epoch."""
    raise NotImplementedError("load_gan_checkpoint_27")'''

SOLUTION_27 = r'''def save_gan_checkpoint_27(path, G, D, opt_g, opt_d, epoch, gen):
    """Save in ONE file (torch.save) everything needed to resume exactly: the state_dicts of G, D, opt_g and opt_d,
    the epoch, the state of the torch.Generator gen (gen.get_state()) and PyTorch's global random state
    (torch.get_rng_state()). Only tensors, numbers, strings and dicts or lists of them: the file must reload with
    torch.load(path, weights_only=True)."""
    torch.save({"G": G.state_dict(), "D": D.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "epoch": int(epoch),
                "generator_state": gen.get_state(),
                "torch_rng_state": torch.get_rng_state()}, path)


def load_gan_checkpoint_27(path, G, D, opt_g, opt_d, gen):
    """Restore into the given objects what save_gan_checkpoint_27 saved (gen.set_state, torch.set_rng_state);
    returns the saved epoch."""
    state = torch.load(path, weights_only=True)
    G.load_state_dict(state["G"])
    D.load_state_dict(state["D"])
    opt_g.load_state_dict(state["opt_g"])
    opt_d.load_state_dict(state["opt_d"])
    gen.set_state(state["generator_state"])
    torch.set_rng_state(state["torch_rng_state"])
    return state["epoch"]'''

CHECK_27 = r'''with checked("27.27"):
    # A: two epochs in a row. B: one epoch, a checkpoint, then OTHER objects (other weights, other random state) that
    # load it and do the second epoch. B must end exactly where A ends.
    G_a, D_a, og_a, od_a = fresh_27(27)
    run_epochs_27(G_a, D_a, og_a, od_a, torch.Generator().manual_seed(27), DATA_27, 2)
    G_b, D_b, og_b, od_b = fresh_27(27)
    gen_b = torch.Generator().manual_seed(27)
    run_epochs_27(G_b, D_b, og_b, od_b, gen_b, DATA_27, 1)
    path_27 = WORK_DIR / "gan_checkpoint.pt"
    save_gan_checkpoint_27(path_27, G_b, D_b, og_b, od_b, 1, gen_b)
    if not path_27.exists():
        print(f"❌ Ex 27.27 : aucun fichier écrit en {path_27}.")
        raise StopCheck
    try:
        torch.load(path_27, weights_only=True)
        safe_27 = True
    except Exception as exc_27:
        safe_27 = False
        print(f"❌ Ex 27.27 : le fichier ne se recharge pas avec weights_only=True ({type(exc_27).__name__}) : n'y mets que des "
              "tenseurs, des nombres et des dictionnaires.")
    G_c, D_c, og_c, od_c = fresh_27(999)
    gen_c = torch.Generator().manual_seed(999)
    epoch_c = load_gan_checkpoint_27(path_27, G_c, D_c, og_c, od_c, gen_c)
    same("27.27", "l'epoch rendue par load_gan_checkpoint_27", epoch_c, 1)
    run_epochs_27(G_c, D_c, og_c, od_c, gen_c, DATA_27, 1)
    gaps_27 = [(x.double() - y.double()).abs().max().item() for x, y in zip(all_states_27(G_a, D_a), all_states_27(G_c, D_c))]
    verdict("27.27", safe_27 and max(gaps_27) == 0.0,
            "la reprise est exacte : après la seconde epoch, tous les poids et les statistiques des batchnorms sont identiques.",
            f"après la reprise, les états diffèrent (écart maximal {max(gaps_27):.2e}) : manque-t-il un optimiseur, l'état du "
            "générateur aléatoire, ou les statistiques des batchnorms ?")'''

EXPERIMENT_27 = r'''with checked("27.27"):
    # What a checkpoint with the weights only would have cost: fresh optimizers and a fresh random generator
    G_w, D_w, og_w, od_w = fresh_27(999)
    state_w = torch.load(WORK_DIR / "gan_checkpoint.pt", weights_only=True)
    G_w.load_state_dict(state_w["G"])
    D_w.load_state_dict(state_w["D"])
    run_epochs_27(G_w, D_w, og_w, od_w, torch.Generator().manual_seed(999), DATA_27, 1)
    gap_w = max((x.double() - y.double()).abs().max().item() for x, y in zip(all_states_27(G_a, D_a), all_states_27(G_w, D_w)))
    print(f"reprise avec les poids seuls : écart maximal avec l'entraînement d'une traite {gap_w:.3f} (les poids valent "
          f"environ ±{fr(max(p.abs().max().item() for p in G_a.parameters()), 2)})")
    print(f"taille du checkpoint complet : {(WORK_DIR / 'gan_checkpoint.pt').stat().st_size / 1e3:.0f} ko")'''

GIVEN_28 = r'''# Fashion-MNIST in [-1, 1], a classifier of clothes for the features, 2 000 public test images and 2 000 hidden ones
X_FASHION, Y_FASHION, XF_TEST, YF_TEST = load_images(wb.datasets.load_fashion_mnist, N_TRAIN, 2800)
start_28 = time.perf_counter()
CLF_FASHION = train_classifier(X_FASHION[:10_000], Y_FASHION[:10_000])
PUBLIC_FEATS_28 = features(CLF_FASHION, XF_TEST[:2000])     # for your trials
HIDDEN_FEATS_28 = features(CLF_FASHION, XF_TEST[2000:])     # revealed once, when the public goal is reached
GOAL_28 = {GOAL}                                            # the distance to reach (reference solution + 10 %)
MAX_SECONDS_28 = {SECONDS} if FAST_MODE else 900            # the time allowed to train_28
MAX_PARAMS_28 = 1_500_000                                   # G and D together
print(f"classifieur de vêtements entraîné en {time.perf_counter() - start_28:.0f} s · accuracy "
      f"{pct(np.mean(predict(CLF_FASHION, XF_TEST[:2000]) == YF_TEST[:2000].numpy()))} · objectif {fr(GOAL_28, 2)} en "
      f"{MAX_SECONDS_28} s au plus")'''

TODO_28 = r'''def make_gan_28():
    """Your generator and discriminator for Fashion-MNIST: (G, D), G: (N, NZ) -> (N, 1, 28, 28) in [-1, 1] (G.nz or a
    first Linear layer gives NZ to latent_dim), D: any discriminator. At most MAX_PARAMS_28 parameters in all."""
    raise NotImplementedError("make_gan_28")


def train_28(G, D, images):
    """Train G and D in place on `images` (X_FASHION, or a part of it), within MAX_SECONDS_28 seconds."""
    raise NotImplementedError("train_28")'''

SOLUTION_28 = r'''def make_gan_28():
    """Your generator and discriminator for Fashion-MNIST: (G, D), G: (N, NZ) -> (N, 1, 28, 28) in [-1, 1] (G.nz or a
    first Linear layer gives NZ to latent_dim), D: any discriminator. At most MAX_PARAMS_28 parameters in all."""
    G, D = make_generator_20({WIDTH}), make_discriminator_20({DWIDTH})
    G.apply(init_weights_20)
    D.apply(init_weights_20)
    return G, D


def train_28(G, D, images):
    """Train G and D in place on `images` (X_FASHION, or a part of it), within MAX_SECONDS_28 seconds."""
    train_gan_12(G, D, images[:10_000], {EPOCHS}, batch_size={BATCH}, lr_g={LR}, lr_d={LR}, seed=2828)'''

CHECK_28 = r'''with checked("27.28"):
    torch.manual_seed(2728)
    pair_28 = make_gan_28()
    if returned("27.28", "make_gan_28", pair_28):
        G_28, D_28 = pair_28
        count_28 = n_params(G_28) + n_params(D_28)
        if count_28 > MAX_PARAMS_28:
            print(f"❌ Ex 27.28 : {count_28:,d} paramètres, plus que les {MAX_PARAMS_28:,d} permis.")
            raise StopCheck
        start_28 = time.perf_counter()
        train_28(G_28, D_28, X_FASHION)
        seconds_28 = time.perf_counter() - start_28
        public_28 = fid_23(G_28, CLF_FASHION, PUBLIC_FEATS_28, n=2000, seed=28)
        print(f"{count_28:,d} paramètres · entraînement {seconds_28:.0f} s · distance de Fréchet publique {fr(public_28, 2)} "
              f"(objectif {fr(GOAL_28, 2)})")
        show_digits(sample_images(G_28, 30, seed=0), title="tes vêtements générés")
        if seconds_28 > MAX_SECONDS_28:
            print(f"❌ Ex 27.28 : l'entraînement a duré plus de {MAX_SECONDS_28} s.")
        elif public_28 > GOAL_28:
            print(f"❌ Ex 27.28 : il faut descendre à {fr(GOAL_28, 2)} sur les images publiques avant de révéler le jeu caché.")
        else:
            REVEALED_28 = globals().setdefault("REVEALED_28", fid_23(G_28, CLF_FASHION, HIDDEN_FEATS_28, n=2000, seed=2828))
            print(f"jeu caché (révélé une fois par session ; un nouvel essai affiche la même valeur) : {fr(REVEALED_28, 2)}")
            verdict("27.28", REVEALED_28 <= GOAL_28, f"défi réussi : {fr(REVEALED_28, 2)} sur le jeu caché.",
                    f"sur le jeu caché, {fr(REVEALED_28, 2)} : plus que les {fr(GOAL_28, 2)} visés.")'''

# ---------------------------------------------------------------------------
# 27.14: four trainings to read (history_ch27.py, "curves14")
# ---------------------------------------------------------------------------
def _curves14_cell() -> str:
    data = HISTORY.get("curves14") or {"quarters": [], "runs": {}}
    return ("# Four trainings of the toy GAN of 27.12 with other learning rates (tools/chapters/history_ch27.py, seed 14),\n"
            "# under shuffled letters: the means of loss_d, loss_g, d_real and d_fake over each quarter of epoch\n"
            f"RUNS_14 = unpack(\"{_packed({k: v for k, v in data.items() if k != 'settings'})}\")\n"
            "fig_14, axes_14 = plt.subplots(1, 4, figsize=(14, 3), sharex=True)\n"
            "for ax_14, key_14 in zip(axes_14, ('loss_d', 'loss_g', 'd_real', 'd_fake')):\n"
            "    for letter_14, run_14 in RUNS_14['runs'].items():\n"
            "        ax_14.plot(RUNS_14['quarters'], run_14[key_14], label=letter_14)\n"
            "    ax_14.set_title(key_14, fontsize=10)\n"
            "    ax_14.set_xlabel('epoch')\n"
            "for value_14 in (2 * LN2, LN2):\n"
            "    axes_14[0].axhline(value_14, color='gray', ls=':', lw=1)\n"
            "    axes_14[1].axhline(value_14, color='gray', ls=':', lw=1)\n"
            "axes_14[2].axhline(0.5, color='gray', ls=':', lw=1)\n"
            "axes_14[3].axhline(0.5, color='gray', ls=':', lw=1)\n"
            "axes_14[0].legend()\n"
            "plt.tight_layout()\n"
            "plt.show()")


TODO_14 = r'''dominant_14 = ...   # a) the letter of the run where D wins all along (it spots almost every fake)
plateau_14 = ...    # b) in run C, the last value of loss_d, read on its curve (2 decimals)
d_fake_14 = ...     # c) in run D, the last value of d_fake, read on its curve (1 decimal)
quality_14 = ...    # d) True or False: a run whose four curves end at their equilibrium values makes good samples'''

SOLUTION_14 = r'''dominant_14, plateau_14, d_fake_14, quality_14 = "D", 1.39, 0.3, False'''


def _record_14() -> str:
    runs = (HISTORY.get("curves14") or {}).get("runs") or {}
    plateau = round(runs["C"]["loss_d"][-1], 2) if runs else 1.39
    d_fake = round(runs["D"]["d_fake"][-1], 1) if runs else 0.3
    assert plateau == 1.39 and d_fake == 0.3, (plateau, d_fake)
    return ("wb.record(\"27.14a\", \"D\", mistakes={\"cet entraînement finit à l'équilibre : cherche celui où d_real reste haut et d_fake bas\": \"C\",\n"
            "                                   \"les courbes de cet entraînement finissent à l'équilibre : D ne domine pas\": \"B\",\n"
            "                                   \"ses courbes approchent de l'équilibre : cherche celui où l'écart reste grand\": \"A\"})\n"
            f"wb.record(\"27.14b\", {plateau}, decimals=2, mistakes={{\"c'est la loss de G, pas celle de D\": 0.69, \"c'est la valeur « idéale » du livre, pas la courbe\": 0.5}})\n"
            f"wb.record(\"27.14c\", {d_fake}, decimals=1, mistakes={{\"c'est d_real, pas d_fake\": {round(1 - d_fake, 1)}, \"lis la courbe d_fake de D à la fin, pas au début\": 0.0}})\n"
            "wb.record(\"27.14d\", False, mistakes={\"regarde l'entraînement B : ses courbes sont à l'équilibre ; ses échantillons le sont-ils ?\": True})")


EXPERIMENT_14 = r'''# Revealed after your answers: what each run makes (generated mean and standard deviation, W1 to the data)
print(f"{'run':>4s} {'moyenne':>8s} {'écart-type':>11s} {'W1':>7s} {'loss_g finale':>14s}")
for letter_14, run_14 in RUNS_14["runs"].items():
    print(f"{letter_14:>4s} {fr(run_14['mean'][-1], 2):>8s} {fr(run_14['std'][-1], 2):>11s} {fr(run_14['w1'], 3):>7s} "
          f"{fr(run_14['loss_g'][-1], 3):>14s}")
wb.check("27.14a", dominant_14)
wb.check("27.14b", plateau_14)
wb.check("27.14c", d_fake_14)
wb.check("27.14d", quality_14)'''


def _p17_ok() -> None:
    """The answers of 27.17 must hold on every calibration seed (history_ch27.py)."""
    rows = (HISTORY.get("calib") or {}).get("p17") or []
    for row in rows:
        assert 0.1 <= row["d_fake_epoch1"] <= 0.4, row
        assert abs(row["final_mean"] - 4) < 0.3, row
        assert row["spread_strong"] < row["spread_base"], row


def _r13_ok() -> None:
    for run in _runs13()["runs"]:
        assert run["mean"][1] > 4.5, run["seed"]
        assert min(run["std"][2:]) < 1.0 or max(run["std"][2:]) > 1.5, run["seed"]
        assert abs(run["loss_d"][-1] - 2 * 0.6931) < 0.1, run["seed"]


PART_A = Part("A", "Des GAN jouets : une gaussienne 1D, le nuage du livre, et ce qui peut mal tourner",
              "Des GAN minuscules, qui s'entraînent en quelques secondes : de quoi voir chaque mécanisme en détail. Tu "
              "écris d'abord ta mesure (la distance de Wasserstein 1D, dans ta librairie), puis la boucle d'entraînement "
              "standard, que tout le reste du chapitre réutilise. Ensuite : lire les courbes, refaire l'expérience du "
              "livre, déboguer, et tester trois affirmations du livre (le discriminateur fort, la loss, le round en "
              "quatre temps).",
              given=GIVEN_A, exercises=[
    Ex("27.11", "🔨", 1, 15, "Distance de Wasserstein 1D entre deux échantillons",
       "écrire dans ta librairie la distance de Wasserstein entre deux échantillons 1D, la mesure qui suivra tes GAN jouets.",
       "ch. 2 (fonction de répartition empirique) · fiche « Au-delà du livre : mesurer un générateur »", tracks="M, C",
       mylearn="gen_metrics.py",
       body=mylearn_howto() + "\n\n" + COMMON_GEN + r"""

Écris `wasserstein_1d(u, v)` dans `gen_metrics.py`. **Contrat** propre à cette fonction :
- `u` et `v` sont deux échantillons 1-D non vides, de tailles égales ou différentes ; un échantillon vide, un nombre seul ou un tableau 2-D lèvent `ValueError` ;
- pour deux tailles égales, le résultat est la moyenne des écarts entre les valeurs triées, appariées dans l'ordre ;
- pour deux tailles différentes, c'est l'aire entre les deux fonctions de répartition empiriques (la définition générale, qui redonne la précédente à tailles égales).

La première cellule de vérification essaie ta fonction sur quelques cas ; la seconde lance les tests (`-k "test_wasserstein_1d_"`), qui la comparent à `scipy.stats.wasserstein_distance`. Une troisième cellule montre ce que la distance voit : la même loi, une loi décalée, une loi plus étalée, et un générateur effondré sur un seul nombre.

Dans tes notes : pourquoi la distance entre les données et l'échantillon décalé de 1 vaut-elle environ 1 ? Que vaut la distance d'un générateur qui ne produit que le nombre 4, et pourquoi n'est-elle pas nulle alors que sa moyenne est parfaite ?""",
       todo=TODO_11, check=RELOAD + CHECK_11, solution=TODO_11 + "\n\n" + CHECK_11,
       after=tests_after("27.11", "mylearn.gen_metrics.wasserstein_1d([0.0], [1.0])", "test_wasserstein_1d_")
       + [("code", EXPERIMENT_11)]),

    Ex("27.12", "📦", 2, 30, "Un GAN minuscule pour une gaussienne 1D",
       "écrire la boucle d'entraînement standard d'un GAN en PyTorch et l'utiliser pour imiter une gaussienne 1D.",
       "✏️ 27.3 · ch. 23 (boucle d'entraînement PyTorch) · fiche §27.4.3", thread="gaussienne 1D", tracks="R, M, C",
       body=r"""Les données : `DATA_1D`, 10 000 nombres tirés de $\mathcal{N}(4 ;\, 1{,}25^2)$. Le GAN : `make_toy_gan(dim=1)` (dans les outils) rend un générateur 4 → 16 → 16 → 1 et un discriminateur 1 → 16 → 16 → 1, qui rend un **logit** (pas de sigmoïde en sortie).

**Contrat**
- `gan_step_12(G, D, opt_g, opt_d, real, z)` fait une itération de la boucle standard : un pas du discriminateur (vrais → label 1, faux → label 0, faux **détachés**), puis un pas du générateur à travers le discriminateur mis à jour (faux → label 1, loss non saturante), où seul `opt_g` avance. Elle rend un dictionnaire de `float` : `"loss_d"` (terme des vrais + terme des faux, deux moyennes de cross-entropy binaire), `"loss_g"`, `"d_real"` et `"d_fake"` (les probabilités moyennes $\sigma(D(\cdot))$ des vrais et des faux, vues par $D$ **avant** son pas) ;
- `train_gan_12(...)` enchaîne les epochs et les mini-batchs dans l'ordre que donne sa docstring, et rend l'historique (une valeur par pas pour chaque clé). Elle resservira tout le chapitre, aux DCGAN compris : le paramètre `step` permet d'y brancher une autre itération, `on_epoch` de regarder le générateur après chaque epoch.

La cellule de vérification fait **un** pas sur un petit GAN et contrôle les quatre valeurs rendues, puis que $D$ distingue mieux après son pas et que les faux trompent mieux $D$ après celui de $G$. La cellule suivante entraîne le GAN 10 epochs (une dizaine de secondes) et compare ce qu'il produit à la cible.

Dans tes notes : pourquoi `D(fake.detach())` à l'étape de $D$, mais `D(fake)` à l'étape de $G$ ? Pendant l'étape de $G$, `loss_g.backward()` remplit-il le `.grad` des poids de $D$ ? Pourquoi est-ce sans conséquence ? Vers quelles valeurs tendent `loss_d`, `loss_g`, `d_real` et `d_fake` (✏️ 27.4) ?""",
       todo=TODO_12, check=CHECK_12, solution=SOLUTION_12 + "\n\n\n" + CHECK_12, after=[("code", EXPERIMENT_12)]),

    Ex("27.13", "🔮", 2, 15, "Prédire la moyenne et l'écart-type générés au fil des epochs",
       "prédire comment la distribution générée rejoint la cible, avant de le voir sur cinq graines.",
       "Ex 27.12 · ✏️ 27.4 · fiche §27.5", thread="gaussienne 1D", tracks="R, C",
       body=r"""Le générateur de 27.12 part d'une sortie presque constante, près de 0 ; la cible est centrée en 4, d'écart-type 1,25. Le livre décrit, pour son nuage 2D, une trajectoire qui « dépasse » le centre avant de revenir. La cellule suivante charge les résultats de l'entraînement de 27.12 (la solution de référence) sur **cinq graines** : après chaque epoch, la moyenne et l'écart-type de 2 000 nombres générés, et les losses moyennes.

**Avant d'exécuter**, écris tes trois prédictions dans la cellule suivante :
a) la moyenne générée après l'epoch 1 : `"A"` sous 3,5 (encore en chemin), `"B"` entre 3,5 et 4,5, `"C"` au-dessus de 4,5 (au-delà de la cible) ;
b) de l'epoch 2 à l'epoch 10, l'écart-type reste-t-il entre 1,0 et 1,5 à chaque epoch, pour les cinq graines ? (`True` ou `False`) ;
c) la loss moyenne de $D$ (vrais + faux) à la dernière epoch : `"A"` environ 0,5, `"B"` environ 0,69, `"C"` environ 1,39.

La cellule d'expérience n'affiche rien tant que tes trois prédictions ne sont pas écrites.""",
       given=GIVEN_13.replace("{RUNS13}", _packed(_runs13())), todo=TODO_13, solution=SOLUTION_13, hypothesis=True,
       record=RECORD_13,
       after=[("code", guarded(EXPERIMENT_13, ["overshoot_13", "steady_13", "loss_d_13"],
                               "⏳ Ex 27.13 : écris d'abord tes trois prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : lesquelles de tes prédictions étaient justes ? Pourquoi un générateur qui part loin "
                     "de la cible peut-il la dépasser (pense au discriminateur qui, à l'epoch 1, n'a vu des faux que d'un "
                     "côté) ? L'écart-type continue de bouger alors que les losses sont à l'équilibre : qu'est-ce que cela "
                     "dit de la loss comme mesure de qualité ?")]),

    Ex("27.14", "📈", 2, 15, "Lire les courbes d'un GAN : losses, D(x) et D(G(z))",
       "reconnaître sur des courbes un GAN à l'équilibre, un discriminateur qui domine et un entraînement trop lent ou trop rapide.",
       "Ex 27.12 · ✏️ 27.4 · fiche §27.5, §27.7", thread="gaussienne 1D", tracks="R, C",
       body=r"""Quatre entraînements du GAN de 27.12, avec d'autres learning rates, sous des lettres mélangées (A à D) : la cellule suivante trace, par quart d'epoch, la loss de $D$ (vrais + faux), celle de $G$, et les probabilités moyennes `d_real` $= D(\mathbf{x})$ et `d_fake` $= D(G(\mathbf{z}))$ vues par $D$. Les lignes pointillées marquent $2\ln 2$, $\ln 2$ et $\frac{1}{2}$.

Réponds dans la cellule de réponse, en lisant les courbes :
a) la lettre de l'entraînement où $D$ domine du début à la fin (il repère presque tous les faux) ;
b) dans l'entraînement C, la dernière valeur de la loss de $D$, lue sur sa courbe (2 décimales) ;
c) dans l'entraînement D, la dernière valeur de `d_fake`, lue sur sa courbe (1 décimale) ;
d) vrai ou faux : un entraînement dont les quatre courbes finissent à leurs valeurs d'équilibre produit de bons échantillons.

La cellule d'après révèle ce que chaque entraînement produit vraiment, une fois tes quatre réponses écrites.""",
       given=_curves14_cell(), todo=TODO_14, solution=SOLUTION_14, record=_record_14(),
       after=[("code", guarded(EXPERIMENT_14, ["dominant_14", "plateau_14", "d_fake_14", "quality_14"],
                               "⏳ Ex 27.14 : écris d'abord tes quatre réponses dans la cellule précédente.")),
              ("md", "Dans tes notes : comment reconnais-tu un discriminateur qui domine sur chacune des quatre courbes ? "
                     "Deux entraînements finissent avec des courbes à l'équilibre : lequel produit de mauvais "
                     "échantillons, et pourquoi son discriminateur ne s'en aperçoit-il pas (regarde les learning rates "
                     "révélés dans 05_solutions.md) ? Quelle mesure regarderais-tu en plus des losses ?")]),

    Ex("27.15", "🎨", 2, 30, "Reproduire l'apprentissage du nuage centré en (5, 5) (fig. 27.20-27.21)",
       "refaire l'expérience du livre sur un nuage gaussien 2D et tracer son évolution, epoch par epoch.",
       "Ex 27.12 · fiche §27.5", thread="nuage 2D du livre", tracks="C",
       body=r"""L'expérience du §27.5 : 10 000 points d'une gaussienne 2D de centre $(5, 5)$ et d'écart-type 1 (`DATA_2D`), les réseaux du livre (`make_book_gan_15` : un générateur 4 → 16 → 2 et un discriminateur 2 → 16 → 16 → 1, Leaky ReLU de pente 0,1), un bruit **uniforme** sur $[0, 1]^4$, des mini-batchs de 32.

Écris :
- `train_book_15(seed, n_epochs)`, qui entraîne avec **ton** `train_gan_12` et garde les points générés à partir de `Z_FIXED_15` après les epochs de `EPOCHS_15` (0 à 10, puis 13, comme les panneaux de la fig. 27.20) ;
- `plot_clouds_15(snapshots, data)`, qui dessine un panneau par epoch, vrais et faux avec leurs ellipses (`add_ellipse`), sur les mêmes axes, et rend les moyennes et les écarts-types de chaque panneau ;
- `plot_losses_15(history, steps_per_epoch)`, qui trace les losses moyennes par epoch avec les niveaux $\ln 2$ et $2\ln 2$, et rend les deux tableaux.

La vérification compare les nombres rendus à un calcul indépendant, puis regarde si le nuage final a rejoint la cible.

Dans tes notes : retrouves-tu la traînée de départ, le dépassement du centre et l'allongement que décrit le livre ? À quelle epoch ? À quelle valeur se stabilisent les deux losses, et que vaut le « 0,5 idéal » du livre en comparaison ?""",
       given=GIVEN_15, todo=TODO_15, check=CHECK_15, solution=SOLUTION_15 + "\n\n\n" + CHECK_15),

    Ex("27.16", "🐛", 2, 25, "Le GAN qui s'entraînait à l'envers",
       "trouver et corriger trois erreurs classiques d'une boucle d'entraînement de GAN.",
       "Ex 27.12 · fiche §27.4.3 (encadré 🧮 sur detach) · ch. 20 (zero_grad)", thread="gaussienne 1D", tracks="C",
       body=r"""La cellule suivante contient une boucle d'entraînement fautive, `train_buggy_16`, dont les lignes sont repérées de (A) à (O), et trois diagnostics qui l'essaient sur `DATA_1D` : (1) la loss de $D$ ne doit pas remonter jusqu'au générateur ; (2) les gradients de $D$ doivent être remis à zéro avant chacun de ses pas ; (3) le générateur doit rejoindre les données. Les diagnostics s'arrêtent au premier point qui échoue.

1. Exécute la cellule et lis les diagnostics. Trois lignes sont fautives : écris leurs lettres dans `faulty_16` (dans l'ordre alphabétique).
2. Écris `train_fixed_16`, la version corrigée : mêmes arguments, mêmes tirages aléatoires dans le même ordre.

La vérification relance les trois diagnostics sur ta version, puis compare son générateur à celui de ton `train_gan_12` (même graine) : une boucle correcte donne exactement le même.

Dans tes notes : pour chaque erreur, ce qu'elle fait au calcul (quel gradient va où), et ce qu'on verrait sans les diagnostics. Pourquoi l'erreur (M) seule suffirait-elle à faire fuir le générateur ?""",
       given=GIVEN_16, todo=TODO_16, check=CHECK_16, solution=SOLUTION_16 + "\n\n\n" + CHECK_16, record=RECORD_16),

    Ex("27.17", "🔮", 2, 15, "Prédire l'effet d'un discriminateur trop fort",
       "prédire ce que devient l'entraînement quand le discriminateur apprend dix fois plus vite que le générateur.",
       "Ex 27.12 · fiche §27.5, §27.7", thread="gaussienne 1D", tracks="C",
       body=r"""Le livre dit qu'il faut un discriminateur « plus puissant » que le générateur, puis que si l'un des deux progresse trop vite, l'autre ne rattrape jamais. On refait l'entraînement de 27.12 (même graine, mêmes données), mais avec un learning rate de **$10^{-2}$ pour $D$** (dix fois celui de $G$), et l'on compare.

**Avant d'exécuter**, écris tes trois prédictions dans la cellule suivante :
a) avec ce discriminateur fort, la valeur moyenne de $D(G(\mathbf{z}))$ pendant la **première** epoch : `"A"` sous 0,1, `"B"` de 0,1 à 0,4, `"C"` au-dessus de 0,4 ;
b) après 10 epochs, la moyenne générée est-elle à moins de 0,3 de 4 ? (`True` ou `False`) ;
c) de l'epoch 6 à l'epoch 10, l'écart-type généré varie-t-il **moins** (écart entre son maximum et son minimum) que dans l'entraînement de 27.12 ? (`True` ou `False`).

La cellule d'expérience entraîne le GAN (quelques secondes) avec ton `train_gan_12`, une fois tes prédictions écrites.""",
       todo=TODO_17, solution=SOLUTION_17, hypothesis=True, record=RECORD_17,
       after=[("code", guarded(EXPERIMENT_17, ["d_fake_17", "reach_17", "steady_17"],
                               "⏳ Ex 27.17 : écris d'abord tes trois prédictions dans la cellule précédente.")),
              ("md", "Dans tes notes : lesquelles de tes prédictions étaient justes ? Avec la loss non saturante, que "
                     "reçoit le générateur quand $D$ repère bien les faux (∂ 27.6) ? Qu'est-ce qui pourrait, au "
                     "contraire, rendre un discriminateur fort vraiment nuisible (🔬 27.18) ? Le WGAN fait cinq pas du "
                     "critique par pas du générateur : est-ce cohérent avec ce que tu observes ?")]),

    Ex("27.18", "🔬", 2, 30, "Loss saturante ou non saturante : la dynamique du début",
       "mesurer le gradient que reçoit le générateur avec les deux losses, et voir quand la saturation bloque vraiment l'entraînement.",
       "∂ 27.6 · Ex 27.12 · ch. 19 (Adam)", thread="gaussienne 1D", tracks="M, C",
       body=r"""∂ 27.6 montre que la loss minimax du générateur, $\ln(1 - D(G(\mathbf{z})))$, donne un gradient presque nul quand $D$ repère facilement les faux. On le mesure, puis on regarde ce que cela change à l'entraînement.

Écris :
- `saturating_step_18(...)`, une itération comme `gan_step_12`, mais où le générateur minimise la loss **minimax** ;
- `generator_grad_norm_18(G, D, z, saturating)`, la norme du gradient de la loss du générateur par rapport à **tous** ses paramètres, sans pas d'optimiseur et sans laisser de `.grad` (avec `torch.autograd.grad`).

La vérification contrôle un pas et deux normes sur un petit GAN. L'expérience part d'un discriminateur **pré-entraîné** seul pendant 500 pas (`pretrain_d_18`, trois graines), affiche les deux normes du gradient, puis entraîne le générateur 600 pas avec chaque loss, une fois avec **SGD** et une fois avec **Adam** (`run_18`), en suivant la moyenne générée.

Dans tes notes : quel rapport entre les deux normes, et comment le relier à $D(G(\mathbf{z}))$ (∂ 27.6, question 4) ? Avec SGD, combien de pas la loss saturante perd-elle ? Avec Adam, que se passe-t-il, et pourquoi (pense à ce qu'Adam fait de l'échelle du gradient, ch. 19) ? Pourquoi garde-t-on quand même la loss non saturante partout ?""",
       todo=TODO_18, check=CHECK_18, solution=SOLUTION_18 + "\n\n\n" + CHECK_18,
       after=[("code", GIVEN_18), ("code", EXPERIMENT_18)]),

    Ex("27.19", "🔬", 3, 40, "Le round en quatre temps du livre contre la boucle standard",
       "programmer le round d'apprentissage du livre et le comparer à la boucle standard sur le nuage 2D.",
       "Ex 27.12 · Ex 27.15 · 27.Q4 · fiche §27.2.3 (encadré ⚠️)", thread="nuage 2D du livre", tracks="C",
       body=r"""Le livre (§27.2.3, §27.4.3) entraîne en rounds de quatre étapes, où chaque réseau n'apprend que des **erreurs** : (a) $D$ apprend des vrais qu'il prend pour des faux ; (b) $G$ apprend des faux que $D$ repère ; (c) $D$ apprend des faux qui le dupent ; (d) comme (b). La fiche affirme que ce n'est pas l'algorithme standard. On teste.

Écris `book_round_19(G, D, opt_g, opt_d, real, z_b, z_c, z_d, errors_only=True)` selon sa docstring : chaque étape choisit ses échantillons avec le $D$ du moment (logit $< 0$ : « faux » ; $\ge 0$ : « vrai »), sa loss est la cross-entropy **sommée** sur les échantillons choisis puis divisée par la taille du mini-batch, et une étape qui ne choisit rien ne fait rien. Avec `errors_only=False`, chaque étape prend tous ses échantillons.

La vérification essaie ta fonction sur deux discriminateurs extrêmes (qui répondent « vrai » à tout, ou « faux » à tout). L'expérience (`run_19`, une trentaine de secondes) entraîne les réseaux du livre sur `DATA_2D` 10 epochs de trois façons : la boucle standard (ton `gan_step_12`), le round du livre (erreurs seules), et le même round sur tous les échantillons ; elle suit la distance de Wasserstein (ta fonction, moyennée sur $x$ et $y$) et montre les nuages finaux.

Dans tes notes : lequel des trois apprend, lequel échoue, et comment ? Le problème vient-il de l'ordre des étapes (D, G, D, G) ou de la règle « n'apprendre que de ses erreurs » ? Que deviennent les gradients d'un faux bien classé, et que perd-on à les jeter ?""",
       todo=TODO_19, check=CHECK_19, solution=SOLUTION_19 + "\n\n\n" + CHECK_19,
       after=[("code", GIVEN_19), ("code", EXPERIMENT_19)]),
])


CHALLENGE_28 = {"width": 32, "dwidth": 32, "epochs": 8, "batch": 64, "lr": 2e-4}   # the reference (history_ch27.py)


def _solution_28() -> str:
    c = CHALLENGE_28
    return (SOLUTION_28.replace("{WIDTH}", f"NZ, {c['width']}").replace("{DWIDTH}", str(c["dwidth"]))
            .replace("{EPOCHS}", str(c["epochs"])).replace("{BATCH}", str(c["batch"])).replace("{LR}", repr(c["lr"])))


def _calib28() -> dict:
    """The goal of 27.28: the reference solution + 10 %, on the worse of its public and hidden distances."""
    rows = [r for r in (HISTORY.get("calib") or {}).get("challenge28", []) if r["name"] == "reference"]
    if not rows:
        return {"goal": 50.0, "seconds": 150}
    worst = max(max(r["public"], r["hidden"]) for r in rows)
    return {"goal": float(round(1.1 * worst, 1)), "seconds": 150}


PART_B = Part("B", "DCGAN sur MNIST : construire, évaluer, conditionner, reprendre",
              "Place aux images. Tu construis un DCGAN selon les règles de Radford, tu le compares au modèle du livre, "
              "puis tu écris la distance de Fréchet dans ta librairie pour **mesurer** tes générateurs sur les features "
              "d'un petit classifieur (un FID en miniature). Viennent ensuite l'effondrement de mode, provoqué puis "
              "soigné, le GAN conditionnel, l'espace latent, un checkpoint complet, et le défi sur Fashion-MNIST. La "
              "cellule suivante charge MNIST dans $[-1, 1]$ et entraîne le classifieur (une quinzaine de secondes).",
              given=GIVEN_B, exercises=[
    Ex("27.20", "📦", 3, 45, "Un DCGAN sur MNIST selon les règles de Radford",
       "construire un générateur et un discriminateur convolutifs qui suivent les règles de Radford, et les entraîner sur MNIST.",
       "✏️ 27.1 · Ex 27.12 · ch. 21 (convolution à pas) · fiche §27.6.1", thread="MNIST", tracks="R, C",
       body=r"""**Contrat** (règles de Radford, fiche §27.6.1, adaptées aux images 28 × 28 et au CPU)
- `make_generator_20(nz=NZ, width=32)` : $(N, n_z) \to (N, 1, 28, 28)$ dans $[-1, 1]$. Une seule couche dense, sans biais, qui projette le bruit sur $2w$ canaux de 7 × 7 ($w$ = `width`), puis deux convolutions transposées de noyau 4, de pas 2, de padding 1 ($7 \to 14 \to 28$) ; une batchnorm et une ReLU après chaque couche sauf la dernière, qui finit par une tanh ; les couches suivies d'une batchnorm n'ont pas de biais. Avec $n_z = 64$ et $w = 32$ : **240 321** paramètres ;
- `make_discriminator_20(width=32, in_channels=1)` : $(N, c, 28, 28) \to (N, 1)$, des **logits**. Deux convolutions de noyau 4, de pas 2, de padding 1 ($28 \to 14 \to 7$, vers $w$ puis $2w$ canaux), une Leaky ReLU de pente 0,2 après chacune, une batchnorm après la seconde seulement (pas sur la couche d'entrée ; la convolution qu'elle suit n'a pas de biais), puis une couche dense vers une valeur. Avec $w = 32$ : **36 577** paramètres ;
- `init_weights_20(module)`, à appliquer avec `G.apply(init_weights_20)` : les poids des couches denses et convolutives selon $\mathcal{N}(0 ;\, 0{,}02^2)$, ceux des batchnorms selon $\mathcal{N}(1 ;\, 0{,}02^2)$ et leurs biais à 0.

La vérification contrôle les formes, les nombres de paramètres, les règles et l'initialisation. La cellule suivante entraîne le DCGAN avec **ton** `train_gan_12` (Adam $2 \cdot 10^{-4}$, $\beta_1 = 0{,}5$, mini-batchs de 64) pendant 3 epochs sur 10 000 images en mode rapide (une trentaine de secondes ; 15 epochs sur 60 000 en mode complet 🚀), garde une copie du générateur après chaque epoch (`SNAPSHOTS_20`, réutilisé en 27.23), puis montre des échantillons et les classes que leur donne le classifieur.

Dans tes notes : à quoi ressemblent les chiffres après une epoch, puis à la fin ? Toutes les classes sont-elles représentées ? Combien de secondes par epoch ? Pourquoi `G.eval()` (que fait `sample_images`) change-t-il les images d'un même $\mathbf{z}$ par rapport au mode `train()` ?""",
       todo=TODO_20, check=CHECK_20, solution=SOLUTION_20 + "\n\n\n" + CHECK_20, after=[("code", EXPERIMENT_20)]),

    Ex("27.21", "🔬", 3, 45, "Règles empiriques à l'épreuve : une architecture de 2016 contre celle de Radford",
       "construire le DCGAN du livre (2016) et le comparer à celui de Radford, entraîné de la même façon.",
       "Ex 27.20 · fiche §27.6, §27.6.1", thread="MNIST", tracks="C",
       body=r"""Le modèle du livre (fig. 27.22, d'après Gildenblat, 2016) enfreint presque toutes les règles de Radford : des tanh partout, un sur-échantillonnage suivi d'une convolution au lieu d'une convolution transposée, du max-pooling dans le discriminateur, des couches denses cachées. On le réduit pour le CPU (largeurs divisées par 4, bruit de 64 au lieu de 100).

**Contrat** : `make_gildenblat_21(nz=NZ, width=16, hidden=256)` rend `(G, D)` :
- $G$ : une couche dense $n_z \to$ `hidden` (tanh), une couche dense vers $2w$ canaux de 7 × 7, une batchnorm, une tanh, puis deux fois « sur-échantillonnage × 2 (`nn.Upsample`), convolution de noyau 5 avec padding 2 » (vers $w$ canaux, puis 1 canal), chacune suivie d'une tanh. Avec les valeurs par défaut : **435 969** paramètres ;
- $D$ : deux fois « convolution de noyau 5 avec padding 2, tanh, max-pooling 2 × 2 » (vers $w$ puis $2w$ canaux), puis une couche dense vers `hidden` (tanh) et une couche dense vers un logit. **415 169** paramètres.

La cellule suivante l'entraîne **exactement** comme le DCGAN de 27.20 (même boucle, mêmes learning rates, mêmes epochs, même graine), puis compare la taille, le temps par epoch, les échantillons et les classes. 27.23 les comparera avec une mesure.

Dans tes notes : lequel des deux produit les chiffres les plus nets après le même nombre d'epochs ? Lequel coûte le plus cher par epoch, et pourquoi (pense aux convolutions 5 × 5 à pleine résolution et aux couches denses) ? Le livre dit qu'appliquer les règles à un modèle déjà réglé l'a dégradé : qu'est-ce qui a changé ici pour que l'expérience soit juste ?""",
       todo=TODO_21, check=CHECK_21, solution=SOLUTION_21 + "\n\n\n" + CHECK_21, after=[("code", EXPERIMENT_21)]),

    Ex("27.22", "🔨", 3, 35, "Distance de Fréchet entre deux gaussiennes (la formule du FID)",
       "écrire dans ta librairie la distance de Fréchet entre deux gaussiennes et entre deux ensembles de features.",
       "ch. 12 (covariance, valeurs propres) · ch. 2 · fiche « Au-delà du livre » (encadré 🧮 sur la trace)", tracks="M, C",
       mylearn="gen_metrics.py",
       body=mylearn_howto() + "\n\n" + COMMON_GEN + r"""

Écris `frechet_distance(mu1, sigma1, mu2, sigma2)` et `frechet_distance_from_features(feats1, feats2)` dans `gen_metrics.py`. **Contrat** propre à ces deux fonctions :
- `frechet_distance` rend $d_F^2 = \lVert \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2 \rVert^2 + \operatorname{Tr}\bigl(\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 - 2(\boldsymbol{\Sigma}_1\boldsymbol{\Sigma}_2)^{1/2}\bigr)$, le terme de la trace calculé à partir des **valeurs propres** de $\boldsymbol{\Sigma}_1\boldsymbol{\Sigma}_2$ (leur partie réelle, ramenée à 0 si elle est négative) ; un résultat négatif d'arrondi est ramené à 0 ; des matrices singulières ne doivent produire ni erreur ni avertissement ;
- les moyennes sont des vecteurs `(d,)`, les covariances des matrices `(d, d)` avec le même $d$ ; toute autre forme lève `ValueError` (pas de diffusion NumPy : une moyenne de longueur 1 contre une de longueur 3 est refusée) ;
- `frechet_distance_from_features` ajuste une gaussienne sur chaque ensemble (moyenne par colonne, covariance avec $n - 1$, comme `np.cov(..., rowvar=False)`, une seule feature comprise), puis appelle `frechet_distance` ; des ensembles qui ne sont pas 2-D, de dimensions différentes, ou de moins de 2 lignes lèvent `ValueError`.

La première cellule de vérification essaie tes fonctions (dont un cas comparé à `scipy.linalg.sqrtm`) ; la seconde lance les tests (`-k "test_frechet_distance_"`).

Dans tes notes : pourquoi passer par les valeurs propres plutôt que par `scipy.linalg.sqrtm` ? Pourquoi les valeurs propres de $\boldsymbol{\Sigma}_1\boldsymbol{\Sigma}_2$ sont-elles réelles et positives, en théorie, et pourquoi ne le sont-elles pas toujours en pratique ?""",
       todo=TODO_22, check=RELOAD + CHECK_22, solution=TODO_22 + "\n\n" + CHECK_22,
       after=tests_after("27.22", "mylearn.gen_metrics.frechet_distance([0.0], [[1.0]], [0.0], [[1.0]])",
                         "test_frechet_distance_")),

    Ex("27.23", "📦", 3, 40, "Évaluer un générateur : distance de Fréchet sur les features d'un classifieur",
       "mesurer des générateurs par la distance de Fréchet entre les features de leurs images et celles de vraies images.",
       "Ex 27.22 · Ex 27.20 · ch. 24 (features d'un CNN)", thread="MNIST", tracks="C",
       body=r"""Le FID compare les **features** d'un réseau entraîné, pas les pixels : deux images que le réseau voit pareilles ont des features proches. Ici, le réseau est le petit classifieur `CLF_MNIST` (sa couche avant-dernière, 64 valeurs) ; la référence, `REAL_FEATS`, les features de 2 000 images de test.

**Contrat** : `fid_23(G, clf, real_feats, n=2000, seed=0)` génère $n$ images avec `sample_images(G, n, seed)`, calcule leurs features avec `features(clf, ...)` et rend leur distance de Fréchet à `real_feats`, avec **ta** `frechet_distance_from_features`.

La vérification compare ta fonction à un calcul direct sur un générateur non entraîné. L'expérience mesure ensuite, sur la même échelle : 2 000 vraies images d'entraînement (le plancher), un réseau non entraîné (du bruit), le DCGAN de 27.20 après une epoch et à la fin, le modèle du livre de 27.21 ; puis le DCGAN final avec 200 à 2 000 images générées.

Dans tes notes : le classement correspond-il à ton jugement visuel de 27.20 et 27.21 ? Pourquoi les vraies images n'ont-elles pas une distance nulle ? Comment la distance varie-t-elle avec le nombre d'images générées, et que faut-il en conclure pour comparer deux FID publiés ? Pourquoi ce « FID » n'est-il pas comparable aux FID des articles ?""",
       todo=TODO_23, check=RELOAD + CHECK_23, solution=SOLUTION_23 + "\n\n\n" + CHECK_23, after=[("code", EXPERIMENT_23)]),

    Ex("27.24", "🔬", 3, 45, "Provoquer puis soigner un effondrement de mode (anneau de 8 gaussiennes)",
       "provoquer un effondrement de mode sur un mélange de gaussiennes, le mesurer, puis le soigner avec une couche d'écart-type par mini-batch.",
       "Ex 27.15 · Ex 27.11 · 27.Q10 · fiche §27.7.2", thread="anneau de gaussiennes", tracks="C",
       body=r"""Le banc d'essai classique de l'effondrement de mode : huit gaussiennes très serrées (écart-type 0,05), régulièrement réparties sur un cercle de rayon 2. Un générateur qui ne couvre que quelques-unes d'entre elles se voit tout de suite.

Écris :
- `ring_24(n, rng, ...)`, qui tire $n$ points de ce mélange, dans l'ordre de tirage que donne sa docstring ;
- `modes_covered_24(points, ...)`, qui rend le nombre de modes couverts et la part de « bons » points (à moins de 0,15 d'un centre), un mode comptant comme couvert s'il reçoit au moins 1 % de tous les points.

La vérification compare tes fonctions à des cas construits. L'expérience (`train_ring_24`, une dizaine de secondes par entraînement) entraîne trois fois le même GAN (MLP de 128 neurones, 2 000 pas, ton `gan_step_12`) : avec des learning rates égaux, avec un discriminateur trente fois plus rapide, puis le même avec une couche d'**écart-type par mini-batch** (`MinibatchStd`) en entrée du discriminateur. Elle affiche les modes couverts tous les 500 pas et les nuages finaux.

Dans tes notes : combien de modes chaque entraînement couvre-t-il, et que devient la part de bons points ? Pourquoi la couche d'écart-type par mini-batch aide-t-elle le discriminateur à voir un effondrement qu'un discriminateur point par point ne voit pas (fiche §27.5, encadré ⚠️) ? Qu'a-t-elle coûté ? Relance avec une autre graine : le résultat tient-il ?""",
       given=GIVEN_24, todo=TODO_24, check=CHECK_24, solution=SOLUTION_24 + "\n\n\n" + CHECK_24,
       after=[("code", EXPERIMENT_24)]),

    Ex("27.25", "📦", 3, 45, "Un GAN conditionnel : choisir le chiffre généré",
       "construire un GAN conditionnel qui génère le chiffre demandé, et vérifier avec le classifieur qu'il obéit.",
       "Ex 27.20 · ch. 24 (embedding) · fiche « Au-delà du livre : le GAN conditionnel »", thread="MNIST", tracks="C",
       body=r"""Un GAN conditionnel donne le label aux **deux** réseaux : le générateur reçoit $(\mathbf{z}, y)$, le discriminateur juge des couples (image, label), et une vraie image de 7 présentée avec le label 3 devient un faux.

**Contrat**
- `CondGenerator25(nz, emb_dim=10, width=32)` : un `nn.Embedding(10, emb_dim)` du label (laissé avec l'initialisation de PyTorch), concaténé **après** $\mathbf{z}$, puis le générateur de 27.20 (`make_generator_20(nz + emb_dim, width)`) ; `forward(z, labels)` ; l'attribut `nz` donne la taille du bruit ;
- `CondDiscriminator25(width=32)` : un `nn.Embedding(10, 28 * 28)` du label, remis en forme en un **second canal** de 28 × 28 placé après l'image, puis le discriminateur de 27.20 à deux canaux d'entrée (`make_discriminator_20(width, in_channels=2)`) ; `forward(images, labels)` ;
- `cgan_step_25(G, D, opt_g, opt_d, real, labels, z, fake_labels)` : `gan_step_12` avec les labels (les vrais avec leurs labels, les faux avec les labels qu'on a demandés), qui rend le même dictionnaire.

La vérification contrôle les formes, les nombres de paramètres, l'effet du label et un pas. La cellule suivante entraîne le GAN conditionnel (`train_cgan_25`, autant d'epochs qu'en 27.20, une trentaine de secondes), affiche une grille où chaque colonne demande un chiffre, compte les chiffres que le classifieur reconnaît, et compare la variété des images d'un même chiffre à celle des vraies.

Dans tes notes : quels chiffres le générateur réussit-il le mieux, et lesquels confond-il ? Les images d'un même chiffre varient-elles autant que les vraies ? Un générateur conditionnel qui ignorerait son bruit, que produirait-il, et comment s'appelle ce défaut ? Que se passerait-il si le discriminateur ne recevait pas le label ? En quoi le conditionnement par le texte des modèles d'images actuels prolonge-t-il cette idée ?""",
       given=GIVEN_25, todo=TODO_25, check=CHECK_25, solution=SOLUTION_25 + "\n\n\n" + CHECK_25,
       after=[("code", EXPERIMENT_25)]),

    Ex("27.26", "🔬", 3, 30, "Se promener dans l'espace latent du générateur",
       "parcourir l'espace latent du DCGAN entre deux bruits, en ligne droite et sur un grand cercle, et mesurer la régularité du chemin.",
       "Ex 27.20 · ch. 25 (interpolate_latents) · fiche §27.4.2", thread="MNIST", tracks="C",
       body=r"""Au ch. 25, tu as écrit `interpolate_latents` (dans ta librairie `vae.py`) : des points de $\mathbf{z}_0$ à $\mathbf{z}_1$, en ligne droite (`"linear"`) ou le long d'un grand cercle (`"slerp"`). On l'applique au DCGAN de 27.20.

Écris :
- `latent_path_26(G, z0, z1, n_steps, method)`, les images générées le long du chemin ;
- `path_report_26(clf, images)`, les classes prédites et la taille des « sauts » entre images voisines (la racine de la moyenne des carrés des écarts de pixels).

La vérification compare tes fonctions à un calcul direct. L'expérience trace trois chemins de dix images, chacun en ligne droite puis en grand cercle, avec les classes, le plus grand saut et la norme de $\mathbf{z}$ le long du chemin.

Dans tes notes : les images passent-elles d'un chiffre à l'autre en douceur, ou par un saut ? Les images du milieu sont-elles des chiffres ? Pourquoi la norme de $\mathbf{z}$ baisse-t-elle au milieu d'une ligne droite, et pourquoi est-ce gênant pour un bruit gaussien de dimension 64 (la plupart des tirages ont une norme proche de 8) ? Qu'est-ce que le grand cercle corrige ?""",
       todo=TODO_26, check=RELOAD + CHECK_26, solution=SOLUTION_26 + "\n\n\n" + CHECK_26, after=[("code", EXPERIMENT_26)]),

    Ex("27.27", "🛠️", 3, 30, "Checkpoint d'un GAN : deux réseaux, deux optimiseurs, états aléatoires, reprise à l'identique",
       "sauvegarder tout l'état d'un entraînement de GAN dans un fichier, et reprendre exactement où on s'était arrêté.",
       "Ex 27.20 · ch. 23 (checkpoint complet)", thread="MNIST", tracks="C",
       body=r"""Un entraînement de GAN dure, et une coupure (fin de session Colab, panne) ne doit pas le faire repartir de zéro. Au ch. 23, tu as sauvegardé un réseau et son optimiseur ; ici, il y a **deux** réseaux, **deux** optimiseurs (avec les moments d'Adam), les statistiques des batchnorms, l'epoch, et les générateurs aléatoires qui fixent l'ordre des mini-batchs et le bruit.

**Contrat**
- `save_gan_checkpoint_27(path, G, D, opt_g, opt_d, epoch, gen)` écrit **un** fichier avec `torch.save` : les `state_dict` de $G$, $D$, `opt_g` et `opt_d`, l'epoch, l'état du `torch.Generator` `gen` (`gen.get_state()`) et l'état aléatoire global de PyTorch (`torch.get_rng_state()`). Il ne contient que des tenseurs, des nombres, des chaînes et des dictionnaires ou listes de ceux-ci : il se recharge avec `torch.load(path, weights_only=True)`, le mode sûr ;
- `load_gan_checkpoint_27(path, G, D, opt_g, opt_d, gen)` remet tout cela dans les objets donnés et rend l'epoch.

La vérification entraîne un petit DCGAN deux epochs d'une traite, puis une epoch, un checkpoint, et la seconde epoch avec **d'autres** objets (d'autres poids, un autre état aléatoire) qui rechargent le checkpoint : les deux fins doivent être identiques, au bit près. La cellule suivante montre ce qu'aurait coûté un checkpoint des seuls poids.

Dans tes notes : qu'est-ce qui, oublié, casserait la reprise exacte (essaie : retire l'état du générateur aléatoire, puis un optimiseur) ? Pourquoi `weights_only=True` compte-t-il quand on charge un fichier téléchargé ? Comment nommerais-tu et garderais-tu les checkpoints d'un long entraînement (le dernier, le meilleur selon le FID) ?""",
       given=GIVEN_27, todo=TODO_27, check=CHECK_27, solution=SOLUTION_27 + "\n\n\n" + CHECK_27,
       after=[("code", EXPERIMENT_27)]),

    Ex("27.28", "🏆", 4, 120, "Défi DCGAN Fashion-MNIST : passer sous un seuil de distance de Fréchet",
       "concevoir et entraîner un GAN de vêtements qui atteint une distance de Fréchet fixée, sous contraintes de taille et de temps.",
       "Ex 27.20 · Ex 27.23 · fiche §27.6.1, §27.7", thread="Fashion-MNIST", tracks="C",
       body=r"""Le défi : un générateur de vêtements (Fashion-MNIST) dont la distance de Fréchet aux vraies images, mesurée par ton `fid_23` sur les features d'un classifieur de vêtements (`CLF_FASHION`), descend sous `GOAL_28`, d'abord sur 2 000 images de test publiques, puis sur 2 000 **autres**, cachées.

**Règles**
- `make_gan_28()` rend `(G, D)` : $G$ reçoit un bruit de taille `NZ` (64) et rend des images $(N, 1, 28, 28)$ dans $[-1, 1]$ ; au plus `MAX_PARAMS_28` paramètres pour les deux réseaux ;
- `train_28(G, D, images)` les entraîne sur `X_FASHION` (tout ou partie) ; tout est permis : architecture, boucle (`train_gan_12` ou la tienne), optimiseurs, learning rates, taille des mini-batchs, nombre d'epochs ; l'entraînement doit tenir en `MAX_SECONDS_28` secondes en mode rapide (900 en mode complet) ;
- règles d'honneur : n'utilise pas `XF_TEST` pour entraîner, et ne règle rien sur le jeu caché ; il n'est révélé qu'**une fois par session**, et le résultat est figé.

La vérification compte les paramètres, chronomètre `train_28`, mesure la distance publique, puis, si l'objectif est atteint, révèle le jeu caché. Le DCGAN de 27.20 tel quel (3 epochs) n'y arrive pas : à toi de trouver ce qui compte.

Dans tes notes : qu'as-tu essayé, dans quel ordre, et qu'est-ce qui a le plus aidé (epochs, largeur, taille des mini-batchs, learning rates) ? Combien de secondes ton entraînement laisse-t-il de marge ? Les vêtements générés couvrent-ils les dix classes (`digit_report` marche aussi avec `CLF_FASHION`) ?""",
       given=GIVEN_28.replace("{GOAL}", repr(_calib28()["goal"])).replace("{SECONDS}", str(_calib28()["seconds"])),
       todo=TODO_28, check=CHECK_28, solution=_solution_28() + "\n\n\n" + CHECK_28),
])


PARTS: list = [PART_A, PART_B]
# PARTS_END


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 27.Q1 à 27.Q10, 27.R1 à 27.R3, 27.1 à 27.4 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rocket = " 🚀" if ex.id in ("27.20", "27.21", "27.25", "27.28") else ""
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title}{rocket} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if kind == "exercise":
        title = "# 27 · Réseaux antagonistes génératifs (GAN) — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » se notent dans la section « Notes sur le notebook » de ta copie "
               "de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`. 🚀 : en mode complet "
               "(`FAST_MODE = False`), l'entraînement est bien plus long (60 000 images, plus d'epochs) ; un GPU aide, "
               "mais tout tourne sur CPU.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch27_gan/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 27`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 27 · Réseaux antagonistes génératifs (GAN) — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Écrire en PyTorch la boucle d'entraînement d'un GAN (deux optimiseurs, `detach`, label 1 pour le "
               "générateur) et la faire tourner sur des données jouets, puis sur des images.\n"
               "- Lire les courbes d'un GAN, déboguer une boucle fautive, tester des affirmations du livre.\n"
               "- Construire un DCGAN et un GAN conditionnel, et les **mesurer** avec tes distances de Wasserstein et "
               "de Fréchet (`mylearn.gen_metrics`).\n\n"
               "**Rappel express.** $L_D = -\\ln D(\\mathbf{x}) - \\ln(1 - D(G(\\mathbf{z})))$ (moyennes sur le "
               "mini-batch), $L_G = -\\ln D(G(\\mathbf{z}))$ (loss non saturante). À l'équilibre $D = \\frac{1}{2}$ : "
               "$L_D = 2\\ln 2 \\approx 1{,}386$, $L_G = \\ln 2 \\approx 0{,}693$. $D^* = p_{\\text{data}} / "
               "(p_{\\text{data}} + p_g)$. $W_1(u, v) = \\frac{1}{n}\\sum_i |u_{(i)} - v_{(i)}|$ (tailles égales). "
               "$d_F^2 = \\lVert \\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2 \\rVert^2 + \\operatorname{Tr}(\\boldsymbol{\\Sigma}_1 "
               "+ \\boldsymbol{\\Sigma}_2 - 2(\\boldsymbol{\\Sigma}_1\\boldsymbol{\\Sigma}_2)^{1/2})$.")]


def tools_cells() -> list:
    return [md("## Outils du notebook\n\nLa cellule suivante importe PyTorch et définit les petits outils des deux "
               "parties : `make_toy_gan`, `sample_noise`, `generate` (un générateur en mode `eval()`, sans graphe), "
               "`mean_epochs`, `add_ellipse`, `show_digits`… Lis leurs docstrings au besoin."),
            code(TOOLS)]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu écrire de mémoire les deux étapes d'une itération de GAN, et dire à quoi servent `detach` "
               "et le label 1 du générateur ?\n"
               "2. Sais-tu dire ce que valent les losses à l'équilibre, et pourquoi elles ne mesurent pas la qualité ?\n"
               "3. Sais-tu reconnaître un effondrement de mode, le mesurer et proposer un remède ?\n\n"
               "**Pour aller plus loin** : le tutoriel de Goodfellow (2016) et le tutoriel DCGAN de PyTorch, cités "
               "dans la fiche. La suite : le ch. 28, le transfert de style, puis le bonus B5 sur la diffusion, où ta "
               "distance de Fréchet resservira.")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT, PAPER_INTRO)
    cells += tools_cells()
    for part in PARTS:
        cells += part_cells(part, kind)
    cells += footer_cells(kind)
    return cells


def main() -> int:
    _r13_ok()
    _p17_ok()
    write_notebook(f"{FOLDER}/03_notebook.ipynb", build("exercise"))
    write_notebook(f"{FOLDER}/05_solutions.ipynb", build("solution"))
    n_checks = sum(len(p.subs) for p in PAPER)
    n_ex = sum(len(part.exercises) for part in PARTS)
    print(f"✅ {FOLDER}/03_notebook.ipynb and 05_solutions.ipynb written ({len(PAPER)} paper exercises, "
          f"{n_checks} paper checks, {n_ex} notebook exercises)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
