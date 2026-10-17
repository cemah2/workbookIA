#!/usr/bin/env python
"""Build the two notebooks of chapter 19 from a single source (used by Claude).

    python tools/chapters/build_ch19.py
    python tools/run_all_notebooks.py chapitres/ch19_optimiseurs/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch19_optimiseurs/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
19.1 to 19.6 and of the 🧮 estimate 19.9 (02_exercices.md). The proofs ∂ 19.7 and 19.8, the reflection (🗣️ 19.10,
⚖️ 19.11, 📄 19.12, 🛠️ 19.13) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 19.14 to 19.28) come at the second generation session: the optimizers of mylearn.optim,
their behaviour on synthetic valleys, diagnosing and comparing them, and the races on Rosenbrock and MNIST.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Paper, badge, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "19"
FOLDER = "chapitres/ch19_optimiseurs"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math


def momentum_torch_192(lrs, mu=0.9, w=1.0):
    """PyTorch momentum on L(w) = w**2 (gradient 2w): b1 = g1, b = mu b + g, w -= lr * b."""
    b = None
    for lr in lrs:
        g = 2 * w
        b = g if b is None else mu * b + g
        w = w - lr * b
    return w


def momentum_book_192(lrs, gamma=0.9, w=1.0):
    """The book's momentum: delta = gamma * delta - lr * g, w += delta."""
    delta = 0.0
    for lr in lrs:
        delta = gamma * delta - lr * 2 * w
        w = w + delta
    return w


def nesterov_book_193(steps, lr=0.1, mu=0.9, w=1.0):
    """Look-ahead Nesterov (the book, Sutskever et al.): returns the list of (look-ahead, gradient, w, v)."""
    v, out = 0.0, []
    for _ in range(steps):
        look = w + mu * v
        g = 2 * look
        v = mu * v - lr * g
        w = w + v
        out.append((look, g, w, v))
    return out


def nesterov_torch_193(steps, lr=0.1, mu=0.9, p=1.0):
    """PyTorch's Nesterov: gradient at the current point, p -= lr * (g + mu * b)."""
    b, out = None, []
    for _ in range(steps):
        g = 2 * p
        b = g if b is None else mu * b + g
        p = p - lr * (g + mu * b)
        out.append(p)
    return out


GRADS_194 = [3.0, -1.0, 2.0]


def adagrad_194(lr=0.1):
    s, w, out = 0.0, 0.0, []
    for g in GRADS_194:
        s += g * g
        w -= lr * g / math.sqrt(s)
        out.append((s, w))
    return out


def rmsprop_194(lr=0.1, alpha=0.9):
    v, w, out = 0.0, 0.0, []
    for g in GRADS_194:
        v = alpha * v + (1 - alpha) * g * g
        w -= lr * g / math.sqrt(v)
        out.append((v, w))
    return out


def adam_195(grads, lr=0.01, b1=0.9, b2=0.999, w=1.0):
    """Adam with eps = 0: returns the list of (m, v, m_hat, v_hat, w) after each step."""
    m = v = 0.0
    out = []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
        w -= lr * m_hat / math.sqrt(v_hat)
        out.append((m, v, m_hat, v_hat, w))
    return out


SIZES_199 = [3072, 512, 256, 10]
P_199 = sum(a * b + b for a, b in zip(SIZES_199[:-1], SIZES_199[1:]))
W_199 = sum(a * b for a, b in zip(SIZES_199[:-1], SIZES_199[1:]))      # weights only, without the biases'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("19.Q1", "Qu'appelle-t-on un optimiseur ?", [
        ("a", TF, "False",
         r'''mistakes={"relis le chapeau de la fiche : qui calcule les gradients ?": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"les données ont déjà servi à la passe avant et à la passe arrière": "A",
          "les labels servent à calculer la loss, avant la passe arrière": "C",
          "l'architecture ne change pas d'un pas à l'autre : que faut-il à une règle comme w ← w − η g ?": "D"}'''),
        ("c", TF, "True",
         r'''mistakes={"au même point, de quoi dépend le gradient : du réseau, des données et des poids, ou de l'optimiseur ?": False}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"backward() calcule les gradients, il ne touche pas aux poids": "A",
          "zero_grad() remet les gradients à zéro avant la passe arrière suivante": "C",
          "eval() change le mode du modèle (dropout, batchnorm), pas ses poids": "D"}'''),
    ]),
    Paper("19.Q2", "Minimum, maximum, plateau, selle : où le gradient s'annule-t-il ?", [
        ("a", LETTER, '"A"',
         r'''mistakes={"x² + 3y² : monte-t-il ou descend-il quand on s'éloigne de 0 ?": "B",
          "les deux coefficients ont-ils des signes contraires ?": "C",
          "la fonction n'est pas constante autour de 0": "D"}'''),
        ("b", LETTER, '"C"',
         r'''mistakes={"regarde le signe du coefficient de y²": "A", "regarde le signe du coefficient de x²": "B",
          "cette fonction n'est pas constante autour de 0": "D"}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"regarde le signe des deux coefficients": "A",
          "les deux coefficients ont-ils des signes contraires ?": "C",
          "cette fonction-là n'est pas constante autour de 0": "D"}'''),
        ("d", LETTER, '"D"',
         r'''mistakes={"une fonction constante monte-t-elle quand on s'éloigne du point ?": "A",
          "une fonction constante descend-elle quand on s'éloigne du point ?": "B",
          "une fonction constante a-t-elle des directions qui montent et d'autres qui descendent ?": "C"}'''),
        ("e", TF, "True",
         r'''mistakes={"écris le pas η g avec g = 0 : les poids bougent-ils ?": False}'''),
    ]),
    Paper("19.Q3", "Minima locaux ou points selles : le vrai piège en grande dimension", [
        ("a", LETTER, '"B"',
         r'''mistakes={"relis la fin du §19.2.1 : la thèse de Dauphin et coll. porte sur un autre type de point": "A",
          "les maxima sont aussi rares que les minima dans ce raisonnement": "C",
          "rien à voir avec les arrondis : relis la fin du §19.2.1": "D"}'''),
        ("b", "a whole number: the probability is 1 over …", "2**10",
         r'''fractional="« 1 sur combien » : un nombre entier (pas une probabilité)",
          mistakes={"une chance sur deux pour CHAQUE direction : on multiplie les probabilités": 10,
          "une chance sur deux pour chacune des 10 directions, toutes doivent monter": 512,
          "on multiplie les probabilités, on ne les additionne pas": 20,
          "une chance sur deux par direction, pas une sur dix": 100}'''),
        ("c", TF, "True",
         r'''mistakes={"que devient la probabilité d'un minimum quand d grandit ? Et celle d'un maximum ?": False}'''),
        ("d", TF, "True",
         r'''mistakes={"une selle a des directions qui descendent : une petite poussée au hasard peut-elle en trouver une ?": False}'''),
    ]),
    Paper("19.Q4", "Pourquoi lisser une erreur de comptage ?", [
        ("a", TF, "True",
         r'''mistakes={"le nombre d'erreurs change-t-il quand le seuil se déplace entre deux points ?": False}'''),
        ("b", TF, "True",
         r'''mistakes={"quelle est la pente d'une fonction constante ?": False}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"lisser ne rend pas le calcul plus rapide : que manque-t-il à la descente sur un escalier ?": "A",
          "un lissage ne supprime pas forcément les minima locaux": "C",
          "le problème n'est pas la valeur au minimum, c'est la pente ailleurs": "D"}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"l'accuracy compte des erreurs : elle est en escalier": "A",
          "un nombre de faux positifs est un comptage": "C",
          "le F1 se calcule à partir de comptages": "D"}'''),
    ]),
    Paper("19.Q5", "Pas constant : pourquoi rebondit-on au fond de la vallée ?", [
        ("a", "a number", "1 - 10 * 0.15",
         r'''decimals=2, mistakes={"regarde le signe de 1 − 10η": 0.5,
          "L(w) = 5w² : sa dérivée vaut 10w, la courbure est 10": 0.25,
          "le facteur dépend de la courbure, pas seulement de η": 0.85}'''),
        ("b", TF, "True",
         r'''mistakes={"le facteur de a) est-il positif ou négatif ?": False}'''),
        ("c", LETTER, '"D"',
         r'''mistakes={"calcule 1 − 10 × 0,25 : est-il positif ?": "A",
          "compare la valeur absolue du facteur à 1": "B",
          "le facteur vaut-il exactement −1 ?": "C"}'''),
        ("d", LETTER, '"C"',
         r'''mistakes={"calcule 1 − 10 × 0,2 : est-il positif ?": "A",
          "un facteur de module exactement 1 rapproche-t-il du fond ?": "B",
          "un facteur de module exactement 1 fait-il grandir l'écart ?": "D"}'''),
        ("e", TF, "True",
         r'''mistakes={"pour quels η le module du facteur est-il strictement inférieur à 1 ?": False}'''),
    ]),
    Paper("19.Q6", "Décroissance exponentielle : que vaut η après 50 pas ?", [
        ("a", "a number, 4 decimals", "0.1 * 0.99**50",
         r'''decimals=4, mistakes={"50 multiplications : la puissance est 50": 0.1 * 0.99**49,
          "50 multiplications, pas 51": 0.1 * 0.99**51,
          "une décroissance exponentielle multiplie, elle ne soustrait pas": 0.05,
          "n'oublie pas η0 = 0,1": 0.99**50}'''),
        ("b", "a whole number", "math.ceil(math.log(0.5) / math.log(0.99))",
         r'''fractional="un nombre de multiplications : un entier, le premier qui convient",
          mistakes={"après ce nombre de multiplications, η est-il déjà sous la moitié de η0 ?": 68,
          "le PREMIER nombre de multiplications qui convient": 70,
          "relis a) : après 50 multiplications, est-on déjà sous la moitié ?": 50}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"calcule 0,8 puissance 10, puis son inverse": "A", "calcule 0,8 puissance 10 : est-ce si petit ?": "C",
          "0,8 puissance 10 n'est pas si petit : refais le calcul": "D"}'''),
        ("d", TF, "True",
         r'''mistakes={"une série géométrique de raison strictement inférieure à 1 converge-t-elle ?": False}'''),
        ("e", "a number", "0.1 / (1 - 0.99)",
         r'''decimals=2, mistakes={"n'oublie pas η0 = 0,1": 100.0,
          "divise par 1 − γ, pas par γ": 0.1 / 0.99,
          "la somme d'une série géométrique divise par 1 − γ, elle ne multiplie pas": 0.001}'''),
    ]),
    Paper("19.Q7", "Plannings : exponentiel, retardé, par paliers, sur l'erreur, bold driver", [
        ("a", LETTER, '"C"',
         r'''mistakes={"η baisse-t-il à chaque epoch, ou toutes les 10 ?": "A", "y a-t-il un délai au départ ?": "B",
          "ici, c'est le calendrier qui décide, pas la loss": "D", "η ne monte jamais dans cette règle": "E"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"que se passe-t-il pendant les 5 premières epochs ?": "A",
          "η baisse-t-il par marches, ou à chaque epoch une fois lancé ?": "C",
          "ici, c'est le calendrier qui décide : la loss n'intervient pas": "D",
          "η ne monte jamais dans cette règle-là": "E"}'''),
        ("c", LETTER, '"D"',
         r'''mistakes={"ici, c'est la loss qui décide, pas un calendrier": "A",
          "ici, c'est la loss qui décide, pas un délai": "B",
          "pas de calendrier fixe dans cette règle": "C",
          "η peut-il monter dans cette règle ?": "E"}'''),
        ("d", LETTER, '"E"',
         r'''mistakes={"η augmente parfois : un planning exponentiel le fait-il ?": "A",
          "η augmente parfois : un planning retardé le fait-il ?": "B",
          "η augmente parfois : un planning par paliers le fait-il ?": "C",
          "η augmente parfois : le planning sur l'erreur le fait-il ?": "D"}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"StepLR suit un calendrier fixe": "A", "ExponentialLR multiplie à chaque epoch, quoi qu'il arrive": "B",
          "CosineAnnealingLR suit une courbe fixée d'avance": "D"}'''),
        ("f", TF, "True",
         r'''mistakes={"relis l'encadré 🕰️ du §19.3.3 : en quoi se comptent les plannings d'aujourd'hui ?": False}'''),
    ]),
    Paper("19.Q8", "Batch, stochastique, mini-batch : compter les mises à jour", [
        ("a", "a whole number", "1",
         r'''mistakes={"le batch, c'est tout le dataset : combien de mises à jour par epoch ?": 1200}'''),
        ("b", "a whole number", "1200",
         r'''mistakes={"un exemple par mise à jour : combien de mises à jour pour 1 200 exemples ?": 1}'''),
        ("c", "a whole number", "math.ceil(1200 / 64)",
         r'''fractional="un nombre de mises à jour : un entier (le dernier mini-batch, incomplet, compte aussi)",
          mistakes={"le dernier mini-batch, incomplet, compte aussi": 18,
          "combien de mini-batches de 64 dans 1 200 exemples ?": 64}'''),
        ("d", "a whole number", "50 * math.ceil(1200 / 64)",
         r'''mistakes={"le dernier mini-batch incomplet compte, à chaque epoch": 900,
          "ce sont des mini-batches de 64, pas des exemples isolés": 60000}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"la descente par batch a besoin de tous les exemples à chaque mise à jour": "A",
          "l'une des deux a besoin de garder tous les exemples": "C",
          "l'une des deux traite chaque exemple dès qu'il arrive": "D"}'''),
        ("f", TF, "False",
         r'''mistakes={"une epoch de SGD et une epoch de batch font-elles le même nombre de mises à jour ?": True}'''),
    ]),
    Paper("19.Q9", "Momentum : que se passe-t-il si γ = 0 ? si γ = 1 ?", [
        ("a", LETTER, '"A"',
         r'''mistakes={"avec μ = 0, b = g : le pas est-il nul ?": "B", "Adam a deux moyennes mobiles": "C",
          "μ = 0 ne change pas la stabilité de la descente ordinaire": "D"}'''),
        ("b", "a number", "1 / (1 - 0.9)",
         r'''decimals=1, mistakes={"b1 = 1, b2 = 1 + 0,9, b3 = 1 + 0,9 + 0,81… : la somme commence à 1": 9.0,
          "la vitesse s'accumule : déroule b2 et b3": 1.0,
          "c'est le coefficient μ, pas la limite de b": 0.9}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"avec μ = 1, b_t = b_(t−1) + 1 : s'approche-t-il d'une limite ?": "A",
          "on ajoute 1 à chaque pas : est-ce une croissance exponentielle ?": "C",
          "b change-t-il d'un pas à l'autre ?": "D"}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"b_t = b_(t−1) + 0 : la vitesse diminue-t-elle ?": "A",
          "rien ne change le signe de la vitesse quand le gradient est nul": "C"}'''),
        ("e", TF, "True",
         r'''mistakes={"relis l'histoire de la bille sur le plateau (§19.5.1)": False}'''),
    ]),
    Paper("19.Q10", "Nesterov : regarder là où l'on va", [
        ("a", LETTER, '"B"',
         r'''mistakes={"relis la formule du livre : où évalue-t-on ∇L ?": "A",
          "Nesterov regarde vers l'avant, pas en arrière": "C", "un seul gradient par pas": "D"}'''),
        ("b", TF, "False",
         r'''mistakes={"compte les hyperparamètres de la formule de Nesterov": True}'''),
        ("c", TF, "True",
         r'''mistakes={"relis la phrase sur le changement de variable (§19.5.2)": False}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"de l'autre côté du fond, dans quel sens pointe le gradient ?": "A",
          "le point anticipé a-t-il le même gradient que le point courant ?": "C",
          "le gradient s'ajoute à l'élan, il ne le remplace pas": "D"}'''),
    ]),
    Paper("19.Q11", "Adagrad, Adadelta, RMSprop : un learning rate par poids", [
        ("a", TF, "True",
         r'''mistakes={"chaque poids a-t-il son propre accumulateur ?": False}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"relis la formule d'Adagrad : oublie-t-il ses anciens gradients ?": "B",
          "le learning rate d'Adagrad est fixe : d'où vient le rétrécissement ?": "C",
          "le carré efface le signe : relis ce qu'on ajoute à s": "D"}'''),
        ("c", "a number", "1 - 0.9",
         r'''decimals=2, mistakes={"c'est le poids de l'ancienne moyenne": 0.9,
          "une part entre 0 et 1, sans pourcentage": 10.0}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"lr est le learning rate": "A", "momentum ajoute une vitesse au pas déjà normalisé": "C",
          "eps évite une division par zéro": "D"}'''),
        ("e", "a number", "0.99",
         r'''decimals=2, mistakes={"c'est la valeur du cours de Hinton et du livre, pas le défaut de PyTorch": 0.9,
          "relis la valeur par défaut de alpha dans PyTorch": 0.999}'''),
        ("f", TF, "True",
         r'''mistakes={"relis la phrase sur l'algorithme d'origine d'Adadelta (§19.5.4)": False}'''),
    ]),
    Paper("19.Q12", "Adam, No Free Lunch et choix d'un optimiseur", [
        ("a", LETTER, '"A"',
         r'''mistakes={"relis les deux moyennes d'Adam et le rôle de chacune": "B",
          "le clipping et le warmup ne font pas partie d'Adam": "C",
          "ce sont des plannings, pas des parties d'Adam": "D"}'''),
        ("b", "a number", "0.9",
         r'''decimals=3, mistakes={"relis : quel β règle la moyenne des gradients ?": 0.999,
          "relis les valeurs proposées par Kingma et Ba": 0.99}'''),
        ("c", "a number", "0.999",
         r'''decimals=3, mistakes={"relis : quel β règle la moyenne des carrés ?": 0.9,
          "c'est le défaut de RMSprop dans PyTorch, pas celui d'Adam": 0.99}'''),
        ("d", TF, "False",
         r'''mistakes={"relis l'encadré sur le No Free Lunch : en moyenne sur quoi ?": True}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"relis le bilan de la figure 19.49 : Adagrad met des milliers d'epochs": "A",
          "relis le bilan de la figure 19.49 : Adadelta n'est pas en tête": "B",
          "relis le bilan de la figure 19.49 : Adam arrive juste derrière": "D"}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("19.R1", "Du delta au gradient d'un poids", [
        ("a", "a number", "0.4 * -2.5",
         r'''decimals=2, mistakes={"multiplie a_i par δ_j, ne les additionne pas": 0.4 - 2.5,
          "a_i fois δ_j, pas δ_j divisé par a_i": -2.5 / 0.4}'''),
        ("b", "a number", "-0.05 * (0.4 * -2.5)",
         r'''decimals=3, mistakes={"w ← w − η ∂L/∂w : attention au double signe": -0.05,
          "η multiplie le gradient du poids, a_i δ_j, pas le delta seul": 0.125}'''),
        ("c", 'a shape as a string, e.g. "(3, 5)"', '"(64, 10)"',
         r'''mistakes={"le gradient a la forme du poids, (n_in, n_out) : relis la transposée": "(10, 64)",
          "on somme sur les exemples : le gradient a la forme du poids": "(32, 10)",
          "Aᵀ Δ : (64, 32) fois (32, 10)": "(32, 32)"}'''),
        ("d", TF, "False",
         r'''mistakes={"l'optimiseur reçoit le gradient : le calcule-t-il ?": True}'''),
    ]),
    Paper("19.R2", "Pourquoi l'initialisation He garde la variance stable", [
        ("a", "a number", "2 / 200",
         r'''decimals=3, mistakes={"He : 2 / n_in, pas 1 / n_in": 0.005, "c'est l'écart-type, on demande la variance": 0.1,
          "c'est la formule de Glorot, avec n_out": 0.008}'''),
        ("b", "a number", "math.sqrt(2 / 200)",
         r'''decimals=3, mistakes={"c'est la variance, on demande l'écart-type": 0.01,
          "He : variance 2 / n_in, puis la racine": math.sqrt(1 / 200)}'''),
        ("c", "a number", "2 / (200 + 50)",
         r'''decimals=4, mistakes={"c'est He : Glorot fait intervenir n_out aussi": 0.01,
          "Glorot : 2 / (n_in + n_out), pas 1 / (n_in + n_out)": 0.004,
          "additionne n_in et n_out avant de diviser": 0.04}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"la formule porte sur une seule couche": "B", "les biais partent de 0": "C",
          "la dérivée de ReLU vaut 0 ou 1": "D"}'''),
    ]),
    Paper("19.R3", "Standardiser les features, et l'effet sur la descente", [
        ("a", "a whole number", "round(1e6 / 0.25)",
         r'''mistakes={"le rapport de la plus grande à la plus petite : divise, ne multiplie pas": 250000,
          "le rapport des valeurs propres, pas de leurs racines": 2000}'''),
        ("b", "a number of millionths (a multiple of 1e-6)", "2 / 1e6 / 1e-6",
         r'''decimals=1, mistakes={"quelle courbure limite le learning rate : la plus grande ou la plus petite ?": 8000000.0,
          "la limite de convergence est 2 / λ_max : relis l'énoncé": 1.0,
          "en millionièmes : écris le nombre de millionièmes, pas la valeur": 0.000002}'''),
        ("c", "a whole number", "1",
         r'''mistakes={"après standardisation, que vaut la moyenne des carrés de chaque feature ?": 4000000}'''),
        ("d", TF, "True",
         r'''mistakes={"relis l'encadré : diviser chaque gradient par sa taille habituelle compense-t-il les échelles ?": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("19.1", "Plannings exponentiel et par paliers à la main", [
        ("a", "a number, 4 decimals", "0.2 * 0.9**3",
         r'''decimals=4, mistakes={"à l'epoch 0, η vaut η0 : combien de multiplications à l'epoch 3 ?": 0.2 * 0.9**4,
          "à l'epoch 3, η a été multiplié 3 fois": 0.2 * 0.9**2, "n'oublie pas η0 = 0,2": 0.9**3}'''),
        ("b", "a number, 4 decimals", "0.2 * 0.9**10",
         r'''decimals=4, mistakes={"à l'epoch 10, η a été multiplié 10 fois": 0.2 * 0.9**9,
          "10 multiplications, pas 11": 0.2 * 0.9**11}'''),
        ("c", "a number", "0.2 * 0.5**(3 // 4)",
         r'''decimals=3, mistakes={"⌊3/4⌋ vaut combien ?": 0.1}'''),
        ("d", "a number", "0.2 * 0.5**(4 // 4)",
         r'''decimals=3, mistakes={"⌊4/4⌋ vaut 1 : la baisse a lieu À l'epoch 4": 0.2,
          "⌊4/4⌋ vaut combien ?": 0.05}'''),
        ("e", "a number", "0.2 * 0.5**(11 // 4)",
         r'''decimals=3, mistakes={"⌊11/4⌋ vaut combien ?": 0.025, "compte les paliers déjà franchis à l'epoch 11": 0.1}'''),
        ("f", "a number, 4 decimals", "0.5**(1 / 4)",
         r'''decimals=4, mistakes={"le planning multiplie : il faut γ⁴ = 0,5, pas une baisse de 12,5 % par epoch": 0.875,
          "une division par 2 toutes les 4 epochs, pas toutes les 2": 0.5**0.5,
          "γ s'applique à chaque epoch : sa puissance 4 doit valoir 0,5": 0.5}'''),
        ("g", "a number, 4 decimals", "0.2 * 0.9**(8 - 4)",
         r'''decimals=4, mistakes={"l'exposant compte les epochs depuis la fin du palier": 0.2 * 0.9**8,
          "k − 4 pour k = 8 : combien d'epochs après le palier ?": 0.2 * 0.9**3,
          "l'exposant est k − 4, pas k − 3": 0.2 * 0.9**5}'''),
    ]),
    Paper("19.2", "Trois pas de momentum, deux conventions", [
        ("a", "a number", "2 * 1.0",
         r'''decimals=2, mistakes={"b1 = g1 : ni η ni (1 − μ) n'y entrent": 0.2, "g = 2w, au point w0 = 1": 1.0}'''),
        ("b", "a number", "momentum_torch_192([0.1])",
         r'''decimals=2, mistakes={"w1 = w0 − η b1, avec b1 = g1": 0.98, "on descend : w − η b": 1.2}'''),
        ("c", "a number", "0.9 * 2 + 2 * momentum_torch_192([0.1])",
         r'''decimals=2, mistakes={"g2 se calcule au point w1, pas en w0": 3.8, "b2 = μ b1 + g2 : n'oublie pas μ b1": 1.6}'''),
        ("d", "a number", "momentum_torch_192([0.1, 0.1])",
         r'''decimals=2, mistakes={"le pas utilise la vitesse b2, pas le gradient g2": 0.64,
          "g2 se calcule au point w1": 0.42}'''),
        ("e", "a number, 3 decimals", "momentum_torch_192([0.1, 0.1, 0.1])",
         r'''decimals=3, mistakes={"le pas utilise la vitesse b3, pas le gradient g3": 0.368}'''),
        ("f", TF, "abs(momentum_book_192([0.1, 0.1]) - momentum_torch_192([0.1, 0.1])) < 1e-12",
         r'''mistakes={"déroule Δ1 = −η g1 et Δ2 = γ Δ1 − η g2, puis compare": False}'''),
        ("g", "a number", "momentum_torch_192([0.1, 0.05])",
         r'''decimals=2, mistakes={"relis la convention de PyTorch : où intervient η ?": 0.54,
          "le learning rate a changé au deuxième pas": 0.46}'''),
        ("h", "a number", "momentum_book_192([0.1, 0.05])",
         r'''decimals=2, mistakes={"relis la convention du livre : Δ1 garde le η du premier pas": 0.63,
          "le learning rate a changé au deuxième pas, dans Δ2": 0.46}'''),
    ]),
    Paper("19.3", "Nesterov : gradient anticipé et reformulation de PyTorch", [
        ("a", "a number", "nesterov_book_193(2)[1][0]",
         r'''decimals=3, mistakes={"le point anticipé ajoute μ v1 à w1": 0.8, "v1 est négatif : attention au signe": 0.98}'''),
        ("b", "a number", "nesterov_book_193(2)[1][1]",
         r'''decimals=3, mistakes={"le gradient se prend au point anticipé, pas en w1": 1.6,
          "v1 est négatif : refais le point anticipé": 1.96}'''),
        ("c", "a number", "nesterov_book_193(2)[1][2]",
         r'''decimals=3, mistakes={"le gradient qui entre dans v2 se prend au point anticipé": 0.46,
          "v2 = μ v1 − η g : n'oublie pas μ v1": 0.676}'''),
        ("d", "a number, 4 decimals", "nesterov_book_193(3)[2][2]",
         r'''decimals=4, mistakes={"à chaque pas, le gradient se prend au point anticipé": 0.062}'''),
        ("e", "a number", "nesterov_torch_193(1)[0]",
         r'''decimals=3, mistakes={"le pas de PyTorch est η (g + μ b), pas η b": 0.8}'''),
        ("f", "a number, 4 decimals", "nesterov_torch_193(2)[1]",
         r'''decimals=4, mistakes={"le pas de PyTorch combine g2 et μ b2": 0.496,
          "le pas utilise g + μ b, pas b seul": 0.316}'''),
        ("g", LETTER, '"B"',
         r'''mistakes={"compare p2 à w2 : sont-ils égaux ?": "A", "compare p2 à w3": "C",
          "v2 est une vitesse, p2 une position": "D"}'''),
    ]),
    Paper("19.4", "Adagrad et RMSprop sur une suite de gradients", [
        ("a", "a number", "adagrad_194()[1][0]",
         r'''decimals=1, mistakes={"s2 = s1 + g2² : le carré de −1 est positif": 8.0,
          "s accumule TOUS les carrés depuis le début": 1.0}'''),
        ("b", "a number, 4 decimals", "adagrad_194()[1][1]",
         r'''decimals=4, mistakes={"g2 est négatif : le pas fait monter w": -0.1 - 0.1 / math.sqrt(10),
          "divise g2 par la racine de s2, pas de s1": -0.1 + 0.1 / 3}'''),
        ("c", "a number, 4 decimals", "adagrad_194()[2][1]",
         r'''decimals=4, mistakes={"s3 = s2 + g3² : le carré de 2": adagrad_194()[1][1] - 0.2 / math.sqrt(12),
          "le troisième pas se divise par la racine de s3, pas de s2": adagrad_194()[1][1] - 0.2 / math.sqrt(10)}'''),
        ("d", "a number", "rmsprop_194()[0][0]",
         r'''decimals=2, mistakes={"le coefficient du nouveau carré est 1 − α": 8.1, "c'est le CARRÉ du gradient qui entre dans v": 0.3}'''),
        ("e", "a number, 4 decimals", "rmsprop_194()[0][1]",
         r'''decimals=4, mistakes={"divise par la racine de v1, celle de d)": -0.1,
          "divise par la racine de v1, pas par v1": -0.1 * 3 / 0.9}'''),
        ("f", "a number", "rmsprop_194()[2][0]",
         r'''decimals=3, mistakes={"continue jusqu'au troisième pas": 0.91,
          "v3 = α v2 + (1 − α) g3²": 0.91 + 0.4}'''),
        ("g", "a number, 4 decimals", "rmsprop_194()[2][1]",
         r'''decimals=4, mistakes={"continue jusqu'au troisième pas : w3": rmsprop_194()[1][1]}'''),
        ("h", "a number, 4 decimals", "1 / math.sqrt(1 - 0.9)",
         r'''decimals=4, mistakes={"l'inverse : le premier pas de RMSprop est-il plus grand ou plus petit ?": math.sqrt(0.1),
          "il y a une racine : le rapport des racines": 10.0}'''),
        ("i", "a number", "1 / math.sqrt(100)",
         r'''decimals=2, mistakes={"Adagrad divise par la RACINE de la somme": 0.01, "l'inverse : le pas grandit-il ?": 10.0}'''),
    ]),
    Paper("19.5", "Un pas d'Adam et la correction de biais", [
        ("a", "a number", "adam_195([0.5])[0][0]",
         r'''decimals=3, mistakes={"m1 = β1 m0 + (1 − β1) g1, avec m0 = 0": 0.45, "c'est m̂1 : on demande m1": 0.5}'''),
        ("b", "a number, 5 decimals", "adam_195([0.5])[0][1]",
         r'''decimals=6, mistakes={"c'est le CARRÉ du gradient qui entre dans v": 0.0005, "c'est v̂1 : on demande v1": 0.25}'''),
        ("c", "a number", "adam_195([0.5])[0][2]",
         r'''decimals=3, mistakes={"c'est m1 : divise-le par 1 − β1": 0.05, "divise par 1 − β1, pas par β1": 0.05 / 0.9}'''),
        ("d", "a number", "adam_195([0.5])[0][4]",
         r'''decimals=3, mistakes={"on descend : w − η m̂ / √v̂": 1.01, "le pas d'Adam est η m̂ / √v̂, pas η g": 0.995}'''),
        ("e", "a number, 4 decimals", "0.01 * 0.05 / math.sqrt(0.00025)",
         r'''decimals=4, mistakes={"c'est le pas corrigé : ici, sans correction": 0.01,
          "divise par la RACINE de v1": 0.01 * 0.05 / 0.00025}'''),
        ("f", "a number", "adam_195([0.5, -0.1])[1][0]",
         r'''decimals=3, mistakes={"g2 est négatif": 0.055, "m2 = β1 m1 + (1 − β1) g2 : n'oublie pas β1 devant m1": 0.04}'''),
        ("g", "a number, 4 decimals", "adam_195([0.5, -0.1])[1][2]",
         r'''decimals=4, mistakes={"corrige m2 avec 1 − β1²": 0.035, "au deuxième pas, t = 2 : 1 − β1², pas 1 − β1": 0.35}'''),
        ("h", "a number, 5 decimals", "adam_195([0.5, -0.1])[1][4]",
         r'''decimals=5, mistakes={"le deuxième pas d'Adam vaut-il encore η exactement ?": 0.98,
          "regarde le signe de m̂2 : w monte-t-il ou baisse-t-il ?": 0.99 + (0.99 - adam_195([0.5, -0.1])[1][4])}'''),
        ("i", TF, "adam_195([0.5, -0.1])[1][4] < adam_195([0.5])[0][4]",
         r'''mistakes={"regarde le signe de m̂2": False}'''),
        ("j", TF, "abs(adam_195([500.0, -100.0])[1][4] - adam_195([0.5, -0.1])[1][4]) < 1e-12",
         r'''mistakes={"multiplie tous les g par 1 000 : que deviennent m et √v ?": False}'''),
    ]),
    Paper("19.6", "Adam + L2 ou AdamW : la même pénalité ? Convertir alpha, lambda et weight_decay", [
        ("a", "a number", "0.01",
         r'''decimals=3, mistakes={"dérive (λ/2)‖w‖² : le ½ disparaît": 0.02, "dérive (λ/2)‖w‖² : pas de ½ en plus": 0.005}'''),
        ("b", "a number", "2 * 0.01",
         r'''decimals=3, mistakes={"sans le ½, la dérivée de λ‖w‖² fait apparaître un 2": 0.01,
          "sans le ½ : la dérivée multiplie, elle ne divise pas": 0.005}'''),
        ("c", "a number", "2 - 0.1 * 0.01 * 2",
         r'''decimals=3, mistakes={"le weight decay est multiplié par le learning rate": 1.98,
          "le terme ajouté au gradient est wd × w, avec w = 2": 1.999}'''),
        ("d", "a number, 5 decimals", "1 - 1e-3 * 1e-2",
         r'''decimals=5, mistakes={"le facteur est 1 − lr × wd : multiplie wd par lr": 0.99,
          "le facteur est 1 − lr × wd, pas 1 − lr": 0.999}'''),
        ("e", "a number, 4 decimals", "(1 - 1e-3 * 1e-2) ** 10_000",
         r'''decimals=4, mistakes={"c'est une puissance du facteur, pas une soustraction répétée": 0.9,
          "10 000 pas : la puissance est 10 000": (1 - 1e-5) ** 1000}'''),
        ("f", "a number", "2 - 1e-3",
         r'''decimals=3, mistakes={"couplé : le terme de décroissance passe par la division par √v̂ d'Adam": 2 * (1 - 1e-5),
          "le premier pas d'Adam a une taille fixe : laquelle ?": 1.98}'''),
        ("g", "a number, 5 decimals", "2 * (1 - 1e-3 * 1e-2)",
         r'''decimals=5, mistakes={"découplé : la décroissance ne passe pas par les moyennes d'Adam": 1.999,
          "la décroissance est multipliée par le learning rate": 1.98}'''),
        ("h", "a number", "1e-4 / 1e-3",
         r'''decimals=2, mistakes={"PyTorch multiplie le weight_decay par le learning rate : il faut compenser": 0.0001}'''),
        ("i", "a number of 1e-7", "1e-4 / 200 / 1e-7",
         r'''decimals=1, mistakes={"divise alpha par la taille du mini-batch": 1000.0,
          "en multiple de 10⁻⁷ : compte bien les zéros": 0.5}'''),
        ("j", TF, "True",
         r'''mistakes={"écris les deux mises à jour de la descente vanille et compare-les": False}'''),
    ]),
    Paper("19.9", "Fermi : la mémoire des optimiseurs, du MLP au LLM", [
        ("a", "a whole number", "P_199",
         r'''mistakes={"les biais sont des paramètres aussi": W_199, "compte aussi la dernière couche, 256 × 10 + 10": P_199 - 2570}'''),
        ("b", "a number of MB, 1 decimal", "4 * P_199 / 1e6",
         r'''decimals=1, mistakes={"4 octets par nombre float32": P_199 / 1e6, "32 bits = 4 octets : compte en octets": 32 * P_199 / 1e6,
          "float32 = 4 octets, pas 2": 2 * P_199 / 1e6}'''),
        ("c", "a number of MB, 1 decimal", "2 * 4 * P_199 / 1e6",
         r'''decimals=1, mistakes={"Adam garde deux nombres par paramètre": 4 * P_199 / 1e6,
          "les états seulement, sans les paramètres ni les gradients": 4 * 4 * P_199 / 1e6}'''),
        ("d", "a number of MB, 1 decimal", "4 * 4 * P_199 / 1e6",
         r'''decimals=1, mistakes={"paramètres, gradients ET deux états": 3 * 4 * P_199 / 1e6,
          "il faut aussi les paramètres et les gradients": 2 * 4 * P_199 / 1e6}'''),
        ("e", "a whole number of GB", "round(7e9 * (2 + 2 + 4 + 4 + 4) / 1e9)",
         r'''fractional="un nombre de Go : un entier",
          mistakes={"compte bien chaque poste : la copie maîtresse float32 et les deux tampons d'Adam, m et v": 84,
          "additionne tous les octets par paramètre": 56}'''),
        ("f", "a whole number of GB", "round(7e9 * (2 + 2 + 4 + 4) / 1e9)",
         r'''fractional="un nombre de Go : un entier",
          mistakes={"SGD avec momentum garde UN tampon": 112, "n'oublie pas la copie maîtresse float32": 56}'''),
        ("g", "a whole number", "math.ceil(112 / 80)",
         r'''fractional="un nombre de GPU : un entier, arrondi au-dessus",
          mistakes={"112 Go tiennent-ils dans un GPU de 80 Go ?": 1}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 19.1 à 19.6 et 🧮 19.9)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; une "
               "forme : entre guillemets (`\"(3, 5)\"`). Arrondis comme l'énoncé le demande, et seulement à la fin "
               "du calcul. Les réponses pas encore remplies affichent ⏳. Les questions « dans ta copie », les "
               "preuves ∂ 19.7 et 19.8, la réflexion (🗣️ 19.10, ⚖️ 19.11, 📄 19.12, 🛠️ 19.13) et l'entretien se "
               "corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 19.14 to 19.28): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "19.14 à 19.19", "un pas constant sur une vallée, puis ta librairie optim.py : plannings, SGD, Momentum "
                           "et Nesterov, Adagrad, RMSprop et Adadelta, Adam et AdamW, warmup et cosinus, clipping"),
    ("B", "19.20 à 19.22", "trop de momentum, les figures du chapitre, un Adam qui diverge"),
    ("C", "19.23 à 19.25", "lire des courbes de loss d'optimiseurs, les solveurs de MLPClassifier, batch contre "
                           "stochastique contre mini-batch sur les lunes"),
    ("D", "19.26 à 19.28", "une course d'optimiseurs sur la fonction de Rosenbrock, huit optimiseurs sur MNIST, le "
                           "défi des 98 % en cinq epochs"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 19.Q1 à 19.Q12, 19.R1 à 19.R3, 19.1 à 19.6, 19.9 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 | "
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
        title = "# 19 · Optimiseurs — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch19_optimiseurs/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 19`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 19 · Optimiseurs — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Calculer à la main les premiers pas d'un planning, du momentum, de Nesterov, d'Adagrad, de "
               "RMSprop et d'Adam.\n"
               "- Écrire et vérifier contre `torch.optim` les optimiseurs et les plannings de `mylearn.optim`.\n"
               "- Comparer des optimiseurs sur des vallées, sur la fonction de Rosenbrock et sur MNIST, et défendre "
               "un choix.\n\n"
               "**Rappel express** (conventions de PyTorch, $g$ le gradient, $t$ compté à partir de 1). Momentum : "
               "$b = \\mu b + g$ ($b_1 = g_1$), $w \\leftarrow w - \\eta b$ ; Nesterov : $w \\leftarrow w - \\eta(g + \\mu b)$ ; "
               "Adagrad : $s \\leftarrow s + g^2$, $w \\leftarrow w - \\eta g/(\\sqrt{s} + \\epsilon)$ ; RMSprop : "
               "$v \\leftarrow \\alpha v + (1 - \\alpha) g^2$, $w \\leftarrow w - \\eta g/(\\sqrt{v} + \\epsilon)$ ; Adam : "
               "$m \\leftarrow \\beta_1 m + (1 - \\beta_1) g$, $v \\leftarrow \\beta_2 v + (1 - \\beta_2) g^2$, "
               "$w \\leftarrow w - \\eta\\,\\frac{m/(1 - \\beta_1^t)}{\\sqrt{v/(1 - \\beta_2^t)} + \\epsilon}$ ; AdamW : "
               "$w \\leftarrow w(1 - \\eta\\lambda)$ puis le pas d'Adam. Plannings : $\\eta_0\\gamma^k$, "
               "$\\eta_0\\gamma^{\\lfloor k/s \\rfloor}$, warmup $\\eta_{\\max} t/T_w$ puis cosinus.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 19.14 à 19.28 (parties A à D, ta librairie `optim.py`) seront ajoutés à la "
             "prochaine session de génération ; relance alors `python tools/start_chapter.py 19` pour obtenir le "
             "notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu dérouler à la main deux pas de momentum, de RMSprop et d'Adam, correction de biais "
               "comprise ?\n"
               "2. Sais-tu expliquer pourquoi la courbure limite le learning rate, et ce que change une vallée "
               "étroite ?\n"
               "3. Sais-tu dire la différence entre Adam avec `weight_decay` et AdamW, et ce que font le warmup, le "
               "cosinus et le clipping ?\n\n"
               "**Pour aller plus loin** : la synthèse de S. Ruder et l'article interactif de G. Goh sur le "
               "momentum, cités dans la fiche. La suite : le ch. 20, où PyTorch fera tout cela avec "
               "`torch.optim`." + later)]


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
