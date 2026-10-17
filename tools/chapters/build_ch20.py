#!/usr/bin/env python
"""Build the two notebooks of chapter 20 from a single source (used by Claude).

    python tools/chapters/build_ch20.py
    python tools/run_all_notebooks.py chapitres/ch20_deep_learning/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch20_deep_learning/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
20.1 to 20.4, 20.6 and 20.7 and of the 🧮 estimate 20.9 (02_exercices.md). The proofs ∂ 20.5 and 20.8, the
reflection (🗣️ 20.10, ⚖️ 20.11, 📄 20.12) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 20.13 to 20.31) come at the second generation session: first PyTorch tensors and
autograd, the library mylearn.nn.regularization, a complete PyTorch training loop, and the regularization
experiments on Fashion-MNIST.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Paper, badge, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402

CHAPTER = "20"
FOLDER = "chapitres/ch20_deep_learning"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np

# Ex 20.3: inverted dropout on five activations, keep a neuron when u >= p
A_203 = np.array([2.0, -1.0, 0.5, 3.0, 4.0])
U_203 = np.array([0.1, 0.7, 0.35, 0.9, 0.5])
P_203 = 0.4
MASK_203 = (U_203 >= P_203) / (1 - P_203)
OUT_203 = A_203 * MASK_203

# Ex 20.4: batch normalization of a mini-batch of four examples, eps = 0
X_204 = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 6.0], [7.0, 10.0]])
GAMMA_204, BETA_204, MOMENTUM_204 = np.array([2.0, 0.5]), np.array([0.0, 1.0]), 0.1
MU_204, VAR_204 = X_204.mean(axis=0), X_204.var(axis=0)                     # biased variance
XHAT_204 = (X_204 - MU_204) / np.sqrt(VAR_204)
Y_204 = GAMMA_204 * XHAT_204 + BETA_204
RUN_MEAN_204 = (1 - MOMENTUM_204) * np.zeros(2) + MOMENTUM_204 * MU_204
RUN_VAR_204 = (1 - MOMENTUM_204) * np.ones(2) + MOMENTUM_204 * X_204.var(axis=0, ddof=1)   # unbiased
EVAL_204 = GAMMA_204 * (np.array([4.0, 12.0]) - RUN_MEAN_204) / np.sqrt(RUN_VAR_204) + BETA_204

# Ex 20.6: the parameters of VGG16
VGG_BLOCKS = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]
VGG_CONV = []
_c_in = 3
for _block in VGG_BLOCKS:
    for _c_out in _block:
        VGG_CONV.append((9 * _c_in + 1) * _c_out)
        _c_in = _c_out
VGG_FC = [25088 * 4096 + 4096, 4096 * 4096 + 4096, 4096 * 1000 + 1000]
VGG_TOTAL = sum(VGG_CONV) + sum(VGG_FC)

# Ex 20.7: layer norm of one example, eps = 0
X_207 = np.array([1.0, 2.0, 3.0, 6.0])
LN_207 = (X_207 - X_207.mean()) / np.sqrt(X_207.var())

# Ex 20.9: training VGG16 on ImageNet
TRAIN_OPS_209 = 3 * 15.5e9                       # forward + backward, per image
TOTAL_OPS_209 = 1.28e6 * 74 * TRAIN_OPS_209'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
SHAPE = 'a shape as a string, e.g. "(3, 5)"'

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("20.Q1", "Deep learning et machine learning : qui contient qui ?", [
        ("a", TF, "True",
         r'''mistakes={"relis le §20.2 de la fiche : quel est l'usage le plus courant ?": False}'''),
        ("b", TF, "False",
         r'''mistakes={"le livre signale que certains auteurs réservent « machine learning » aux méthodes sans réseaux": True}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"un MLP à deux couches cachées est une pile de couches": "A",
          "un réseau de convolution est un réseau de neurones à plusieurs couches": "B",
          "un réseau récurrent est un réseau de neurones": "D"}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"la hiérarchie ne porte pas sur le nombre de neurones, mais sur ce que chaque couche reçoit": "A",
          "rien dans le réseau ne trie les classes par fréquence": "C",
          "les poids ne sont pas triés : qu'est-ce que chaque couche reçoit en entrée ?": "D"}'''),
    ]),
    Paper("20.Q2", "Tenseurs : nombre de dimensions, taille, forme", [
        ("a", "a whole number", "4",
         r'''mistakes={"compte les nombres de la forme, pas les dimensions d'une image": 3,
          "le nombre de dimensions est la longueur de la forme": 32}'''),
        ("b", "a whole number", "32 * 3 * 28 * 28",
         r'''mistakes={"le nombre d'éléments est le PRODUIT des tailles": 32 + 3 + 28 + 28,
          "multiplie aussi par le nombre de canaux, 3": 32 * 28 * 28,
          "multiplie aussi par la taille du mini-batch, 32": 3 * 28 * 28}'''),
        ("c", SHAPE, '"(3, 28, 28)"',
         r'''mistakes={"x[0] retire seulement la première dimension, celle des exemples": "(28, 28)",
          "x[0] retire la première dimension : il reste trois nombres": "(32, 3, 28)",
          "x[0] garde les canaux, puis la hauteur et la largeur": "(28, 28, 3)",
          "la bonne forme, mais écris-la comme un tuple, sans torch.Size : (3, 28, 28)": "torch.Size([3, 28, 28])",
          "la bonne forme, mais écris-la entre parenthèses, comme un tuple : (3, 28, 28)": "[3, 28, 28]",
          "c'est le même tuple, mais écris-le sans la virgule finale, comme PyTorch l'affiche : (3, 28, 28)": "(3, 28, 28,)"}'''),
        ("d", TF, "True",
         r'''mistakes={"relis la définition du livre (§20.2.1) : un tenseur est-il « troué » ?": False}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"float64 est le type par défaut de NumPy, pas de PyTorch": "A",
          "1.0 est un réel : un tenseur d'entiers viendrait de torch.tensor([1, 2])": "C",
          "float16 n'est jamais un type par défaut": "D"}'''),
    ]),
    Paper("20.Q3", "Couche d'entrée, couche de sortie : laquelle contient des neurones ?", [
        ("a", TF, "False",
         r'''mistakes={"relis le §20.3.1 : la couche d'entrée calcule-t-elle quelque chose ?": True}'''),
        ("b", TF, "True",
         r'''mistakes={"qui produit les scores de sortie, sinon des neurones ?": False}'''),
        ("c", "a whole number", "2",
         r'''mistakes={"la couche d'entrée et la couche de sortie ne sont pas cachées": 4,
          "seules les couches entre l'entrée et la sortie sont cachées : la sortie n'en est pas une": 3,
          "784 → 256 → 128 → 10 : combien de couches entre l'entrée et la sortie ?": 1}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"le livre dessine les entrées en bas : où est le haut ?": "A",
          "« plus haute » parle de position dans la pile, pas de taille": "C",
          "toutes les couches sont entraînées en même temps": "D"}'''),
    ]),
    Paper("20.Q4", "Couche dense et couche d'activation séparée", [
        ("a", "a whole number", "4 * 3",
         r'''mistakes={"sans les biais : un poids par connexion": 15, "chaque neurone reçoit les 4 sorties": 7}'''),
        ("b", "a whole number", "4 * 3 + 3",
         r'''mistakes={"un biais par neurone, pas par entrée": 16, "ajoute les 3 biais": 12}'''),
        ("c", TF, "True",
         r'''mistakes={"relis le §20.4.2 : une activation dans la couche ou juste après, est-ce différent ?": False}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"CrossEntropyLoss applique elle-même log_softmax : une softmax avant serait de trop": "A",
          "la loss a besoin de tous les scores, pas seulement de la classe gagnante": "C",
          "la sigmoïde va avec BCEWithLogitsLoss, et encore : celle-ci la fait elle-même": "D"}'''),
    ]),
    Paper("20.Q5", "Vrai ou faux : le dropout tire ses neurones une fois par epoch", [
        ("a", TF, "False",
         r'''mistakes={"c'est ce que dit le livre ; relis l'encadré « Le livre, corrigé » du §20.4.3": True}'''),
        ("b", "a whole number", "500",
         r'''mistakes={"un nouveau masque à chaque passe avant, donc à chaque mini-batch": 1}'''),
        ("c", "a number", "1 / (1 - 0.2)",
         r'''decimals=2, mistakes={"on divise par 1 − p, on ne multiplie pas": 0.8, "on divise par 1 − p, pas par p": 5.0}'''),
        ("d", LETTER, '"C"',
         r'''mistakes={"en évaluation, le dropout ne coupe plus rien": "A",
          "c'est la règle de l'article d'origine, appliquée aux poids ; le dropout inversé n'a rien à corriger au test": "B",
          "en évaluation, aucun neurone n'est coupé": "D"}'''),
    ]),
    Paper("20.Q6", "Vrai ou faux : la batchnorm n'a aucun paramètre", [
        ("a", TF, "False",
         r'''mistakes={"c'est ce que dit le livre ; relis l'encadré « Le livre, corrigé » du §20.4.4": True}'''),
        ("b", "a whole number", "2 * 64",
         r'''mistakes={"γ ET β : deux paramètres par feature": 64,
          "les statistiques glissantes ne sont pas apprises par la rétropropagation": 4 * 64}'''),
        ("c", "a whole number", "2 * 64",
         r'''mistakes={"une moyenne ET une variance par feature": 64,
          "on compte les valeurs stockées : une moyenne et une variance pour chacune des 64 features": 2}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"en évaluation, une prédiction ne doit pas dépendre des autres exemples du mini-batch": "A",
          "la batchnorm normalise aussi en évaluation, avec d'autres statistiques": "C",
          "les statistiques glissantes sont mises à jour à chaque mini-batch d'entraînement": "D"}'''),
    ]),
    Paper("20.Q7", "Convolution et pooling en un coup d'œil", [
        ("a", "a whole number", "2 * (3 * 3 * 1 + 1)",
         r'''mistakes={"ajoute un biais par filtre": 18, "un biais par filtre, pas un par poids": 36,
          "deux filtres : multiplie par 2": 10}'''),
        ("b", "a whole number", "512 // 2", r'''mistakes={"des blocs de 2 × 2 : on divise la largeur par 2": 128}'''),
        ("c", "a whole number", "30 // 3", r'''mistakes={"des blocs de 3 × 3 disjoints : on divise par 3": 28}'''),
        ("d", TF, "True",
         r'''mistakes={"un filtre 3 × 3 déborde d'un pixel de chaque côté : que comble un padding de 1 ?": False}'''),
    ]),
    Paper("20.Q8", "Récurrence n'est pas récursion", [
        ("a", TF, "False",
         r'''mistakes={"relis le §20.4.7 : le livre insiste sur la différence": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"une photo se classe d'un coup, sans ce qui précède": "A",
          "les features d'une maison arrivent toutes ensemble": "C",
          "une image se traite d'un coup": "D"}'''),
        ("c", TF, "True",
         r'''mistakes={"une couche dense garde-t-elle un état d'une entrée à l'autre ?": False}'''),
    ]),
    Paper("20.Q9", "Aplatir, remodeler, rogner, rembourrer, suréchantillonner, bruiter", [
        ("a", "a whole number", "8 + 2 * 1",
         r'''mistakes={"une bordure de 1 de CHAQUE côté": 9}'''),
        ("b", "a whole number", "7 * 2", r'''mistakes={"le suréchantillonnage agrandit": 3}'''),
        ("c", "a whole number", "(28 - 24) // 2",
         r'''mistakes={"28 − 24 = 4 pixels en tout, répartis sur deux côtés": 4}'''),
        ("d", LETTER, '"D"',
         r'''mistakes={"aplatir change la forme, sans rien apprendre": "A",
          "le suréchantillonnage par défaut répète les valeurs, sans rien apprendre": "B",
          "le zero-padding ajoute des 0 fixes": "C"}'''),
        ("e", TF, "False",
         r'''mistakes={"comme le dropout, une couche de bruit est une régularisation : quand doit-elle agir ?": True}'''),
    ]),
    Paper("20.Q10", "VGG16 : ce qu'il reconnaît et ce qu'il ignore", [
        ("a", "a whole number", "13", r'''mistakes={"16 couches de calcul en tout, dont 3 denses": 16}'''),
        ("b", "a whole number", "3", r'''mistakes={"le pooling et l'aplatissement ne sont pas des couches denses": 5}'''),
        ("c", "a whole number", "1000", r'''mistakes={"relis le §20.6 : le nombre de classes d'ImageNet utilisées au concours": 100}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"relis l'encadré « Le livre, corrigé » du §20.6 : quelle erreur le concours mesure-t-il ?": "A",
          "le concours mesure l'erreur sur des images qui ne servent pas à l'entraînement": "C",
          "c'est une accuracy, pas une précision": "D"}'''),
        ("e", TF, "False",
         r'''mistakes={"sa couche de sortie a 1 000 neurones : combien de noms peut-il donner ?": True}'''),
    ]),
    Paper("20.Q11", "Construire un réseau profond : par où commencer ?", [
        ("a", LETTER, '"B"',
         r'''mistakes={"avant l'architecture, il faut connaître les données": "A",
          "on règle le learning rate une fois le réseau choisi": "C",
          "un grand réseau sur des données qu'on n'a pas regardées fait perdre du temps": "D"}'''),
        ("b", LETTER, '"C"',
         r'''mistakes={"relis le §20.7 : le livre nomme un hyperparamètre de l'optimiseur": "A",
          "la taille des mini-batches compte, mais le livre en nomme un autre": "B",
          "une graine ne s'optimise pas": "D"}'''),
        ("c", TF, "False", r'''mistakes={"relis le §20.7 : le livre parle d'expérience et d'essais": True}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"un MLP est fait de couches denses": "A", "un RNN est fait de couches récurrentes": "C",
          "un GAN est une association de deux réseaux (ch. 27)": "D"}'''),
    ]),
    Paper("20.Q12", "Explication complète ou explication satisfaisante ?", [
        ("a", TF, "False",
         r'''mistakes={"relis le §20.8.1 : l'exemple des chaussettes bleues est complet, est-il satisfaisant ?": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"le volume des données n'empêche pas de vérifier": "A",
          "le nombre de couches gêne l'explication, pas la vérification": "C",
          "rien n'oblige la banque à jeter les dossiers : que lui manque-t-il pour les refusés ?": "D"}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"une boîte blanche est au contraire un modèle qu'on peut lire": "B",
          "une régression linéaire s'explique très bien par ses coefficients": "C",
          "la dérive est un changement du monde après l'entraînement": "D"}'''),
        ("d", TF, "False",
         r'''mistakes={"relis le §20.8.1 : le monde a-t-il pu changer en trois ans ?": True}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("20.R1", "Que fait weight_decay dans AdamW ?", [
        ("a", "a number", "2 * (1 - 0.01 * 0.1)",
         r'''decimals=3, mistakes={"le facteur est 1 − η λ, avec η = 0,01 : multiplie λ par η": 1.8,
          "le gradient est nul, mais le weight decay agit quand même": 2.0}'''),
        ("b", TF, "False",
         r'''mistakes={"c'est la différence entre Adam(weight_decay=…) et AdamW : relis la fiche du ch. 19": True}'''),
        ("c", "a number", "0.01",
         r'''decimals=3, mistakes={"c'est la valeur par défaut d'Adam (0), pas d'AdamW": 0.0}'''),
    ]),
    Paper("20.R2", "Pourquoi calcule-t-on la loss à partir des logits ?", [
        ("a", TF, "False",
         r'''mistakes={"CrossEntropyLoss fait elle-même log_softmax : quelle entrée attend-elle ?": True}'''),
        ("b", "a whole number", "round(-1000 - math.log1p(math.exp(-1000)))",
         r'''mistakes={"log_softmax(z)_2 = z_2 − log(e^{z_1} + e^{z_2}) : le logarithme d'une probabilité est négatif": 1000,
          "log_softmax(z)_2 = z_2 − log(e^{z_1} + e^{z_2}) ≈ 0 − 1000": 0}'''),
        ("c", "a whole number", "1", ""),
        ("d", LETTER, '"B"',
         r'''mistakes={"le but n'est pas d'écrire moins de code": "A", "la softmax a une dérivée (ch. 17)": "C",
          "torch.softmax existe : pourquoi ne pas l'appliquer avant la loss ?": "D"}'''),
    ]),
    Paper("20.R3", "Limiter l'overfitting d'un arbre, et l'équivalent pour un réseau", [
        ("a", LETTER, '"A"',
         r'''mistakes={"la graine change l'arbre, pas sa complexité": "B", "n_jobs change la vitesse": "C",
          "le critère change le choix des seuils, pas la taille de l'arbre": "D"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"plus de neurones, c'est plus de capacité, donc plus d'overfitting": "A",
          "plus d'epochs laisse plus de temps pour apprendre par cœur": "C",
          "sans validation, on ne voit plus l'overfitting": "D"}'''),
        ("c", TF, "True",
         r'''mistakes={"l'early stopping arrête l'entraînement avant que l'erreur de validation ne remonte : que fait-il à l'overfitting ?": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("20.1", "Suivre la forme d'un tenseur couche après couche", [
        ("a", "a whole number", "10 * 3 * 5", r'''mistakes={"le nombre d'éléments est le produit des tailles": 18}'''),
        ("b", SHAPE, '"(10, 15)"',
         r'''mistakes={"on garde la première dimension, 10 : 150 éléments à répartir": "(150,)",
          "garde la première dimension et multiplie les deux autres": "(10, 8)",
          "le remodelage garde la première dimension, 10, en premier": "(15, 10)"}'''),
        ("c", SHAPE, '"(32, 784)"',
         r'''mistakes={"nn.Flatten garde la dimension des exemples": "(25088,)",
          "nn.Flatten met à plat tout ce qui suit la dimension des exemples, canal compris": "(32, 1, 784)",
          "1 × 28 × 28 = 784": "(32, 28)"}'''),
        ("d", SHAPE, '"(32, 128)"',
         r'''mistakes={"la couche dense remplace les 784 features par 128 sorties": "(32, 784)",
          "la dimension des exemples reste la première": "(128, 32)"}'''),
        ("e", "a whole number", "784 * 128 + 128",
         r'''mistakes={"ajoute les 128 biais": 784 * 128, "un biais par sortie, pas par entrée": 784 * 128 + 784}'''),
        ("f", SHAPE, '"(32, 3072)"',
         r'''mistakes={"3 × 32 × 32 = 3 072 : compte aussi les trois canaux": "(32, 1024)",
          "nn.Flatten garde la dimension des exemples": "(98304,)"}'''),
        ("g", SHAPE, '"(25088,)"',
         r'''mistakes={"torch.flatten met TOUT à plat, la dimension des exemples comprise": "(1, 25088)",
          "une forme à une dimension s'écrit avec une virgule : (7,)": "(25088)",
          "512 × 7 × 7 = 25 088": "(512, 49)"}'''),
        ("h", SHAPE, '"(1, 25088)"',
         r'''mistakes={"nn.Flatten garde la première dimension (start_dim=1)": "(25088,)",
          "512 × 7 × 7 = 25 088": "(1, 512, 49)"}'''),
    ]),
    Paper("20.2", "Dimensionner la couche de sortie et choisir la loss", [
        ("a", "a whole number", "1", r'''mistakes={"un prix : une seule valeur à prédire": 2}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"un prix n'est pas une probabilité : c'est une régression": "B",
          "un prix n'est pas une classe": "C"}'''),
        ("c", "a whole number", "26", r'''mistakes={"un score par classe": 1}'''),
        ("d", LETTER, '"C"',
         r'''mistakes={"une régression ne convient pas pour 26 classes exclusives": "A",
          "les 26 classes sont exclusives : une seule lettre par image": "B"}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"une probabilité de spam, pas une valeur réelle quelconque": "A",
          "CrossEntropyLoss attend un score par classe, donc au moins deux sorties": "C"}'''),
        ("f", "a whole number", "5",
         r'''mistakes={"une sortie par label, chacun présent ou non": 1,
          "les labels ne sont pas exclusifs : pas besoin d'une sortie par combinaison": 2 ** 5}'''),
        ("g", LETTER, '"B"',
         r'''mistakes={"chaque label est un oui ou non : une probabilité par label": "A",
          "CrossEntropyLoss suppose qu'une seule classe est vraie : ici, plusieurs labels peuvent l'être": "C"}'''),
        ("h", LETTER, '"C"',
         r'''mistakes={"deux scores, une classe vraie parmi deux : c'est une classification à K = 2 classes": "B",
          "c'est une classification": "A"}'''),
    ]),
    Paper("20.3", "Dropout inversé à la main", [
        ("a", "a whole number", "round(0.2 * 100)", r'''mistakes={"20 % de 100 neurones sont coupés : combien en garde-t-on ?": 80}'''),
        ("b", "a whole number", "int((U_203 >= P_203).sum())",
         r'''mistakes={"on GARDE un neurone quand u ≥ p, on le coupe quand u < p": 2}'''),
        ("c", "a number, 4 decimals", "1 / (1 - P_203)",
         r'''decimals=4, mistakes={"on divise par 1 − p, on ne multiplie pas": 0.6, "on divise par 1 − p, pas par p": 2.5}'''),
        ("d", "a number, 4 decimals", "OUT_203[1]",
         r'''decimals=4, mistakes={"multiplie par 1/(1 − p)": -1.0, "u₂ = 0,7 ≥ 0,4 : ce neurone est gardé": 0.0,
          "on divise par 1 − p = 0,6, pas par p": -2.5}'''),
        ("e", "a number, 4 decimals", "OUT_203.sum()",
         r'''decimals=4, mistakes={"multiplie par 1/(1 − p) les neurones gardés": float(A_203[U_203 >= P_203].sum()),
          "on garde quand u ≥ p, pas quand u < p": float((A_203 * (U_203 < P_203)).sum() / (1 - P_203)),
          "seuls les neurones gardés comptent dans la somme": float(A_203.sum() / (1 - P_203))}'''),
        ("f", "a number", "A_203.sum()",
         r'''decimals=4, mistakes={"en moyenne sur les masques, le dropout inversé garde l'espérance de chaque activation": float(OUT_203.sum()),
          "chaque activation garde son espérance, sans facteur 1 − p": float((1 - P_203) * A_203.sum())}'''),
        ("g", "a number, 4 decimals", "MASK_203.sum()",
         r'''decimals=4, mistakes={"le gradient passe par le même masque, avec le même facteur 1/(1 − p)": 3.0,
          "seuls les neurones gardés transmettent le gradient": 5 / (1 - P_203)}'''),
        ("h", TF, "True",
         r'''mistakes={"en évaluation, le dropout inversé est l'identité": False}'''),
    ]),
    Paper("20.4", "Batchnorm à la main sur un mini-batch de quatre exemples", [
        ("a", "a number", "MU_204[0]", r'''decimals=4, mistakes={"une moyenne par feature, donc par colonne": float(X_204[0].mean())}'''),
        ("b", "a number", "MU_204[1]", r'''decimals=4, mistakes={"une moyenne par feature, donc par colonne": float(X_204[1].mean())}'''),
        ("c", "a number", "VAR_204[0]",
         r'''decimals=4, mistakes={"pour normaliser, la variance biaisée : divise par 4, pas par 3": float(X_204[:, 0].var(ddof=1)),
          "la variance, pas l'écart-type": float(X_204[:, 0].std())}'''),
        ("d", "a number", "VAR_204[1]",
         r'''decimals=4, mistakes={"pour normaliser, la variance biaisée : divise par 4, pas par 3": float(X_204[:, 1].var(ddof=1)),
          "la variance, pas l'écart-type": float(X_204[:, 1].std())}'''),
        ("e", "a number, 4 decimals", "XHAT_204[0, 0]",
         r'''decimals=4, mistakes={"divise par l'écart-type, la racine de la variance": -3 / 5,
          "soustrais d'abord la moyenne de la feature, 4": float((1 - 0) / np.sqrt(5)),
          "avec la variance non biaisée, ce n'est plus la normalisation de PyTorch": float(-3 / X_204[:, 0].std(ddof=1))}'''),
        ("f", "a number, 4 decimals", "Y_204[1, 1]",
         r'''decimals=4, mistakes={"y = γ x̂ + β : n'oublie pas β₂ = 1": float(Y_204[1, 1] - 1),
          "c'est x̂, avant l'échelle γ et le décalage β": float(XHAT_204[1, 1]),
          "γ₂ = 0,5 et β₂ = 1, pas l'inverse": float(XHAT_204[1, 1] + 0.5)}'''),
        ("g", TF, "True",
         r'''mistakes={"la moyenne des x̂ d'une feature est nulle : que vaut celle de γ x̂ + β ?": False}'''),
        ("h", "a number", "RUN_MEAN_204[0]",
         r'''decimals=4, mistakes={"PyTorch : (1 − ρ) × ancienne + ρ × nouvelle, avec ρ = 0,1": 3.6,
          "la statistique glissante ne prend que 10 % de la nouvelle moyenne": 4.0}'''),
        ("i", "a number", "RUN_MEAN_204[1]",
         r'''decimals=4, mistakes={"PyTorch : (1 − ρ) × ancienne + ρ × nouvelle, avec ρ = 0,1": 9.0,
          "la statistique glissante ne prend que 10 % de la nouvelle moyenne": 10.0}'''),
        ("j", "a number, 4 decimals", "RUN_VAR_204[0]",
         r'''decimals=4, mistakes={"la variance glissante utilise la variance NON biaisée, 20/3": 0.9 + 0.1 * 5,
          "partir de σ²_run = 1, pas de 0": 0.1 * 20 / 3,
          "(1 − ρ) × ancienne + ρ × nouvelle, avec ρ = 0,1, pas l'inverse": 0.1 + 0.9 * 20 / 3}'''),
        ("k", "a number, 3 decimals", "EVAL_204[0]",
         r'''decimals=3, mistakes={"en évaluation, les statistiques GLISSANTES, pas celles du mini-batch": 0.0,
          "n'oublie pas γ₁ = 2": float(EVAL_204[0] / 2),
          "la racine de la variance glissante, pas la variance": float(2 * 3.6 / RUN_VAR_204[0])}'''),
    ]),
    Paper("20.6", "Compter les 138 millions de paramètres de VGG16", [
        ("a", "a whole number", "224 // 2 ** 5",
         r'''mistakes={"le zero-padding garde la taille ; seuls les 5 poolings la divisent par 2": 224,
          "5 poolings : 224 / 2⁵": 14}'''),
        ("b", "a whole number", "VGG_CONV[0]",
         r'''mistakes={"(3² × 3 + 1) × 64 : n'oublie pas le biais de chaque filtre": 27 * 64,
          "chaque filtre a 3 × 3 poids PAR canal d'entrée": 10 * 64}'''),
        ("c", "a whole number", "sum(VGG_CONV)",
         r'''mistakes={"ajoute les biais : un par filtre": sum(VGG_CONV) - sum(sum(b) for b in VGG_BLOCKS)}'''),
        ("d", "a whole number", "VGG_FC[0]",
         r'''mistakes={"ajoute les 4 096 biais": 25088 * 4096, "l'entrée est l'aplatissement de 512 × 7 × 7 = 25 088 valeurs": 512 * 4096 + 4096}'''),
        ("e", "a whole number", "VGG_TOTAL",
         r'''mistakes={"ajoute les biais des couches denses": VGG_TOTAL - 4096 - 4096 - 1000}'''),
        ("f", "a number, 1 decimal", "100 * sum(VGG_FC) / VGG_TOTAL",
         r'''decimals=1, mistakes={"compte les trois couches denses, pas seulement la première": 100 * VGG_FC[0] / VGG_TOTAL,
          "la part des denses, pas celle des convolutions": 100 * sum(VGG_CONV) / VGG_TOTAL}'''),
        ("g", "a number of MB, 1 decimal", "4 * VGG_TOTAL / 1e6",
         r'''decimals=1, mistakes={"4 octets par nombre float32": VGG_TOTAL / 1e6,
          "Mo de 10⁶ octets, pas Mio de 2²⁰": 4 * VGG_TOTAL / 2 ** 20}'''),
        ("h", LETTER, '"B"',
         r'''mistakes={"compare ta réponse de g) à 527,8 : quelle unité donne le plus petit nombre ?": "A",
          "en mégabits, le nombre serait 8 fois plus grand": "C",
          "les convolutions ne font qu'environ 59 Mo : le fichier contient tout": "D"}'''),
    ]),
    Paper("20.7", "Connexion résiduelle et LayerNorm à la main", [
        ("a", "a number", "X_207.mean()", "decimals=4"),
        ("b", "a number", "X_207.var()",
         r'''decimals=4, mistakes={"la layer norm utilise la variance biaisée : divise par 4": float(X_207.var(ddof=1)),
          "la variance, pas l'écart-type": float(X_207.std())}'''),
        ("c", "a number, 4 decimals", "LN_207[3]",
         r'''decimals=4, mistakes={"divise par l'écart-type, la racine de la variance": 3 / 3.5,
          "avec la variance non biaisée, ce n'est plus la layer norm de PyTorch": float(3 / X_207.std(ddof=1))}'''),
        ("d", "a number", "float((LN_207 ** 2).sum())",
         r'''decimals=4, mistakes={"la moyenne des carrés vaut 1 : et leur somme ?": 1.0, "c'est la somme des valeurs qui est nulle": 0.0}'''),
        ("e", TF, "True",
         r'''mistakes={"la layer norm calcule ses statistiques sur les features de chaque exemple, seul": False}'''),
        ("f", "a number of 1e-9, 4 decimals", "0.5 ** 30 / 1e-9",
         r'''decimals=4, mistakes={"en multiple de 10⁻⁹ : écris le coefficient devant 10⁻⁹, pas la valeur": 0.5 ** 30,
          "c'est un produit de 30 facteurs 0,5": 0.5 * 30 / 1e-9}'''),
        ("g", "a whole number", "math.ceil(math.log(0.01) / math.log(0.9))",
         r'''mistakes={"0,9⁴³ vaut encore un peu plus de 0,01 : il en faut une de plus": 43,
          "cherche n tel que 0,9ⁿ < 0,01, avec des logarithmes": 10}'''),
        ("h", "a number, 4 decimals", "1.01 ** 30",
         r'''decimals=4, mistakes={"la dérivée d'un bloc résiduel est 1 + F′, pas F′": 0.01 ** 30,
          "c'est un produit de 30 facteurs 1,01, pas une somme": 1 + 30 * 0.01}'''),
    ]),
    Paper("20.9", "Fermi : entraîner VGG16 sur ImageNet chez soi ?", [
        ("a", "a number of billions, 1 decimal", "TRAIN_OPS_209 / 1e9",
         r'''decimals=1, mistakes={"passe avant ET passe arrière : 3 fois le coût de la passe avant": 15.5,
          "3 fois plus en tout, pas 3 fois plus en plus": 4 * 15.5}'''),
        ("b", "a number of 1e18, 2 decimals", "TOTAL_OPS_209 / 1e18",
         r'''decimals=2, mistakes={"le coût d'entraînement est 3 fois celui de la passe avant": 1.28e6 * 74 * 15.5e9 / 1e18,
          "74 epochs sur 1,28 million d'images : multiplie par les deux": 1.28e6 * TRAIN_OPS_209 / 1e18}'''),
        ("c", "a number of days, 1 decimal", "TOTAL_OPS_209 / 1e13 / 86400",
         r'''decimals=1, mistakes={"une journée a 86 400 secondes": TOTAL_OPS_209 / 1e13 / 3600,
          "le résultat en jours, pas en secondes": TOTAL_OPS_209 / 1e13}'''),
        ("d", "a whole number of days", "round(TOTAL_OPS_209 / 1e11 / 86400)",
         r'''fractional="un nombre de jours, arrondi à l'entier",
          mistakes={"une journée a 86 400 secondes": round(TOTAL_OPS_209 / 1e11 / 3600)}'''),
        ("e", "a number of GB, 2 decimals", "4 * 4 * 138_357_544 / 1e9",
         r'''decimals=2, mistakes={"paramètres, gradients ET les deux états d'Adam : 4 nombres par paramètre": 4 * 138_357_544 / 1e9,
          "4 nombres par paramètre, de 4 octets chacun": 3 * 4 * 138_357_544 / 1e9,
          "Go de 10⁹ octets, pas Gio": 4 * 4 * 138_357_544 / 2 ** 30}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 20.1 à 20.4, 20.6, 20.7 et 🧮 20.9)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; une "
               "forme : entre guillemets (`\"(3, 5)\"`, ou `\"(7,)\"` pour une seule dimension). Arrondis comme "
               "l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. "
               "Les questions « dans ta copie », les preuves ∂ 20.5 et 20.8, la réflexion (🗣️ 20.10, ⚖️ 20.11, "
               "📄 20.12) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 20.13 to 20.31): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "20.13 à 20.17", "tes premiers tenseurs PyTorch, prédire les formes des couches utilitaires, l'autograd sur "
                           "le réseau minuscule du ch. 18, puis ta librairie regularization.py : dropout, layer norm, "
                           "pénalité L2 et early stopping"),
    ("B", "20.18 à 20.22", "ta classe nn.Module, Dataset et DataLoader, le zoo des couches, le plan de VGG16, la "
                           "batchnorm de ta librairie"),
    ("C", "20.23 à 20.27", "la boucle d'entraînement complète sur MNIST, une chasse aux bugs, des courbes "
                           "d'entraînement et de validation, un modèle ImageNet face à l'inconnu, la sauvegarde"),
    ("D", "20.28 à 20.31", "dropout, batchnorm, AdamW et early stopping sur Fashion-MNIST, une ablation, un réseau "
                           "de 30 couches, le défi des 90 %"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 20.Q1 à 20.Q12, 20.R1 à 20.R3, 20.1 à 20.4, 20.6, 20.7, 20.9 | vérifier tes réponses courtes | "
            "🧠 🔁 ✏️ 🧮 | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# 20 · Deep learning et premiers pas en PyTorch — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou les tests de ta librairie "
               "`mylearn`). « Exécuter tout » va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait "
               "affiche ⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? "
               "`04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch20_deep_learning/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 20`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 20 · Deep learning et premiers pas en PyTorch — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Prévoir la forme d'un tenseur couche après couche, et dimensionner la sortie d'un réseau.\n"
               "- Calculer à la main un dropout inversé, une batchnorm (entraînement et évaluation) et une layer "
               "norm, puis les programmer dans `mylearn.nn.regularization` et les retrouver dans PyTorch.\n"
               "- Écrire un `nn.Module`, un `DataLoader` et une boucle d'entraînement PyTorch, et mener une ablation "
               "de régularisation.\n\n"
               "**Rappel express.** Dropout inversé : $\\tilde a = a\\,m/(1 - p)$, $m \\sim \\mathrm{Bernoulli}(1 - p)$, "
               "un nouveau masque à chaque passe avant, l'identité en évaluation. Batchnorm : "
               "$\\hat x = (x - \\mu_B)/\\sqrt{\\sigma_B^2 + \\epsilon}$ (variance biaisée), $y = \\gamma\\hat x + \\beta$ ; "
               "statistiques glissantes $\\mu^{\\text{run}} \\leftarrow (1 - \\rho)\\,\\mu^{\\text{run}} + \\rho\\,\\mu_B$ avec "
               "$\\rho = 0{,}1$ et la variance non biaisée. Layer norm : la même formule, sur les features de chaque "
               "exemple. Pénalité L2 : $\\frac{\\lambda}{2}\\sum\\lVert\\mathbf{W}\\rVert^2$, de gradient $\\lambda\\mathbf{W}$. "
               "Boucle PyTorch : `model.train()`, puis pour chaque mini-batch `logits = model(X)`, "
               "`loss = loss_fn(logits, y)`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` ; "
               "`model.eval()` et `torch.no_grad()` pour mesurer.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 20.13 à 20.31 (parties A à D, ta librairie `regularization.py` et PyTorch) "
             "seront ajoutés à la prochaine session de génération ; relance alors `python tools/start_chapter.py 20` "
             "pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu prévoir la forme d'un tenseur après une couche dense, un aplatissement ou un pooling, et "
               "choisir la sortie et la loss d'un réseau ?\n"
               "2. Sais-tu calculer à la main un dropout inversé et une batchnorm, et dire ce que changent "
               "`model.train()` et `model.eval()` ?\n"
               "3. Sais-tu écrire une boucle d'entraînement PyTorch complète et expliquer chacune de ses lignes ?\n\n"
               "**Pour aller plus loin** : le tutoriel « Learn the Basics » de PyTorch et *Dive into Deep "
               "Learning*, cités dans la fiche. La suite : le ch. 21, les réseaux de convolution, en PyTorch." + later)]


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
