#!/usr/bin/env python
"""Build the two notebooks of chapter 24 from a single source (used by Claude).

    python tools/chapters/build_ch24.py
    python tools/run_all_notebooks.py chapitres/ch24_pytorch_2/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch24_pytorch_2/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
24.1 to 24.5 and of the 🧮 estimate 24.7 (02_exercices.md). The derivation ∂ 24.6, the reflection (🗣️ 24.8,
⚖️ 24.9, 📄 24.10) and the interview are corrected with 05_solutions.md.
Parts A to E (exercises 24.11 to 24.31) come at the second generation session: improving an MLP on
Fashion-MNIST (batch size, capacity, dropout and max-norm, the experiment log), a PyTorch network dressed as a
scikit-learn estimator (cross-validation, grid and random search), CNNs on CIFAR-10 (a buggy translation,
simple to deep, batchnorm and dropout, augmentation, synthetic images), series and LSTMs (sums of sines,
window, depth and data, outputs and carried state on the sunspots), and beyond nn.Sequential (branches and
residual blocks, mixed precision and torch.compile, text generation, the frugal CIFAR-10 challenge). The
chapter has no mylearn module: the notebook reuses mylearn.sequences (ch. 22) and
mylearn.preprocessing.MinMaxScaler (ch. 12).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402,F401

CHAPTER = "24"
FOLDER = "chapitres/ch24_pytorch_2"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def conv_params(c_in, c_out, k):
    """Parameters of nn.Conv2d(c_in, c_out, k): weights and biases."""
    return c_out * (c_in * k * k + 1)


def dense(n_in, n_out):
    """Parameters of nn.Linear(n_in, n_out): weights and biases."""
    return (n_in + 1) * n_out


def conv_mac(h_out, w_out, c_out, c_in, k):
    """Multiply-accumulates of one convolution, for one image."""
    return h_out * w_out * c_out * c_in * k * k


def lstm_layer_params(d_in, h):
    """Parameters of one PyTorch LSTM layer: 4 blocks of weights, two bias vectors."""
    return 4 * h * (d_in + h) + 2 * 4 * h


# Recall 24.R3: logits of three tokens
Z_R3 = np.array([2.0, 1.0, 0.0])

# Ex 24.2: the book's pooling CNN on MNIST (fig. 24.31), in PyTorch
P_CONV1, P_CONV2 = conv_params(1, 30, 5), conv_params(30, 16, 3)
N_FLAT = 16 * 7 * 7
P_DENSE = dense(N_FLAT, 128) + dense(128, 64) + dense(64, 10)
P_TOTAL = P_CONV1 + P_CONV2 + P_DENSE

# Ex 24.3: multiply-accumulates of one image, pooling version and stride version
MAC_POOL = conv_mac(28, 28, 30, 1, 5) + conv_mac(14, 14, 16, 30, 3)
MAC_STRIDE = conv_mac(14, 14, 30, 1, 5) + conv_mac(7, 7, 16, 30, 3)
MAC_DENSE = 784 * 128 + 128 * 64 + 64 * 10

# Ex 24.7: a small CNN, 25 million MAC per image, 20 epochs on 50 000 images
OPS_247 = 25e6 * 2 * 3 * 50_000 * 20
CPU_MIN_247 = OPS_247 / 50e9 / 60
'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "F"'
LETTERS = 'the letters, e.g. "AF"'
SHAPE = 'a shape as a string, e.g. "(3, 5, 1)"'
WHOLE = "a whole number"
NUMBER = "a number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("24.Q1", "Compter les hyperparamètres et n'en changer qu'un", [
        ("a", LETTERS, '"BCE"',
         r'''mistakes={"les poids de la première couche sont calculés par l'entraînement, pas choisis": "ABCE",
          "les biais sont appris comme les poids": "BCDE",
          "l'état d'Adam est calculé pendant l'entraînement, pas choisi": "BCEF",
          "le taux de dropout se choisit avant l'entraînement": "BC",
          "le learning rate se choisit avant l'entraînement": "BE"}'''),
        ("b", WHOLE, "3 ** 6",
         r'''mistakes={"chaque hyperparamètre multiplie le nombre de combinaisons, il ne l'additionne pas": 18,
          "3 valeurs pour chacun des 6 : 3 puissance 6, pas 6 puissance 3": 6 ** 3}'''),
        ("c", TF, "False", r'''mistakes={"garder la valeur par défaut, est-ce un choix ou non ?": True}'''),
        ("d", TF, "False",
         r'''mistakes={"si le meilleur réglage de A dépend de B, régler A une seule fois suffit-il ?": True}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"les objectifs se contredisent souvent : on ne peut pas tous les servir à la fois": "A",
          "le livre cherche un réglage assez bon, pas le meilleur exact": "B",
          "changer plusieurs choses à la fois empêche de savoir laquelle a agi": "D"}'''),
    ]),
    Paper("24.Q2", "Taille de batch : bruit, sur-apprentissage et temps", [
        ("a", WHOLE, "math.ceil(50_000 / 64)",
         r'''fractional="un nombre de mises à jour est entier : le dernier mini-batch, incomplet, compte aussi",
          mistakes={"arrondis au-dessus : le dernier mini-batch, incomplet, compte aussi": 50_000 // 64,
          "une mise à jour par mini-batch : divise N par la taille du mini-batch": 50_000}'''),
        ("b", WHOLE, "math.ceil(50_000 / 256)",
         r'''fractional="un nombre de mises à jour est entier : le dernier mini-batch, incomplet, compte aussi",
          mistakes={"arrondis au-dessus : le dernier mini-batch, incomplet, compte aussi": 50_000 // 256,
          "une mise à jour par mini-batch : divise N par la taille du mini-batch": 50_000}'''),
        ("c", NUMBER, "math.sqrt(400) / math.sqrt(16)",
         r'''decimals=1, mistakes={"l'écart-type varie comme 1/√B, pas comme 1/B": 25, "dans quel sens varie le bruit quand le batch rapetisse ?": 0.2}'''),
        ("d", TF, "False",
         r'''mistakes={"compare le nombre de pas par epoch et le bruit de chaque pas dans les deux cas": True}'''),
        ("e", LETTER, '"A"',
         r'''mistakes={"plus de mini-batches par epoch, c'est plus de pas, chacun avec son coût fixe": "B",
          "relis la fig. 24.5 : la durée dépend bien de la taille de batch": "C"}'''),
    ]),
    Paper("24.Q3", "Plus de neurones, moins de neurones, dropout", [
        ("a", LETTER, '"D"',
         r'''mistakes={"plus de neurones, c'est plus de capacité, pas moins": "A",
          "relis ce que montre la fig. 24.7 sur la loss de validation": "B",
          "un réseau plus grand apprend le train au moins aussi bien": "C"}'''),
        ("b", DEC1, "0.3",
         r'''decimals=1, mistakes={"p est la probabilité de mettre à zéro, pas de garder": 0.7, "une fraction entre 0 et 1, pas un pourcentage": 30}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"relis l'encadré « Le livre, corrigé » : le masque ne dure pas une epoch": "A",
          "chaque exemple du mini-batch a son propre masque": "B",
          "un masque fixe éteindrait toujours les mêmes neurones : ce serait un réseau plus petit": "D"}'''),
        ("d", DEC1, "2.0 / (1 - 0.2)",
         r'''decimals=1, mistakes={"le dropout inversé divise les valeurs gardées par 1 − p : il ne les laisse pas telles quelles": 2.0,
          "on divise par 1 − p, on ne multiplie pas": 1.6, "on divise par 1 − p, pas par p": 10.0}'''),
        ("e", TF, "False",
         r'''mistakes={"c'est le dropout d'origine qui multipliait par 1 − p en évaluation ; que fait le dropout inversé ?": True}'''),
    ]),
    Paper("24.Q4", "Valider un réseau : coût, jeu non représentatif, normalisation dans chaque fold", [
        ("a", WHOLE, "5 * 40",
         r'''mistakes={"chaque fold est un entraînement complet": 40,
          "la validation croisée seule, sans réentraînement final sur toutes les données": 6 * 40}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"2 000 radios sur 10 000, la taille n'est pas le problème : regarde leur provenance": "B",
          "rien ne fait passer les labels dans les données : regarde de quel hôpital viennent ces radios": "C",
          "regarde de quel hôpital viennent les 2 000 dernières radios": "D"}'''),
        ("c", LETTER, '"D"',
         r'''mistakes={"le pipeline ne fait rien gagner en vitesse": "A",
          "cross_val_score accepte n'importe quelles données": "B",
          "le scaler s'applique aux features, pas aux labels": "C"}'''),
        ("d", TF, "True",
         r'''mistakes={"relis ce que scikit-learn fait d'un cv entier quand l'estimateur est un classifieur (ch. 15)": False}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"relis le score du listing 24.16 et compare-le à celui du listing 24.12": "A",
          "les deux scores du livre sont presque égaux": "B"}'''),
    ]),
    Paper("24.Q5", "🕰️ KerasClassifier, grille, hasard ou zoom : chercher des hyperparamètres", [
        ("a", LETTER, '"C"',
         r'''mistakes={"KerasClassifier a été retiré de Keras et n'a jamais enveloppé de réseau PyTorch": "A",
          "MLPClassifier est le réseau de scikit-learn, pas ton nn.Module": "B",
          "un estimateur n'est qu'un objet qui suit quelques conventions": "D"}'''),
        ("b", WHOLE, "5", r'''mistakes={"c'était la valeur par défaut en 2017, dans le livre ; relis l'encadré 🕰️": 3,
          "pense à la valeur par défaut, pas à un choix courant": 10,
          "c'est un réglage par défaut de scikit-learn, pas un calcul : relis l'encadré 🕰️ de la recherche": 4}'''),
        ("c", WHOLE, "2 * 3 * 2 * 4",
         r'''mistakes={"chaque candidat est entraîné une fois par fold": 2 * 3 * 2,
          "multiplie les nombres de valeurs, ne les additionne pas": (2 + 3 + 2) * 4,
          "relis la question : on demande les entraînements de la validation croisée seulement": 2 * 3 * 2 * 4 + 1}'''),
        ("d", WHOLE, "2 * 3 * 2 * 4 + 1",
         r'''mistakes={"le réentraînement final s'ajoute une fois, sur toutes les données": 2 * 3 * 2 * 4,
          "un seul réentraînement final, pas un par candidat": 2 * 3 * 2 * 5}'''),
        ("e", LETTER, '"D"',
         r'''mistakes={"il s'agit de la finesse de la recherche, pas de la taille des images": "A",
          "le hasard est une autre méthode de recherche": "B",
          "il s'agit de la finesse de la recherche, pas du matériel": "C"}'''),
    ]),
    Paper("24.Q6", "🕰️ Couches utilitaires de Keras et leurs équivalents PyTorch", [
        ("a", LETTER, '"C"',
         r'''mistakes={"cette couche agrandit l'image : elle ne retire rien": "A",
          "elle répète des valeurs existantes ; elle n'ajoute pas de zéros": "B",
          "elle agrandit l'image ; un pooling la rétrécit": "D",
          "elle garde la forme d'une image": "E"}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"le recadrage retire des bords, il n'en ajoute pas": "B",
          "le recadrage rétrécit l'image": "C",
          "le recadrage garde les valeurs telles quelles, sans les résumer": "D",
          "le recadrage garde la forme d'une image": "E"}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"ZeroPadding2D ajoute des bords": "A",
          "ZeroPadding2D ajoute des zéros, il ne répète pas les valeurs": "C",
          "ZeroPadding2D agrandit l'image": "D",
          "ZeroPadding2D garde la forme d'une image": "E"}'''),
        ("d", LETTER, '"D"',
         r'''mistakes={"un pooling résume des blocs, il ne retire pas seulement les bords": "A",
          "un pooling rétrécit l'image": "B", "un pooling rétrécit l'image, il ne l'agrandit pas": "C",
          "un pooling garde la forme d'une image": "E"}'''),
        ("e", TF, "False", r'''mistakes={"relis quand ce bruit est actif dans le tableau du §24.4.1": True}'''),
        ("f", LETTER, '"C"',
         r'''mistakes={"relis l'encadré « Le livre, corrigé » : ce ne sont pas les poids": "A",
          "la batchnorm agit pendant la passe avant, pas sur les gradients": "B",
          "la batchnorm agit dans le réseau, sur les sorties d'une couche": "D"}'''),
    ]),
    Paper("24.Q7", "Préparer des images pour nn.Conv2d", [
        ("a", SHAPE, '"(16, 3, 32, 32)"',
         r'''mistakes={"PyTorch veut les canaux juste après l'axe des exemples": "(16, 32, 32, 3)",
          "n'oublie pas l'axe des exemples": "(3, 32, 32)"}'''),
        ("b", LETTER, '"D"',
         r'''mistakes={"reshape garde l'ordre de lecture : il mélangerait les pixels et les canaux": "A",
          "la forme est bonne, mais transpose(1, 3) échange aussi la hauteur et la largeur": "B",
          "écris la forme obtenue axe par axe : l'axe des canaux n'arrive pas en position 1": "C"}'''),
        ("c", SHAPE, '"(16, 12, 32, 32)"',
         r'''mistakes={"padding=\"same\" garde la hauteur et la largeur": "(16, 12, 28, 28)",
          "les canaux de sortie sont le nombre de filtres": "(16, 3, 32, 32)"}'''),
        ("d", TF, "False",
         r'''mistakes={"relis la première des trois différences avec Keras au §24.4.3": True}'''),
        ("e", DEC1, "51 / 127.5 - 1",
         r'''decimals=1, mistakes={"x / 127,5 − 1 : n'oublie pas de retrancher 1": 0.4, "x / 127,5 − 1, pas x / 255": 0.2}'''),
        ("f", TF, "False", r'''mistakes={"relis la deuxième des trois différences avec Keras au §24.4.3": True}'''),
    ]),
    Paper("24.Q8", "Augmentation et données synthétiques : quelles transformations sont plausibles ?", [
        ("a", LETTER, '"B"',
         r'''mistakes={"la validation doit ressembler aux données réelles : on ne l'augmente pas": "A",
          "le test doit ressembler aux données réelles": "C",
          "l'augmentation fabrique des exemples d'entraînement": "D"}'''),
        ("b", TF, "False", r'''mistakes={"quand le Dataset tire-t-il ses transformations ?": True}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"trop forte, une augmentation empêche le modèle d'apprendre": "A",
          "une augmentation garde le label de l'image": "B",
          "on n'augmente jamais le test": "D"}'''),
        ("d", LETTER, '"D"',
         r'''mistakes={"le générateur fabrique les batches à la demande : la mémoire n'est pas en jeu": "A",
          "le learning rate est un réglage de l'optimiseur": "B",
          "le mélange n'a rien à voir avec la fin d'une epoch": "C"}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"un IterableDataset n'a pas de __len__": "A", "la taille du batch est un réglage du DataLoader": "C"}'''),
        ("f", TF, "True", r'''mistakes={"le modèle revoit-il deux fois le même exemple ?": False}'''),
    ]),
    Paper("24.Q9", "Motifs conv-conv-pool, stride contre pooling, batchnorm contre dropout", [
        ("a", LETTER, '"C"',
         r'''mistakes={"relis le listing 24.44 : la convolution est écrite sans activation": "A",
          "la batchnorm normalise la sortie de la convolution": "B"}'''),
        ("b", TF, "True",
         r'''mistakes={"que devient un biais constant après la soustraction de la moyenne du canal ?": False}'''),
        ("c", DEC2, "9 / 4",
         r'''decimals=2, mistakes={"la surface est divisée par 4 : écris aussi le facteur des canaux": 0.25,
          "les canaux d'entrée et de sortie sont multipliés par 3 tous les deux : C_in × C_out est multiplié par 9": 0.75,
          "la hauteur et la largeur sont divisées par 2 : la surface est divisée par 4": 9,
          "relis l'énoncé : les canaux sont multipliés par 3": 1.0}'''),
        ("d", LETTER, '"A"',
         r'''mistakes={"compare les accuracies de validation du tableau du §24.4.4": "B",
          "compare les temps par epoch du tableau du §24.4.4": "C"}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"relis la fin du §24.4.5 : les grosses couches denses portent l'essentiel des paramètres": "A",
          "relis la fin du §24.4.5 : on résume d'abord chaque canal": "B"}'''),
    ]),
    Paper("24.Q10", "Séries pour un RNN : fenêtres, scaler, inverse_transform, RMSE", [
        ("a", LETTER, '"D"',
         r'''mistakes={"le scaler ne doit pas voir le test": "A", "un scaler par fenêtre changerait l'échelle d'un exemple à l'autre": "B",
          "le scaler s'ajuste sur les données d'entraînement": "C"}'''),
        ("b", SHAPE, '"(500, 12, 1)"',
         r'''mistakes={"avec batch_first=True, l'axe des exemples vient en premier": "(12, 500, 1)",
          "n'oublie pas l'axe des features, même pour une seule variable": "(500, 12)"}'''),
        ("c", LETTER, '"A"', r'''mistakes={"une erreur se lit dans l'unité de ce qu'on prédit": "B"}'''),
        ("d", TF, "False",
         r'''mistakes={"un scaler ajusté sur le test, c'est une fuite, et une autre échelle que celle apprise par le réseau": True}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"une sigmoïde écraserait la prédiction dans un intervalle": "A", "une softmax sert à des classes, pas à une valeur": "B"}'''),
        ("f", LETTER, '"B"',
         r'''mistakes={"un réseau plus gros n'est pas une référence simple": "A", "la RMSE du train ne dit rien de la prévision": "C"}'''),
    ]),
    Paper("24.Q11", "🕰️ return_sequences, stateful, TimeDistributed : que devient chaque option en PyTorch ?", [
        ("a", LETTER, '"D"',
         r'''mistakes={"h_n ne garde que le dernier pas": "A", "c_n est la mémoire du dernier pas": "B",
          "output[:, -1] ne garde que le dernier pas": "C"}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"h_n[0] est l'état de la première couche": "B", "c_n est la mémoire de la cellule, pas sa sortie": "C",
          "output[0] est la séquence du premier exemple": "D"}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"aplatir mélangerait tous les pas dans une seule couche dense": "A",
          "relis ce que fait nn.Linear sur un tenseur à trois dimensions": "C"}'''),
        ("d", LETTER, '"C"',
         r'''mistakes={"relis l'encadré « Le livre, corrigé » du §24.5.9 : deux exemples consécutifs d'un batch sont indépendants": "A",
          "chaque ligne a son propre état": "B"}'''),
        ("e", TF, "True", r'''mistakes={"que devient le graphe du batch précédent après son backward() ?": False}'''),
        ("f", TF, "False", r'''mistakes={"relis le début du §24.5.9 : de quel état part chaque séquence ?": True}'''),
    ]),
    Paper("24.Q12", "🕰️ API fonctionnelle de Keras 3 et forward() de PyTorch : branches, partage, gel", [
        ("a", LETTER, '"C"',
         r'''mistakes={"un module est un seul objet, avec un seul jeu de poids": "A",
          "appeler deux fois un module est permis": "B"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"torch.cat met bout à bout, il n'additionne pas": "A", "torch.stack crée un nouvel axe": "C"}'''),
        ("c", SHAPE, '"(4, 32, 8, 8)"',
         r'''mistakes={"dim=1 est l'axe des canaux": "(8, 16, 8, 8)", "torch.cat ne crée pas d'axe": "(4, 2, 16, 8, 8)",
          "torch.cat met bout à bout ; il n'additionne pas les valeurs": "(4, 16, 8, 8)"}'''),
        ("d", TF, "False",
         r'''mistakes={"relis le piège classique du §24.6 : qu'est-ce qui change encore en mode train() ?": True}'''),
        ("e", LETTER, '"A"',
         r'''mistakes={"relis l'encadré « Le livre, corrigé » du §24.6.2 : le modèle se construit-il ?": "B",
          "qu'attend une convolution 2D en entrée ?": "C"}'''),
        ("f", TF, "True", r'''mistakes={"relis l'encadré 🕰️ du §24.6": False}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("24.R1", "Ch. 23 : Ce qu'un historique d'entraînement doit contenir pour comparer deux runs", [
        ("a", LETTERS, '"ABD"',
         r'''mistakes={"le test ne se regarde qu'une fois, à la fin : pas à chaque epoch": "ABDE",
          "une copie des poids à chaque epoch coûte trop de mémoire ; on garde le meilleur checkpoint": "ABCD",
          "le learning rate de chaque epoch aide à lire les courbes": "AB",
          "la loss d'entraînement fait partie de l'historique": "BD"}'''),
        ("b", LETTERS, '"BCD"',
         r'''mistakes={"la couleur des courbes ne distingue pas deux runs": "ABCD",
          "sans la graine, peux-tu refaire le run ?": "CD",
          "deux runs peuvent différer par un réglage laissé par défaut": "BD",
          "la durée et la machine comptent pour comparer des coûts": "BC"}'''),
        ("c", TF, "False",
         r'''mistakes={"relis comment train_one_epoch moyenne sa loss pendant que les poids changent": True}'''),
        ("d", WHOLE, "5",
         r'''mistakes={"relis comment les epochs sont numérotées dans l'énoncé": 4, "la meilleure epoch a la plus petite loss": 1,
          "cherche la plus petite des six valeurs": 3}'''),
    ]),
    Paper("24.R2", "Ch. 21 : Taille de sortie d'une convolution à padding 'same' et stride 2", [
        ("a", WHOLE, "math.ceil(28 / 2)", r'''mistakes={"avec un stride de 2, la taille est divisée par 2": 28,
          "« same » complète de zéros : la convolution ne perd pas les bords": (28 - 5) // 2 + 1}'''),
        ("b", WHOLE, "math.ceil(15 / 2)",
         r'''fractional="une hauteur est un nombre entier de lignes : arrondis au-dessus", mistakes={"Keras arrondit au-dessus : ⌈15/2⌉": 7}'''),
        ("c", WHOLE, "5 // 2", r'''mistakes={"p = ⌊k/2⌋, pas k": 5}'''),
        ("d", WHOLE, "(32 + 2 * 2 - 5) // 2 + 1",
         r'''mistakes={"n'oublie pas le + 1 de la formule": 15,
          "avec un stride de 2, la sortie est deux fois plus petite": 32,
          "applique la formule avec le padding de c)": (32 - 5) // 2 + 1}'''),
        ("e", WHOLE, "(15 - 3) // 2 + 1",
         r'''mistakes={"sans padding, la taille n'est pas ⌈15/2⌉": 8, "n'oublie pas le + 1 de la formule": 6,
          "le stride vaut 2 : divise par 2": 13}'''),
        ("f", TF, "True",
         r'''mistakes={"relis le ch. 21 : PyTorch accepte-t-il un noyau pair avec padding=\"same\" ?": False}'''),
    ]),
    Paper("24.R3", "Ch. 17 : Ce que change la division des logits avant le softmax", [
        ("a", DEC3, "softmax(Z_R3).max()",
         r'''decimals=3, mistakes={"la softmax prend des exponentielles, pas les logits eux-mêmes": 2 / 3,
          "c'est la plus grande probabilité qu'on demande": softmax(Z_R3).min()}'''),
        ("b", DEC3, "softmax(Z_R3 / 0.5).max()",
         r'''decimals=3, mistakes={"pour T = 0,5, diviser par T revient à doubler les logits": softmax(Z_R3 * 0.5).max(),
          "applique la température : divise chaque logit par T avant la softmax": softmax(Z_R3).max()}'''),
        ("c", DEC2, "softmax(Z_R3 / 2).max()",
         r'''decimals=2, mistakes={"pour T = 2, diviser par T revient à diviser les logits par 2": softmax(Z_R3 * 2).max(),
          "applique la température : divise chaque logit par T avant la softmax": softmax(Z_R3).max()}'''),
        ("d", LETTER, '"B"',
         r'''mistakes={"c'est quand T grandit que la distribution s'aplatit": "A",
          "regarde ce que deviennent les écarts entre logits divisés par un T minuscule": "C"}'''),
        ("e", TF, "True", r'''mistakes={"diviser par un nombre positif change-t-il l'ordre des logits ?": False}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("24.1", "Le coût d'une grille (27 hyperparamètres, 54 entraînements) contre une recherche aléatoire : "
                  "probabilité de tomber dans les 5 % meilleurs réglages en n essais", [
        ("a", WHOLE, "2 ** 27",
         r'''mistakes={"2 valeurs pour chacun des 27 : 2 puissance 27, pas 27 au carré": 27 ** 2,
          "les nombres de valeurs se multiplient, ils ne s'additionnent pas": 54}'''),
        ("b", DEC1, "2 ** 27 / (3600 * 24 * 365)",
         r'''decimals=1, mistakes={"une année compte 3 600 × 24 × 365 secondes": 2 ** 27 / (3600 * 24)}'''),
        ("c", WHOLE, "3 * (3 * 3 * 2)",
         r'''mistakes={"chaque candidat est entraîné une fois par fold": 18, "multiplie les nombres de valeurs": 3 * (3 + 3 + 2)}'''),
        ("d", WHOLE, "54 * 70 // 60",
         r'''mistakes={"en minutes, pas en secondes": 54 * 70, "chaque candidat est entraîné une fois par fold": 18 * 70 // 60}'''),
        ("e", DEC2, "1 - 0.95 ** 20",
         r'''decimals=2, mistakes={"c'est la probabilité de tout rater ; on demande celle d'en trouver au moins un": 0.95 ** 20,
          "0,05 × 20 compterait deux fois les tirages qui réussissent ensemble": 1.0}'''),
        ("f", WHOLE, "math.ceil(math.log(0.05) / math.log(0.95))",
         r'''mistakes={"arrondis au-dessus : le premier entier n tel que 0,95ⁿ ≤ 0,05": 58,
          "pose 0,95ⁿ ≤ 0,05, puis prends les logarithmes": 20},
          fractional="un nombre de tirages est entier : prends le premier entier qui suffit"'''),
        ("g", DEC3, "1 - 0.95 ** 18",
         r'''decimals=3, mistakes={"c'est la probabilité de tout rater ; on demande celle d'en trouver au moins un": 0.95 ** 18,
          "tu as pris le mauvais nombre de tirages": 1 - 0.95 ** 20}'''),
    ]),
    Paper("24.2", "Formes et paramètres du CNN à pooling du livre", [
        ("a", SHAPE, '"(256, 30, 28, 28)"',
         r'''mistakes={"avec padding=2, un noyau de 5 garde la taille": "(256, 30, 24, 24)",
          "PyTorch range les canaux juste après l'axe des exemples": "(256, 28, 28, 30)",
          "n'oublie pas l'axe des exemples": "(30, 28, 28)"}'''),
        ("b", SHAPE, '"(256, 30, 14, 14)"',
         r'''mistakes={"le pooling 2 × 2 divise la hauteur et la largeur par 2": "(256, 30, 28, 28)",
          "le pooling ne change pas le nombre de canaux": "(256, 15, 14, 14)",
          "n'oublie pas l'axe des exemples": "(30, 14, 14)"}'''),
        ("c", WHOLE, "P_CONV1", r'''mistakes={"n'oublie pas les 30 biais": 30 * 25, "un filtre 5 × 5 a 25 poids, pas 5": 30 * (5 + 1)}'''),
        ("d", WHOLE, "P_CONV2",
         r'''mistakes={"n'oublie pas les 16 biais": 16 * 30 * 9, "chaque filtre a autant de canaux que l'entrée (30)": 16 * (9 + 1),
          "un filtre 3 × 3 a 9 poids par canal": 16 * (30 * 3 + 1)}'''),
        ("e", WHOLE, "N_FLAT",
         r'''mistakes={"compte aussi les canaux : 16 valeurs par position": 49,
          "la sortie du second pooling est en 7 × 7, pas en 14 × 14": 16 * 14 * 14, "la seconde convolution a 16 filtres, pas 30": 30 * 7 * 7}'''),
        ("f", WHOLE, "dense(N_FLAT, 128)",
         r'''mistakes={"n'oublie pas les 128 biais": N_FLAT * 128, "la couche reçoit 16 × 7 × 7 valeurs": dense(49, 128)}'''),
        ("g", WHOLE, "P_TOTAL",
         r'''mistakes={"n'oublie pas les deux dernières couches denses": P_CONV1 + P_CONV2 + dense(N_FLAT, 128),
          "n'oublie pas les deux convolutions": P_DENSE,
          "la première couche dense reçoit 16 × 7 × 7 valeurs": P_CONV1 + P_CONV2 + dense(49, 128) + dense(128, 64) + dense(64, 10)}'''),
        ("h", DEC3, "P_DENSE / P_TOTAL",
         r'''decimals=3, mistakes={"relis quelle part est demandée": (P_CONV1 + P_CONV2) / P_TOTAL,
          "une fraction entre 0 et 1, pas un pourcentage": 100 * P_DENSE / P_TOTAL,
          "les trois couches denses, pas seulement la première": dense(N_FLAT, 128) / P_TOTAL}'''),
    ]),
    Paper("24.3", "Stride ou pooling : compter les multiplications", [
        ("a", WHOLE, "conv_mac(28, 28, 30, 1, 5)",
         r'''mistakes={"un noyau 5 × 5 compte 25 multiplications par position": conv_mac(28, 28, 30, 1, 1) * 5,
          "avec padding=2, la sortie est en 28 × 28": conv_mac(24, 24, 30, 1, 5)}'''),
        ("b", WHOLE, "conv_mac(14, 14, 16, 30, 3)",
         r'''mistakes={"la seconde convolution travaille sur la sortie du premier pooling, en 14 × 14": conv_mac(28, 28, 16, 30, 3),
          "chaque filtre a 30 canaux d'entrée": conv_mac(14, 14, 16, 1, 3),
          "le pooling vient après la convolution : sa sortie est en 14 × 14, pas en 7 × 7": conv_mac(7, 7, 16, 30, 3)}'''),
        ("c", WHOLE, "MAC_DENSE",
         r'''mistakes={"les trois couches denses : 784 → 128 → 64 → 10": 784 * 128,
          "l'énoncé néglige les biais": 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10,
          "la première couche dense reçoit 784 valeurs": 49 * 128 + 128 * 64 + 64 * 10}'''),
        ("d", WHOLE, "MAC_STRIDE",
         r'''mistakes={"avec un stride de 2, la seconde convolution sort en 7 × 7": conv_mac(14, 14, 30, 1, 5) + conv_mac(14, 14, 16, 30, 3),
          "les deux convolutions ensemble": conv_mac(14, 14, 30, 1, 5),
          "additionne les deux convolutions": conv_mac(7, 7, 16, 30, 3),
          "avec un stride de 2, la première convolution sort en 14 × 14": conv_mac(28, 28, 30, 1, 5) + conv_mac(7, 7, 16, 30, 3)}'''),
        ("e", DEC2, "(MAC_POOL + MAC_DENSE) / (MAC_STRIDE + MAC_DENSE)",
         r'''decimals=2, mistakes={"ajoute les couches denses aux deux versions": MAC_POOL / MAC_STRIDE,
          "la version à pooling au numérateur": (MAC_STRIDE + MAC_DENSE) / (MAC_POOL + MAC_DENSE)}'''),
        ("f", WHOLE, "30 * 14 * 14 * 3",
         r'''mistakes={"3 comparaisons par valeur de sortie, pas par valeur d'entrée": 30 * 28 * 28 * 3,
          "compte les comparaisons, pas seulement les valeurs de sortie": 30 * 14 * 14,
          "n'oublie pas les 30 canaux": 14 * 14 * 3}'''),
        ("g", DEC1, "147 / 36",
         r'''decimals=1, mistakes={"le rapport de la version à pooling à la version à stride": 36 / 147}'''),
    ]),
    Paper("24.4", "Ce que renvoie nn.LSTM : output, h_n et c_n", [
        ("a", SHAPE, '"(8, 5, 4)"',
         r'''mistakes={"output a H valeurs par pas, pas D": "(8, 5, 3)", "avec batch_first=True, l'axe des exemples vient en premier": "(5, 8, 4)",
          "output garde tous les pas de temps": "(8, 4)"}'''),
        ("b", SHAPE, '"(2, 8, 4)"',
         r'''mistakes={"batch_first ne change pas la forme de h_n": "(8, 2, 4)", "h_n a une ligne par couche": "(1, 8, 4)",
          "h_n a H valeurs par exemple": "(2, 8, 3)"}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"output[0] est la séquence du premier exemple : écris sa forme": "B",
          "c_n a la forme de h_n tout entier, pas celle d'une de ses lignes": "C",
          "output[:, :, 0] garde tous les pas de temps : écris sa forme": "D"}'''),
        ("d", SHAPE, '"(8, 5, 8)"',
         r'''mistakes={"les deux sens sont côte à côte sur la dernière dimension": "(8, 10, 4)",
          "avec deux sens, la dernière dimension double": "(8, 5, 4)"}'''),
        ("e", SHAPE, '"(4, 8, 4)"',
         r'''mistakes={"une ligne par couche et par sens": "(2, 8, 4)", "les sens multiplient le nombre de lignes, pas H": "(2, 8, 8)"}'''),
        ("f", WHOLE, "lstm_layer_params(3, 4) + lstm_layer_params(4, 4)",
         r'''mistakes={"PyTorch range deux vecteurs de biais par couche": 4 * 4 * (3 + 4) + 16 + 4 * 4 * (4 + 4) + 16,
          "la seconde couche reçoit H = 4 valeurs, pas D = 3": 2 * lstm_layer_params(3, 4),
          "un LSTM a quatre blocs de poids (les trois portes et le candidat)": (4 * (3 + 4) + 8) + (4 * (4 + 4) + 8),
          "n'oublie pas la seconde couche": lstm_layer_params(3, 4),
          "n'oublie pas la première couche": lstm_layer_params(4, 4)}'''),
        ("g", SHAPE, '"(8, 5, 1)"',
         r'''mistakes={"nn.Linear remplace la dernière dimension seulement": "(8, 1)", "nn.Linear agit sur la dernière dimension": "(8, 5, 4)"}'''),
        ("h", SHAPE, '"(5, 8, 3)"',
         r'''mistakes={"sans batch_first, le temps vient en premier": "(8, 5, 3)", "les features restent sur la dernière dimension": "(3, 5, 8)"}'''),
    ]),
    Paper("24.5", "Fenêtres, pas d'optimisation et durée : de 200 à 20 000 points", [
        ("a", WHOLE, "200 - 13",
         r'''mistakes={"la dernière fenêtre doit laisser une valeur pour sa cible": 200 - 13 + 1,
          "les fenêtres commencent à chaque pas : elles se chevauchent": 200 // 13}'''),
        ("b", WHOLE, "20_000 - 13",
         r'''mistakes={"la dernière fenêtre doit laisser une valeur pour sa cible": 20_000 - 13 + 1,
          "les fenêtres commencent à chaque pas : elles se chevauchent": 20_000 // 13}'''),
        ("c", WHOLE, "20_000 - 13",
         r'''mistakes={"un pas par exemple : compte les fenêtres, pas les points": 20_000}'''),
        ("d", WHOLE, "math.ceil((20_000 - 13) / 32)",
         r'''fractional="un nombre de pas est entier : le dernier mini-batch, incomplet, compte aussi",
          mistakes={"arrondis au-dessus : le dernier mini-batch, incomplet, compte aussi": (20_000 - 13) // 32}'''),
        ("e", DEC1, "100 * 5 / 60",
         r'''decimals=1, mistakes={"en minutes, pas en secondes": 500, "500 s font 8 min 20 s, soit 8,33 min et non 8,20": 8.2}'''),
        ("f", WHOLE, "28 * 3600 // 100",
         r'''mistakes={"28 heures pour 100 epochs : divise par 100": 28 * 3600, "en secondes : une heure fait 3 600 s": 28 * 60 // 100}'''),
        ("g", DEC3, "5 * (200 - 13) / (20_000 - 13)",
         r'''decimals=3, mistakes={"proportionnelle au nombre d'exemples, pas au nombre de points": 5 * 200 / 20_000,
          "la petite série a moins d'exemples : l'epoch dure moins de 5 s": 5 * (20_000 - 13) / (200 - 13)}'''),
    ]),
    Paper("24.7", "Fermi : le budget d'une recherche d'hyperparamètres sur CPU, sur GPU et en précision mixte", [
        ("a", WHOLE, "round(OPS_247 / 1e12)",
         r'''mistakes={"compte 2 opérations par MAC": round(OPS_247 / 2 / 1e12),
          "un entraînement coûte 3 fois la passe avant": round(OPS_247 / 3 / 1e12),
          "en milliers de milliards (10¹²), pas en milliards": round(OPS_247 / 1e9),
          "n'oublie pas les 20 epochs": round(OPS_247 / 20 / 1e12)}'''),
        ("b", WHOLE, "round(CPU_MIN_247)",
         r'''mistakes={"en minutes, pas en secondes": round(CPU_MIN_247 * 60), "en minutes, pas en heures": round(CPU_MIN_247 / 60),
          "un entraînement coûte 3 fois la passe avant": round(CPU_MIN_247 / 3)}'''),
        ("c", WHOLE, "4 * 3 * 3 * 3",
         r'''mistakes={"chaque candidat est entraîné une fois par fold": 36, "multiplie les nombres de valeurs": (4 + 3 + 3) * 3}'''),
        ("d", WHOLE, "round(108 * CPU_MIN_247 / 60)",
         r'''mistakes={"en heures, pas en minutes": round(108 * CPU_MIN_247), "n'oublie pas les 3 folds": round(36 * CPU_MIN_247 / 60)}'''),
        ("e", DEC2, "108 * OPS_247 / 1e13 / 3600",
         r'''decimals=2, mistakes={"en heures, pas en secondes": 108 * OPS_247 / 1e13, "en heures, pas en minutes": 108 * OPS_247 / 1e13 / 60}'''),
        ("f", DEC1, "108 * OPS_247 / 1e13 / 2.5 / 60",
         r'''decimals=1, mistakes={"2,5 fois plus vite : divise la durée, ne la multiplie pas": 108 * OPS_247 / 1e13 * 2.5 / 60,
          "en minutes, pas en heures": 108 * OPS_247 / 1e13 / 2.5 / 3600,
          "648 s font 10 min 48 s, soit 10,8 min": 6.5}'''),
        ("g", WHOLE, "round(30 * CPU_MIN_247 / 60)",
         r'''mistakes={"en heures, pas en minutes": round(30 * CPU_MIN_247),
          "un seul jeu de validation : un entraînement par réglage, pas trois": round(90 * CPU_MIN_247 / 60)}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 24.1 à 24.5 et 🧮 24.7)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"F\"`) ; "
               "plusieurs lettres : entre guillemets (`\"AF\"`) ; une forme : entre guillemets (`\"(3, 5, 1)\"`). "
               "Arrondis comme l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore "
               "remplies affichent ⏳. Les questions « dans ta copie », la dérivation ∂ 24.6, la réflexion "
               "(🗣️ 24.8, ⚖️ 24.9, 📄 24.10) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to E (exercises 24.11 to 24.31): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "24.11 à 24.15", "améliorer un MLP sur Fashion-MNIST : taille de batch, quatre MLP, dropout et max-norm, "
                           "validation au-dessus du train, journal d'expériences"),
    ("B", "24.16 à 24.18", "un réseau PyTorch en estimateur scikit-learn : validation croisée, grille, hasard et zoom"),
    ("C", "24.19 à 24.24", "des CNN sur CIFAR-10 : un 🐛 de traduction, du simple au profond, batchnorm et dropout, "
                           "augmentation, images synthétiques"),
    ("D", "24.25 à 24.27", "des séries et un LSTM : sommes de sinus, fenêtre, profondeur et données, sorties et état "
                           "porté sur les taches solaires"),
    ("E", "24.28 à 24.31", "au-delà de nn.Sequential : branches et blocs résiduels, précision mixte et compilation, "
                           "génération de texte, défi CIFAR-10 frugal"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 24.Q1 à 24.Q12, 24.R1 à 24.R3, 24.1 à 24.5, 24.7 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 | "
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
        title = "# 24 · PyTorch en pratique 2 : améliorer, chercher, CNN et RNN — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch24_pytorch_2/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 24`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 24 · PyTorch en pratique 2 : améliorer, chercher, CNN et RNN — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Améliorer un réseau par expériences : un hyperparamètre à la fois, dropout et max-norm, un journal "
               "de chaque run.\n"
               "- Régler un réseau PyTorch avec scikit-learn : estimateur maison, `Pipeline`, validation croisée, "
               "recherches en grille et au hasard.\n"
               "- Construire des CNN sur CIFAR-10 (stride, batchnorm, augmentation, blocs résiduels), préparer une "
               "série pour un LSTM, lire les sorties de `nn.LSTM`, générer du texte.\n\n"
               "**Rappel express.** Mises à jour par epoch : $\\lceil N/B \\rceil$ ; bruit d'un mini-batch en "
               "$1/\\sqrt{B}$. Dropout inversé : $\\tilde h = m\\,h/(1-p)$. Max-norm : "
               "$\\mathbf{w} \\leftarrow \\mathbf{w}\\min(1, c/\\lVert\\mathbf{w}\\rVert)$. Coût d'une grille : "
               "$K \\times \\prod_j |V_j|$. Convolution : $C_{out}(C_{in}k^2 + 1)$ paramètres, "
               "$H_{out}W_{out}C_{out}C_{in}k^2$ MAC. `nn.LSTM(batch_first=True)` : `output` $(N, T, H)$, `h_n` "
               "$(L, N, H)$. Bloc résiduel : $y = F(x) + x$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 24.11 à 24.31 (parties A à E : améliorer un MLP, scikit-learn, les CNN "
             "sur CIFAR-10, les séries et le LSTM, les architectures à branches, la précision mixte, la génération "
             "de texte et le défi) seront ajoutés à la prochaine session de génération ; relance alors "
             "`python tools/start_chapter.py 24` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu mener une amélioration par expériences (un réglage à la fois, un journal) et chiffrer le "
               "coût d'une recherche d'hyperparamètres ?\n"
               "2. Sais-tu envelopper un réseau PyTorch dans un estimateur scikit-learn et le valider sans fuite ?\n"
               "3. Sais-tu construire un CNN et un LSTM en PyTorch, lire leurs formes, et écrire un bloc résiduel "
               "dans `forward()` ?\n\n"
               "**Pour aller plus loin** : le « Deep Learning Tuning Playbook » de Google et l'article de He et coll. "
               "(2016), cités dans la fiche. La suite : le checkpoint de la partie V, puis le ch. 25, les "
               "autoencodeurs." + later)]


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
