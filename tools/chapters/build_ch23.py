#!/usr/bin/env python
"""Build the two notebooks of chapter 23 from a single source (used by Claude).

    python tools/chapters/build_ch23.py
    python tools/run_all_notebooks.py chapitres/ch23_pytorch_1/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch23_pytorch_1/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q12, of the recalls R1 to R3, of the ✏️ paper exercises
23.1 to 23.7 and of the 🧮 estimate 23.8 (02_exercices.md). The reflection (🗣️ 23.9, ⚖️ 23.10, 📄 23.11) and the
interview are corrected with 05_solutions.md.
Parts A to D (exercises 23.12 to 23.29) come at the second generation session: tensors and Fashion-MNIST, the
preprocessing pipeline and the book's MLP, the learner's own fit() loop with its history and error analysis,
prediction, checkpoints, a pretrained torchvision model, callbacks written as plain PyTorch, and the
reproducibility challenge. The chapter has no mylearn module: the notebook reuses
mylearn.metrics.confusion_matrix (ch. 3) and mylearn.nn.regularization.EarlyStopping (ch. 20).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import STARS, Ex, Paper, Part, badge, guarded, md, paper_cells, part_cells, setup_cell, write_notebook  # noqa: E402,F401

CHAPTER = "23"
FOLDER = "chapitres/ch23_pytorch_1"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np

# Recall 23.R3: five pixels of a tiny training set, already divided by 255
PIXELS_R3 = np.array([0.0, 0.2, 0.4, 0.4, 1.0])
MU_R3, SIGMA_R3 = PIXELS_R3.mean(), PIXELS_R3.std()          # ddof=0, like np.std
SIGMA1_R3 = PIXELS_R3.std(ddof=1)

# Ex 23.1: the letters M to X at positions 0 to 11, reshaped in C order
LETTERS_231 = np.array(list("MNOPQRSTUVWX"))


def flat_index(index, shape):
    """Position in memory (C order) of the element `index` of a tensor of shape `shape`."""
    return int(np.ravel_multi_index(index, shape))


# Ex 23.2: an RGB image of 2 rows and 3 columns, channels_last, x[h, w, c] at position 9h + 3w + c
X_232 = np.arange(18).reshape(2, 3, 3)
Y_232 = X_232.transpose(2, 0, 1)                                # x.permute(2, 0, 1): y[c, h, w] = x[h, w, c]
Z_232 = np.ascontiguousarray(Y_232)                             # y.contiguous()


def dense(n_in, n_out):
    """Parameters of nn.Linear(n_in, n_out): weights and biases."""
    return (n_in + 1) * n_out


def softmax(z):
    e = np.exp(np.asarray(z, float) - max(z))
    return e / e.sum()


def cross_entropy(z, label):
    """What nn.CrossEntropyLoss() returns for one example: -z_y + ln sum_j exp(z_j)."""
    z = np.asarray(z, float)
    return float(-z[label] + np.log(np.exp(z).sum()))


# Ex 23.4: logits of a 3-class classifier, label 1
Z_234 = [1.0, 2.0, 0.0]
P_234 = softmax(Z_234)

# Ex 23.5: 10 % of the 60 000 training images kept for validation, mini-batches of 256
N_235 = 60_000 - 6_000

# Ex 23.6: validation losses of epochs 1 to 16
VAL_236 = [0.60, 0.45, 0.40, 0.41, 0.38, 0.385, 0.379, 0.39, 0.40, 0.41, 0.37, 0.42, 0.43, 0.44, 0.45, 0.46]


def early_stopping(losses, patience, min_delta=0.0):
    """(stop epoch, best epoch), epochs counted from 1; the rule of the statement and of mylearn's EarlyStopping."""
    best, best_epoch, counter = None, None, 0
    for epoch, loss in enumerate(losses, start=1):
        if best is None or loss < best - min_delta:
            best, best_epoch, counter = loss, epoch, 0
        else:
            counter += 1
            if counter >= patience:
                return epoch, best_epoch
    return None, best_epoch


STOP_236A, BEST_236A = early_stopping(VAL_236, 3)
STOP_236C, BEST_236C = early_stopping(VAL_236, 3, 0.005)
STOP_236E, BEST_236E = early_stopping(VAL_236, 5)


# Ex 23.7: the book's schedule, a factor for LambdaLR (epochs counted from 0)
def book_lambda(e):
    return max(0.1, 1 - 0.01 * e)


# Ex 23.8: the book's MLP on 60 000 images, 6PN operations per epoch, a CPU at 50 billion operations per second
P_238, N_238 = dense(784, 784) + dense(784, 10), 60_000
OPS_238 = 6 * P_238 * N_238
SECONDS_238 = OPS_238 / 50e9'''

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "E"'
LETTERS = 'the letters, e.g. "EF"'
SHAPE = 'a shape as a string, e.g. "(3, 5, 1)"'
INDEX = 'the indices as a string, e.g. "(5, 7, 1)"'
WHOLE = "a whole number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
DEC4 = "a number, 4 decimals"

PAPER = [
    # ---------------------------------------------------------------- quizzes
    Paper("23.Q1", "Pourquoi utiliser une bibliothèque plutôt que tout coder soi-même ?", [
        ("a", TF, "False",
         r'''mistakes={"écrire mylearn t'a appris ce que calculent les fonctions que tu appelles maintenant": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"les formules sont les mêmes : c'est la façon de les calculer qui change": "A",
          "aucune bibliothèque ne garantit une accuracy : elle dépend des données et du modèle": "C",
          "sans données, il n'y a rien à apprendre, quelle que soit la bibliothèque": "D"}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"c'est l'inverse : la bibliothèque fait monter d'un niveau d'abstraction": "A",
          "l'analogie parle de niveau d'abstraction, pas d'une langue à apprendre": "B",
          "l'analogie oppose deux façons d'écrire un programme, pas un plan et une maison": "D"}'''),
        ("d", TF, "True",
         r'''mistakes={"relis comment les tests de mylearn décident si ta fonction est juste": False}'''),
        ("e", TF, "False",
         r'''mistakes={"qui choisit les données, le prétraitement, la loss et la façon d'évaluer ?": True}'''),
    ]),
    Paper("23.Q2", "Avertissement ou erreur ? Versions et dépréciations", [
        ("a", LETTER, '"B"',
         r'''mistakes={"un avertissement n'arrête pas le programme": "A",
          "un avertissement ne dit pas que le résultat est faux": "C",
          "une bibliothèque absente lève une ImportError, pas un avertissement": "D"}'''),
        ("b", TF, "True", r'''mistakes={"que fait Python d'une exception que personne ne rattrape ?": False}'''),
        ("c", TF, "False",
         r'''mistakes={"relis les encadrés 🕰️ : des méthodes et des arguments du livre ont disparu": True}'''),
        ("d", LETTER, '"D"',
         r'''mistakes={"la dernière version est justement ce qui change d'une année à l'autre": "A",
          "faire taire les avertissements ne fige rien": "B",
          "sans bibliothèque, ton code est plus lent, pas plus reproductible": "C"}'''),
        ("e", TF, "False",
         r'''mistakes={"tu ne verrais plus aucune annonce de changement à venir": True}'''),
    ]),
    Paper("23.Q3", "Déboguer méthodiquement : réduire, isoler, reconstruire", [
        ("a", LETTER, '"C"',
         r'''mistakes={"le code d'une grande bibliothèque se compte en centaines de milliers de lignes": "A",
          "une erreur de forme ou de type revient à chaque exécution": "B",
          "le même appel fautif planterait aussi avec une autre bibliothèque": "D"}'''),
        ("b", TF, "True", r'''mistakes={"relis la fin du §23.2.2 de la fiche": False}'''),
        ("c", WHOLE, "32",
         r'''mistakes={"la taille du mini-batch est le nombre de lignes de mat1": 784,
          "748 est la première dimension de mat2, la matrice de poids": 748,
          "10 est le nombre de sorties de la dernière couche": 10}'''),
        ("d", WHOLE, "784",
         r'''mistakes={"748 est justement la faute de frappe": 748,
          "in_features est le nombre de valeurs reçues par image, pas le nombre d'images": 32,
          "10 est le nombre de sorties (out_features)": 10}'''),
        ("e", TF, "False", r'''mistakes={"quel code a fourni les formes qui ne vont pas ensemble ?": True}'''),
    ]),
    Paper("23.Q4", "Les trois sens du mot « modèle »", [
        ("a", LETTER, '"A"',
         r'''mistakes={"dessiné avant tout entraînement : y a-t-il des poids appris ?": "B",
          "on parle d'un réseau précis, pas d'une façon de programmer": "C"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"l'envoi contient des poids appris": "A", "on parle d'un réseau précis, prêt à prédire": "C"}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"torch.nn, torch.optim et torch.utils.data ne sont pas un réseau": "A",
          "aucun poids appris n'est en jeu ici": "B"}'''),
        ("d", TF, "False",
         r'''mistakes={"un state_dict contient des tenseurs rangés par nom : où est décrite l'architecture ?": True}'''),
        ("e", TF, "True", r'''mistakes={"des formes compatibles disent seulement que le calcul s'exécute": False}'''),
    ]),
    Paper("23.Q5", "Tenseurs, arrays et type float32", [
        ("a", WHOLE, "4",
         r'''mistakes={"PyTorch garde l'axe des canaux (C = 1), en plus de N, H et W": 3,
          "un mini-batch d'images a plus de deux axes : N, C, H et W": 2}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"float16 ne sert qu'à la précision mixte, jamais par défaut": "A",
          "float64 est le défaut de NumPy, pas de PyTorch": "C",
          "uint8 est le type des pixels bruts, des entiers": "D"}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"from_numpy garde le type du tableau NumPy : quel est le défaut de NumPy ?": "B",
          "NumPy ne crée pas de float16 par défaut": "A", "0,5 n'est pas un entier": "D"}'''),
        ("d", TF, "True", r'''mistakes={"from_numpy ne copie pas : modifie a et regarde le tenseur": False}'''),
        ("e", WHOLE, "4", r'''mistakes={"32 bits, et un octet fait 8 bits": 32, "8 octets, c'est un float64": 8}'''),
    ]),
    Paper("23.Q6", "channels_first, channels_last et NCHW", [
        ("a", SHAPE, '"(3, 48, 64)"',
         r'''mistakes={"la hauteur (le nombre de lignes) vient avant la largeur": "(3, 64, 48)",
          "en channels_first, les canaux viennent d'abord": "(48, 64, 3)"}'''),
        ("b", SHAPE, '"(48, 64, 3)"',
         r'''mistakes={"la hauteur (le nombre de lignes) vient avant la largeur": "(64, 48, 3)",
          "en channels_last, les canaux viennent à la fin": "(3, 48, 64)"}'''),
        ("c", LETTER, '"A"',
         r'''mistakes={"c'est le format channels_last de Keras": "B",
          "l'axe des exemples vient toujours en premier": "C",
          "la hauteur vient avant la largeur": "D"}'''),
        ("d", SHAPE, '"(0, 3, 1, 2)"',
         r'''mistakes={"c'est la conversion inverse : donne, pour chaque axe du résultat, l'axe d'origine qui y va": "(0, 2, 3, 1)",
          "l'axe des exemples reste en position 0": "(3, 0, 1, 2)"}'''),
        ("e", TF, "False", r'''mistakes={"pense à une image en couleur de 3 × 3 pixels": True}'''),
    ]),
    Paper("23.Q7", "CPU, GPU, MPS, TPU : où calcule PyTorch ?", [
        ("a", LETTER, '"C"',
         r'''mistakes={"cuda est le device des GPU NVIDIA (et AMD, avec ROCm)": "A",
          "Metal est le framework graphique d'Apple, mais le device porte un autre nom": "B",
          "un Mac n'a pas de TPU": "D"}'''),
        ("b", TF, "True", r'''mistakes={"où se ferait le produit de matrices, si les deux tenseurs sont à deux endroits ?": False}'''),
        ("c", LETTER, '"C"',
         r'''mistakes={"l'horloge d'un GPU est plutôt plus lente que celle d'un CPU": "A",
          "un GPU a souvent moins de mémoire que l'ordinateur qui l'héberge": "B",
          "les GPU calculent en réels (float32, float16)": "D"}'''),
        ("d", TF, "True", r'''mistakes={"y a-t-il un GPU dans un environnement d'exécution CPU ?": False}'''),
        ("e", TF, "False", r'''mistakes={"relis l'encadré 🕰️ sur les accélérateurs : que faut-il pour un TPU ?": True}'''),
    ]),
    Paper("23.Q8", "Iris, MNIST, Fashion-MNIST : les « hello world » du ML", [
        ("a", WHOLE, "60000",
         r'''mistakes={"70 000 est le total, train et test réunis": 70000, "10 000 est la taille du test": 10000}'''),
        ("b", WHOLE, "10", ""),
        ("c", LETTER, '"A"',
         r'''mistakes={"MNIST est toujours gratuit": "B",
          "c'est ImageNet qui a retiré des catégories pour des raisons éthiques, pas MNIST": "C",
          "MNIST est en niveaux de gris": "D"}'''),
        ("d", TF, "True", r'''mistakes={"relis ce que veut dire « remplaçant direct »": False}'''),
        ("e", WHOLE, "4",
         r'''mistakes={"deux parties de la fleur, deux mesures pour chacune": 2, "3 est le nombre d'espèces": 3}'''),
    ]),
    Paper("23.Q9", "Labels entiers, one-hot et logits", [
        ("a", LETTER, '"B"',
         r'''mistakes={"la loss applique déjà sa propre softmax": "A",
          "les one-hot sont des cibles, pas des sorties du modèle": "C",
          "les labels sont la cible : le second argument": "D"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"les one-hot ne sont pas obligatoires : la loss prend des indices de classe": "A",
          "la loss ne lit pas de texte": "C", "des indices de classe sont des entiers (int64)": "D"}'''),
        ("c", TF, "False", r'''mistakes={"une seconde softmax produit-elle une valeur interdite ?": True}'''),
        ("d", SHAPE, '"(3, 10)"',
         r'''mistakes={"une ligne par label, une colonne par classe": "(10, 3)",
          "one_hot ajoute un axe de num_classes valeurs": "(3,)"}'''),
        ("e", SHAPE, '"(3, 9)"',
         r'''mistakes={"sans num_classes, le nombre de classes est deviné d'après le plus grand label": "(3, 10)",
          "les classes commencent à 0 : de 0 à 8, il y en a 9": "(3, 8)"}'''),
    ]),
    Paper("23.Q10", "fit() décortiqué : batch, epoch, validation, train() et eval()", [
        ("a", TF, "False", r'''mistakes={"relis l'encadré « model.eval() ne coupe pas les gradients »": True}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"le learning rate est réglé par l'optimiseur": "A",
          "le device se change avec .to(device)": "C", "le nombre d'epochs est dans ta boucle": "D"}'''),
        ("c", LETTER, '"D"',
         r'''mistakes={"des images jamais vues pendant l'entraînement, mais qui servent à choisir": "A",
          "10 000 images donnent une mesure assez précise": "B",
          "Keras ne vérifie pas d'où viennent les données de validation": "C"}'''),
        ("d", TF, "True", r'''mistakes={"relis le paragraphe sur loss.item() au §23.7": False}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"torch.compile accélère un modèle, il ne choisit ni loss ni optimiseur": "A",
          "eval() change le mode de certaines couches": "B", "to(device) déplace le modèle": "D"}'''),
    ]),
    Paper("23.Q11", "Sauvegarder tout, les poids ou l'architecture ?", [
        ("a", LETTER, '"C"',
         r'''mistakes={"sans l'état de l'optimiseur, Adam repartirait de moments nuls": "A",
          "l'objet modèle seul ne contient ni l'optimiseur, ni l'epoch, ni les générateurs": "B",
          "les prédictions ne permettent pas de continuer à apprendre": "D"}'''),
        ("b", LETTER, '"B"',
         r'''mistakes={"l'architecture n'est pas dans le state_dict : c'est le code": "A",
          "un state_dict ne contient aucun code": "C", "l'optimiseur a son propre state_dict": "D"}'''),
        ("c", TF, "False", r'''mistakes={"weights_only=True existe justement pour empêcher cela": True}'''),
        ("d", WHOLE, "4",
         r'''mistakes={"chaque couche Linear a deux tenseurs : ses poids et ses biais": 2,
          "Flatten et ReLU n'ont pas de paramètres": 8}'''),
        ("e", TF, "True", r'''mistakes={"relis ce que vérifie load_state_dict": False}'''),
    ]),
    Paper("23.Q12", "🕰️ Que devient le code Keras du livre en 2026 ?", [
        ("a", LETTER, '"B"',
         r'''mistakes={"ce sont les backends de Keras 2, en 2017": "A",
          "PyTorch n'est que l'un des backends de Keras 3": "C",
          "TensorFlow n'est que l'un des backends de Keras 3": "D"}'''),
        ("b", LETTER, '"A"',
         r'''mistakes={"predict_proba a été retiré lui aussi": "B", "to_categorical code des labels": "C",
          "fit entraîne le modèle": "D"}'''),
        ("c", TF, "False", r'''mistakes={"compare ce que font les deux fonctions": True}'''),
        ("d", TF, "False", r'''mistakes={"relis l'encadré 🕰️ du §23.3.3": True}'''),
        ("e", LETTER, '"C"',
         r'''mistakes={"to_yaml a été retiré pour des raisons de sécurité": "A",
          ".pt est l'extension habituelle de PyTorch": "B",
          ".h5 est un format ancien que Keras 3 lit encore": "D"}'''),
    ]),
    # ---------------------------------------------------------------- recalls
    Paper("23.R1", "Ch. 22 : (N, T, D) et batch_first", [
        ("a", SHAPE, '"(32, 20, 50)"',
         r'''mistakes={"avec batch_first=True, les exemples viennent d'abord": "(20, 32, 50)",
          "le temps (20 tokens) vient avant les features (50 valeurs)": "(32, 50, 20)"}'''),
        ("b", SHAPE, '"(20, 32, 50)"',
         r'''mistakes={"sans batch_first, le temps vient d'abord": "(32, 20, 50)"}'''),
        ("c", SHAPE, '"(2, 32, 64)"',
         r'''mistakes={"h_n n'est jamais batch_first : (couches × sens, N, H)": "(32, 2, 64)",
          "deux couches : un état final par couche": "(1, 32, 64)",
          "H = 64, la taille de l'état, pas celle des entrées": "(2, 32, 50)"}'''),
        ("d", TF, "False", r'''mistakes={"relis la documentation de batch_first : concerne-t-il les états ?": True}'''),
        ("e", TF, "True", r'''mistakes={"compare la place de N dans les deux formes": False}'''),
    ]),
    Paper("23.R2", "Ch. 20 : Les cinq lignes d'une étape d'entraînement", [
        ("a", LETTERS, '"BCEAD"',
         r'''mistakes={"il faut les gradients avant de mettre à jour les poids": "BCEDA",
          "il faut des logits avant de calculer la loss": "BECAD",
          "il faut une loss avant de calculer les gradients": "BCAED"}'''),
        ("b", TF, "True", r'''mistakes={"backward ajoute au .grad existant": False}'''),
        ("c", LETTER, '"B"',
         r'''mistakes={"c'est le travail d'optimizer.step()": "A", "c'est le travail d'optimizer.zero_grad()": "C",
          "c'est le travail de loss_fn(logits, yb)": "D"}'''),
        ("d", TF, "True", r'''mistakes={"un tenseur de loss est relié à tout le graphe de calcul du mini-batch": False}'''),
        ("e", LETTER, '"B"',
         r'''mistakes={"un planning par epoch avancerait alors une fois par mini-batch": "A",
          "PyTorch avertit si scheduler.step() précède optimizer.step()": "C",
          "l'optimiseur ne connaît pas le planning": "D"}'''),
    ]),
    Paper("23.R3", "Ch. 2 : Moyenne et écart-type du train pour normaliser les pixels", [
        ("a", DEC3, "MU_R3", r'''decimals=3, mistakes={"divise par 5, le nombre de pixels": 0.5}'''),
        ("b", DEC3, "SIGMA_R3",
         r'''decimals=3, mistakes={"avec le diviseur n (ddof=0), pas n − 1": SIGMA1_R3,
          "prends la racine de la variance": SIGMA_R3 ** 2}'''),
        ("c", DEC3, "(0.9 - MU_R3) / SIGMA_R3",
         r'''decimals=3, mistakes={"σ se calcule avec le diviseur n": (0.9 - MU_R3) / SIGMA1_R3,
          "soustrais d'abord la moyenne": 0.9 / SIGMA_R3}'''),
        ("d", TF, "False", r'''mistakes={"les statistiques de normalisation s'apprennent sur le train seul": True}'''),
        ("e", LETTER, '"A"',
         r'''mistakes={"c'est une remise à l'échelle par le minimum et le maximum": "B",
          "c'est seulement la division par 255": "C", "c'est l'inverse de la normalisation": "D"}'''),
    ]),
    # ---------------------------------------------------------------- paper exercises
    Paper("23.1", "Remodeler 12 éléments : indices et ordre de lecture", [
        ("a", 'a letter as a string, e.g. "M"', "str(LETTERS_231.reshape(3, 4)[2, 1])",
         r'''mistakes={"le dernier indice varie le plus vite : position i × 4 + j": "R",
          "les indices commencent à 0": "Q", "la forme (3, 4) a 4 colonnes : une ligne avance de 4 positions": "S"}'''),
        ("b", 'a letter as a string, e.g. "M"', "str(LETTERS_231.reshape(4, 3)[2, 1])",
         r'''mistakes={"la forme (4, 3) a 3 colonnes : position i × 3 + j": "V",
          "le dernier indice varie le plus vite": "S"}'''),
        ("c", 'a letter as a string, e.g. "M"', "str(LETTERS_231.reshape(2, 3, 2)[1, 2, 0])",
         r'''mistakes={"ordre C : position i × 6 + j × 2 + k": "R"}'''),
        ("d", INDEX, '"(0, 1, 1)"',
         r'''mistakes={"ordre C : 4 = i × 6 + j × 3 + k": "(0, 0, 1)", "Q est à la position 4 (M est à 0)": "(0, 1, 2)"}'''),
        ("e", LETTERS, '"AD"',
         r'''mistakes={"(4, 4) n'a pas 12 éléments non plus": "A", "(5, 3) n'a pas 12 éléments non plus": "D",
          "(12,) a bien 12 éléments": "ADE"}'''),
        ("f", "a list of 4 whole numbers, e.g. [9, 9, 9, 9]", "np.arange(12).reshape(3, 4).T.reshape(12)[:4].tolist()",
         r'''mistakes={"a.T n'est pas lu dans l'ordre de la mémoire : sa première ligne est la première colonne de a": [0, 1, 2, 3]}'''),
        ("g", WHOLE, "flat_index((1, 2, 1), (2, 3, 2))",
         r'''mistakes={"pour la forme (2, 3, 2), le deuxième indice avance de 2 positions, pas de 3": 13,
          "les positions commencent à 0": 12}'''),
    ]),
    Paper("23.2", "De NHWC à NCHW : où atterrit chaque pixel ?", [
        ("a", WHOLE, "flat_index((1, 2, 0), (2, 3, 3))",
         r'''mistakes={"dans x, c'est le dernier indice (le canal) qui varie le plus vite : 9h + 3w + c": 5,
          "les positions commencent à 0": 16}'''),
        ("b", SHAPE, "str(Y_232.shape)",
         r'''mistakes={"permute(2, 0, 1) met en tête l'axe 2 de x, puis son axe 0, puis son axe 1": "(3, 3, 2)"}'''),
        ("c", INDEX, '"(1, 2, 0)"',
         r'''mistakes={"y[c, h, w] = x[h, w, c] : le premier indice de y est le canal": "(0, 1, 2)",
          "ce sont les arguments de permute, pas des indices": "(2, 0, 1)"}'''),
        ("d", WHOLE, "int(np.argwhere(Z_232.ravel() == Y_232[0, 1, 2])[0, 0])",
         r'''mistakes={"contiguous a recopié les valeurs dans l'ordre de y : 6c + 3h + w": 15}'''),
        ("e", TF, "False", r'''mistakes={"y est-il encore contigu après permute ?": True}'''),
        ("f", LETTER, '"B"',
         r'''mistakes={"reshape relit la mémoire de x dans l'ordre : w[0, 0, 1] est à la position 1": "A",
          "la position 1 de la mémoire est x[0, 0, 1] : quel canal ?": "C",
          "reshape(3, 2, 3) a bien 18 éléments, rien ne l'empêche": "D"}'''),
        ("g", LETTER, '"A"',
         r'''mistakes={"y[0, 0, 1] = x[0, 1, 0] : de quel pixel s'agit-il ?": "B",
          "y[0, 0, 1] = x[0, 1, 0] : quel pixel, quel canal ?": "C",
          "permute ne lève pas d'erreur": "D"}'''),
        ("h", WHOLE, "64 * 3 * 32 * 32 * 4",
         r'''mistakes={"un float32 fait 4 octets": 64 * 3 * 32 * 32, "4 octets par valeur, pas 8": 64 * 3 * 32 * 32 * 8,
          "il y a 3 canaux": 64 * 32 * 32 * 4}'''),
    ]),
    Paper("23.3", "623 290 paramètres : refaire le compte du livre", [
        ("a", WHOLE, "dense(784, 784)", r'''mistakes={"n'oublie pas les 784 biais": 784 * 784}'''),
        ("b", WHOLE, "dense(784, 10)", r'''mistakes={"n'oublie pas les 10 biais": 784 * 10}'''),
        ("c", WHOLE, "dense(784, 784) + dense(784, 10)",
         r'''mistakes={"le total compte toutes les couches": dense(784, 784), "n'oublie pas les biais": 784 * 784 + 784 * 10}'''),
        ("d", DEC3, "dense(784, 784) / (dense(784, 784) + dense(784, 10))",
         r'''decimals=3, mistakes={"une fraction entre 0 et 1, pas un pourcentage": 100 * dense(784, 784) / (dense(784, 784) + dense(784, 10)), "a) divisé par c), pas l'inverse": (dense(784, 784) + dense(784, 10)) / dense(784, 784),
          "la part de la première couche, pas de la seconde": dense(784, 10) / (dense(784, 784) + dense(784, 10))}'''),
        ("e", WHOLE, "dense(784, 256) + dense(256, 10)",
         r'''mistakes={"n'oublie pas les biais": 784 * 256 + 256 * 10,
          "la couche de sortie reçoit 256 valeurs, pas 784": dense(784, 256) + dense(784, 10),
          "n'oublie pas les 10 biais de la couche de sortie": dense(784, 256) + 256 * 10}'''),
        ("f", WHOLE, "dense(784, 512) + dense(512, 256) + dense(256, 10)",
         r'''mistakes={"n'oublie pas les biais": 784 * 512 + 512 * 256 + 256 * 10,
          "n'oublie pas la couche de sortie": dense(784, 512) + dense(512, 256)}'''),
        ("g", WHOLE, "0", r'''mistakes={"Flatten ne fait que changer la forme : il n'a rien à apprendre": 784}'''),
    ]),
    Paper("23.4", "Une cross-entropy à la main : indice de classe ou one-hot", [
        ("a", DEC3, "P_234[1]",
         r'''decimals=3, mistakes={"la softmax prend des exponentielles, pas les logits eux-mêmes": 2 / 3,
          "la classe 1 est celle du logit 2 (les classes commencent à 0)": P_234[0]}'''),
        ("b", DEC3, "math.log(sum(math.exp(z) for z in Z_234))",
         r'''decimals=3, mistakes={"le logarithme népérien, pas décimal": math.log10(sum(math.exp(z) for z in Z_234)),
          "le logarithme de la somme, pas la somme": sum(math.exp(z) for z in Z_234)}'''),
        ("c", DEC3, "cross_entropy(Z_234, 1)",
         r'''decimals=3, mistakes={"L = −z_y + ln Σ e^z : il manque un terme": cross_entropy(Z_234, 1) + 2,
          "la classe 1 est celle du logit 2": cross_entropy(Z_234, 0),
          "le logarithme népérien, pas décimal": -math.log10(P_234[1])}'''),
        ("d", TF, "True", r'''mistakes={"avec la cible (0, 1, 0), un seul terme de la somme reste": False}'''),
        ("e", DEC3, "cross_entropy(P_234, 1)",
         r'''decimals=3, mistakes={"prends les probabilités de a) comme logits, puis recalcule −z₁ + ln Σ e^z": cross_entropy(Z_234, 1),
          "c'est la probabilité de la classe 1 après la seconde softmax ; la loss est son −ln": softmax(P_234)[1]}'''),
        ("f", DEC3, "(cross_entropy(Z_234, 1) + cross_entropy([0, 0, 0], 2)) / 2",
         r'''decimals=3, mistakes={"la moyenne du mini-batch, pas la somme": cross_entropy(Z_234, 1) + cross_entropy([0, 0, 0], 2),
          "la seconde image a trois logits égaux : sa loss est ln 3": cross_entropy(Z_234, 1) / 2,
          "la loss du mini-batch est la moyenne des deux images, pas celle de la première": cross_entropy(Z_234, 1),
          "la loss du mini-batch est la moyenne des deux images, pas celle de la seconde": math.log(3)}'''),
        ("g", DEC3, "math.log(10)",
         r'''decimals=3, mistakes={"le logarithme népérien, pas décimal": 1.0, "−ln(1/10), pas 1/10": 0.1}'''),
    ]),
    Paper("23.5", "Batches par epoch et nombre de mises à jour", [
        ("a", WHOLE, "math.ceil(N_235 / 256)",
         r'''mistakes={"arrondis au-dessus : le dernier mini-batch, incomplet, compte aussi": N_235 // 256,
          "retire d'abord les 6 000 images de validation": math.ceil(60_000 / 256)}'''),
        ("b", WHOLE, "N_235 // 256",
         r'''mistakes={"avec drop_last, le dernier mini-batch incomplet est laissé de côté": math.ceil(N_235 / 256)}'''),
        ("c", WHOLE, "N_235 - (N_235 // 256) * 256",
         r'''mistakes={"le dernier mini-batch contient le reste de la division, pas 256 images": 256}'''),
        ("d", WHOLE, "math.ceil(N_235 / 256) * 20",
         r'''mistakes={"sans drop_last, le dernier mini-batch incomplet compte aussi": (N_235 // 256) * 20,
          "n'oublie pas de retirer la validation": math.ceil(60_000 / 256) * 20}'''),
        ("e", WHOLE, "math.ceil(60_000 / 256)", r'''mistakes={"arrondis au-dessus": 60_000 // 256}'''),
        ("f", WHOLE, "math.ceil(12_000 / 64)", r'''mistakes={"arrondis au-dessus": 12_000 // 64}'''),
        ("g", WHOLE, "(N_235 // 256) * 256",
         r'''mistakes={"avec drop_last, le dernier mini-batch n'est pas vu": N_235}'''),
    ]),
    Paper("23.6", "Early stopping à la main : patience et min_delta", [
        ("a", WHOLE, "STOP_236A",
         r'''mistakes={"0,379 < 0,38 : l'epoch 7 est une amélioration": 8,
          "on s'arrête quand le compteur atteint la patience, pas quand il la dépasse": 15}'''),
        ("b", WHOLE, "BEST_236A",
         r'''mistakes={"on recharge les poids de la dernière amélioration": 10,
          "0,379 < 0,38 : la dernière amélioration est plus tardive": 5}'''),
        ("c", WHOLE, "STOP_236C",
         r'''mistakes={"avec min_delta = 0,005, 0,379 n'est pas sous 0,38 − 0,005": 10}'''),
        ("d", WHOLE, "BEST_236C",
         r'''mistakes={"0,379 n'est pas une amélioration avec min_delta = 0,005": 7,
          "on recharge les poids de la dernière amélioration": 8}'''),
        ("e", WHOLE, "STOP_236E",
         r'''mistakes={"relis la règle : une amélioration remet le compteur à 0": 12}'''),
        ("f", WHOLE, "BEST_236E",
         r'''mistakes={"on recharge les poids de la dernière amélioration": 16,
          "l'epoch 11 (0,37) bat 0,379": 7}'''),
        ("g", WHOLE, "STOP_236A",
         r'''mistakes={"une règle qui ne recharge rien garde les poids de la dernière epoch entraînée": 7}'''),
    ]),
    Paper("23.7", "Plannings de learning rate : le planning du livre avec LambdaLR", [
        ("a", DEC2, "1.0 * book_lambda(25)",
         r'''decimals=2, mistakes={"relis la formule λ(e) = max(0,1 ; 1 − 0,01 e)": 0.25,
          "pendant l'epoch 25, le planning a reçu 25 appels (epochs comptées depuis 0)": book_lambda(24)}'''),
        ("b", WHOLE, "min(e for e in range(200) if book_lambda(e) <= 0.1)",
         r'''mistakes={"les epochs se comptent à partir de 0 dans cet exercice": 91}'''),
        ("c", DEC4, "0.01 * book_lambda(50)",
         r'''decimals=4, mistakes={"LambdaLR multiplie λ par le learning rate de l'optimiseur": book_lambda(50),
          "pendant l'epoch 50, le planning a reçu 50 appels : λ(50)": 0.01 * book_lambda(49),
          "les epochs se comptent à partir de 0 : c'est λ(50) qu'il faut": 0.01 * book_lambda(51)}'''),
        ("d", WHOLE, "round(book_lambda(0) / 0.05)",
         r'''mistakes={"il faut 0,05 × λ(0) = 1": 1}'''),
        ("e", DEC4, "0.1 * 0.1 ** (65 // 30)",
         r'''decimals=4, mistakes={"⌊65/30⌋ = 2 paliers sont passés": 0.01, "deux paliers, pas trois": 0.0001}'''),
        ("f", DEC4, "0.1 * (1 + math.cos(math.pi * 25 / 100)) / 2",
         r'''decimals=4, mistakes={"le cosinus se calcule en radians": 0.1 * (1 + math.cos(math.radians(math.pi * 25 / 100))) / 2,
          "n'oublie pas de diviser 1 + cos par 2": 0.1 * (1 + math.cos(math.pi * 25 / 100))}'''),
        ("g", WHOLE, "sum(book_lambda(e) <= 0.1 for e in range(100))",
         r'''mistakes={"de 90 à 99 inclus": 9}'''),
    ]),
    Paper("23.8", "Fermi : durée d'une epoch et poids d'un checkpoint", [
        ("a", DEC1, "OPS_238 / 1e9",
         r'''decimals=1, mistakes={"6 P N : la passe avant et la passe arrière": P_238 * N_238 / 1e9,
          "la passe arrière coûte environ le double de la passe avant : 6 P N, pas 2 P N": 2 * P_238 * N_238 / 1e9}'''),
        ("b", DEC1, "SECONDS_238",
         r'''decimals=1, mistakes={"compte 6 P N opérations, pas P N": P_238 * N_238 / 50e9}'''),
        ("c", DEC1, "P_238 * 4 / 1e6",
         r'''decimals=1, mistakes={"4 octets par float32": P_238 / 1e6, "un float32 fait 4 octets, pas 8": P_238 * 8 / 1e6,
          "en mégaoctets de 10⁶ octets, pas en Mio de 2²⁰": P_238 * 4 / 2 ** 20}'''),
        ("d", DEC1, "3 * P_238 * 4 / 1e6",
         r'''decimals=1, mistakes={"les poids et les deux moments d'Adam : trois tenseurs de la taille des poids": 2 * P_238 * 4 / 1e6,
          "le checkpoint garde les deux moments d'Adam, pas les gradients": 4 * P_238 * 4 / 1e6,
          "ajoute les deux moments d'Adam aux poids": P_238 * 4 / 1e6}'''),
        ("e", WHOLE, "int(3600 // SECONDS_238)",
         r'''mistakes={"utilise la durée non arrondie de b)": 800}'''),
    ]),
]

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 23.1 à 23.7 et 🧮 23.8)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"E\"`) ; "
               "plusieurs lettres ou un ordre : entre guillemets (`\"EF\"`) ; une forme ou un triplet d'indices : "
               "entre guillemets (`\"(5, 7, 1)\"`) ; une liste : entre crochets (`[9, 9, 9]`). Arrondis comme "
               "l'énoncé le demande, et seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. "
               "Les questions « dans ta copie », la réflexion (🗣️ 23.9, ⚖️ 23.10, 📄 23.11) et l'entretien se "
               "corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 23.12 to 23.29)
# ---------------------------------------------------------------------------
TOOLS = r'''# Tools for the notebook exercises (parts A to D)
import json
import math
import pickle
import platform
import re
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from wb.nbtools import StopCheck, as_numpy, checked, filled, fr, pair, refuses, returned, same, shape_text, verdict, written

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:                       # a computer without PyTorch (Mac Intel): this chapter is done on Colab
    torch = None
    print("⏳ PyTorch absent sur cet ordinateur : ce chapitre se fait sur Colab (00_setup/INSTALL_LOCAL.md). Les "
          "cellules qui utilisent PyTorch affichent ⏳ au lieu de s'exécuter.")

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
CLASSES = wb.datasets.FASHION_CLASSES      # the 10 class names, in the order of the labels 0 to 9
WORK_DIR = Path(tempfile.mkdtemp(prefix="wb_ch23_"))   # a temporary folder for the files the exercises write
N_TRAIN = 12_000 if FAST_MODE else 54_000  # training images of the exercises that train for many epochs
# The data the checks use (23.13 asks you to load it yourself): Fashion-MNIST in the order of its files
X_TRAIN_ALL, Y_TRAIN_ALL = wb.datasets.load_fashion_mnist("train", seed=None)   # (60000, 28, 28) uint8, (60000,) int64
X_TEST, Y_TEST = wb.datasets.load_fashion_mnist("test", seed=None)               # (10000, 28, 28), (10000,)


def n_params(model):
    """The number of parameters (learned numbers) of a torch module."""
    return sum(p.numel() for p in model.parameters())


print(f"tools ready · torch {torch.__version__ if torch is not None else 'absent'} · device {DEVICE} · "
      f"{len(X_TRAIN_ALL)} training and {len(X_TEST)} test images · N_TRAIN = {N_TRAIN}")'''

# ---------------------------------------------------------------------------
# Part A: environment, data, preprocessing, the book's MLP (23.12 to 23.16)
# ---------------------------------------------------------------------------
GIVEN_12 = r'''IMAGES_12 = np.random.default_rng(2312).integers(0, 256, size=(4, 5, 6, 3), dtype=np.uint8)   # 4 RGB images, NHWC
print(IMAGES_12.shape, IMAGES_12.dtype)'''

TODO_12 = r'''def library_versions_12():
    """The versions of Python, NumPy, PyTorch and torchvision, as strings:
    {"python": "3.13.5", "numpy": "...", "torch": "...", "torchvision": "..."}."""
    raise NotImplementedError("library_versions_12")


def pick_device_12():
    """"cuda" if a CUDA GPU is available, else "mps" if an Apple GPU is available, else "cpu"."""
    raise NotImplementedError("pick_device_12")


def to_nchw_12(images):
    """A NumPy array of images (N, H, W, C), uint8 -> a float32 tensor (N, C, H, W), contiguous, with the same values
    (0 to 255). An array that does not have 4 axes raises ValueError."""
    raise NotImplementedError("to_nchw_12")


def caught_warnings_12(fn):
    """Call fn() and return the list of the class names of the warnings it emitted, in order (e.g. ["UserWarning"]),
    without displaying them; a warning emitted twice is listed twice."""
    raise NotImplementedError("caught_warnings_12")'''

SOLUTION_12 = r'''import torchvision


def library_versions_12():
    """The versions of Python, NumPy, PyTorch and torchvision, as strings:
    {"python": "3.13.5", "numpy": "...", "torch": "...", "torchvision": "..."}."""
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "torchvision": torchvision.__version__}


def pick_device_12():
    """"cuda" if a CUDA GPU is available, else "mps" if an Apple GPU is available, else "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_nchw_12(images):
    """A NumPy array of images (N, H, W, C), uint8 -> a float32 tensor (N, C, H, W), contiguous, with the same values
    (0 to 255). An array that does not have 4 axes raises ValueError."""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"expected 4 axes (N, H, W, C), got the shape {images.shape}")
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().contiguous()   # permute, never reshape


def caught_warnings_12(fn):
    """Call fn() and return the list of the class names of the warnings it emitted, in order (e.g. ["UserWarning"]),
    without displaying them; a warning emitted twice is listed twice."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")        # every warning, even one already shown in this session
        fn()
    return [w.category.__name__ for w in caught]'''

CHECK_12 = r'''with checked("23.12"):
    versions_12 = library_versions_12()
    if returned("23.12", "library_versions_12", versions_12):
        import torchvision
        want_12 = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                   "torchvision": torchvision.__version__}
        verdict("23.12", versions_12 == want_12, f"versions : {versions_12}.",
                f"attendu le dictionnaire {want_12}, reçu {versions_12!r:.200}.")
    device_12 = pick_device_12()
    want_device_12 = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    verdict("23.12", device_12 == want_device_12, f"device choisi : {device_12!r}.",
            f"sur cette machine, attendu {want_device_12!r}, reçu {device_12!r}.")
    x_12 = to_nchw_12(IMAGES_12)
    if returned("23.12", "to_nchw_12", x_12):
        verdict("23.12", isinstance(x_12, torch.Tensor) and x_12.dtype == torch.float32 and x_12.is_contiguous(),
                "to_nchw_12 rend un tenseur float32 contigu.",
                f"attendu un torch.Tensor float32 contigu, reçu {type(x_12).__name__} "
                f"{getattr(x_12, 'dtype', '')}" + ("" if not isinstance(x_12, torch.Tensor) or x_12.is_contiguous()
                                                    else " non contigu (.contiguous())") + ".")
        same("23.12", "to_nchw_12(IMAGES_12) contre les pixels de IMAGES_12 rangés en (N, C, H, W)", x_12,
             np.transpose(IMAGES_12, (0, 3, 1, 2)))
        wb.check("23.12a", float(x_12[1, 2, 3, 4]), computed=True)
    refuses("23.12", "to_nchw_12 sur une seule image (H, W, C)", lambda: to_nchw_12(IMAGES_12[0]))
    quiet_12 = caught_warnings_12(lambda: torch.zeros(3).sum())
    verdict("23.12", quiet_12 == [], "une fonction sans avertissement donne [].",
            f"attendu [] pour une fonction qui n'avertit de rien, reçu {quiet_12!r}.")
    caught_12 = caught_warnings_12(lambda: nn.MSELoss()(torch.zeros(4, 1), torch.zeros(4)))
    if returned("23.12", "caught_warnings_12", caught_12):
        wb.check("23.12b", " ".join(caught_12), computed=True)
    twice_12 = caught_warnings_12(lambda: [warnings.warn("old API", FutureWarning) for _ in range(2)])
    verdict("23.12", twice_12 == ["FutureWarning", "FutureWarning"], "un avertissement émis deux fois est compté deux fois.",
            f"deux warnings.warn(..., FutureWarning) : attendu ['FutureWarning', 'FutureWarning'], reçu {twice_12!r} "
            "(un filtre « always » les garde tous).")'''

RECORD_12 = r'''x_12 = to_nchw_12(IMAGES_12)
wb.record("23.12a", float(x_12[1, 2, 3, 4]), decimals=1, mistakes={
    "reshape garde l'ordre des valeurs en mémoire : pour changer l'ordre des axes, il faut permute": float(IMAGES_12.reshape(4, 3, 5, 6)[1, 2, 3, 4])})
wb.record("23.12b", "UserWarning", mistakes={
    "MSELoss avertit quand la cible n'a pas la forme de la prédiction : ta fonction l'attrape-t-elle ?": "",
    "le nom de la classe seulement, sans le module (w.category.__name__)": "builtins.UserWarning"})'''

GIVEN_13 = ""

TODO_13 = r'''X_train_13 = ...               # the 60 000 training images, in the order of the files...
y_train_13 = ...               # ... and their labels
X_test_13 = ...                # the 10 000 test images...
y_test_13 = ...                # ... and their labels
first_class_13 = ...           # a) the NAME of the class of the first training image (CLASSES gives the names)
per_class_13 = ...             # b) the number of training images of each class (they are all equal: one int)
darkest_13 = ...               # c) the NAME of the class whose training images have the smallest mean pixel value
brightest_13 = ...             # d) the NAME of the class whose training images have the largest mean pixel value


def show_grid_13(images, labels, per_class=8):
    """One row per class (10 rows), `per_class` images of that class in each row, the class name at the start of
    the row; ends with plt.show()."""
    raise NotImplementedError("show_grid_13")'''

SOLUTION_13 = r'''X_train_13, y_train_13 = wb.datasets.load_fashion_mnist("train", seed=None)
X_test_13, y_test_13 = wb.datasets.load_fashion_mnist("test", seed=None)
print(X_train_13.shape, X_train_13.dtype, X_train_13.min(), X_train_13.max(), y_train_13.dtype)
first_class_13 = CLASSES[y_train_13[0]]
per_class_13 = int(np.bincount(y_train_13)[0])            # np.bincount(y_train_13): 6000 for each of the 10 classes
means_13 = [X_train_13[y_train_13 == k].mean() for k in range(10)]
darkest_13 = CLASSES[int(np.argmin(means_13))]
brightest_13 = CLASSES[int(np.argmax(means_13))]


def show_grid_13(images, labels, per_class=8):
    """One row per class (10 rows), `per_class` images of that class in each row, the class name at the start of
    the row; ends with plt.show()."""
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 0.9, 10 * 0.95))
    for k in range(10):
        chosen = np.flatnonzero(labels == k)[:per_class]
        for j, ax in enumerate(axes[k]):
            ax.imshow(images[chosen[j]], cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[k, 0].set_ylabel(CLASSES[k], rotation=0, ha="right", va="center", fontsize=8)
    plt.show()'''

CHECK_13 = r'''if not filled(X_train_13, y_train_13, X_test_13, y_test_13):
    print("⏳ Ex 23.13 : charge d'abord les quatre tableaux.")
else:
    verdict("23.13", isinstance(X_train_13, np.ndarray) and X_train_13.shape == (60000, 28, 28)
            and X_train_13.dtype == np.uint8 and np.array_equal(X_train_13, X_TRAIN_ALL)
            and np.array_equal(y_train_13, Y_TRAIN_ALL) and np.array_equal(X_test_13, X_TEST),
            "les 60 000 images d'entraînement et les 10 000 de test, uint8, dans l'ordre des fichiers.",
            "attendu les images (60000, 28, 28) uint8 dans l'ordre des fichiers : load_fashion_mnist(..., seed=None).")
wb.check("23.13a", first_class_13)
wb.check("23.13b", per_class_13)
wb.check("23.13c", darkest_13)
wb.check("23.13d", brightest_13)
with checked("23.13"):
    show_grid_13(X_TRAIN_ALL, Y_TRAIN_ALL)'''

RECORD_13 = r'''wb.record("23.13a", first_class_13, mistakes={"le NOM de la classe, pas son label : CLASSES[label]": "9"})
wb.record("23.13b", per_class_13, mistakes={"le train compte 60 000 images, pas 70 000": 7000,
          "les images d'entraînement, pas celles du test": 1000})
means_13 = [X_TRAIN_ALL[Y_TRAIN_ALL == k].mean() for k in range(10)]
order_13 = np.argsort(means_13)
wb.record("23.13c", darkest_13, mistakes={"c) demande la classe la plus SOMBRE : np.argmin des moyennes": CLASSES[int(order_13[-1])],
          "compare toutes les classes : une autre a une moyenne encore plus petite": CLASSES[int(order_13[1])]})
wb.record("23.13d", brightest_13, mistakes={"d) demande la classe la plus CLAIRE : np.argmax des moyennes": CLASSES[int(order_13[0])],
          "compare toutes les classes : une autre a une moyenne encore plus grande": CLASSES[int(order_13[-2])]})'''

GIVEN_14 = r'''def train_variant_14(images_train, y_train, images_val, y_val, scale, center, epochs=3, seed=2314):
    """The book's MLP, Adam(1e-3), mini-batches of 256, trained on float32 pixels (x / scale - center[0]) / center[1]:
    returns the mean training loss of each epoch and the validation accuracy of each epoch."""
    def prepare(images):
        x = (images.astype(np.float32) / scale - center[0]) / center[1]
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(1)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 784), nn.ReLU(), nn.Linear(784, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(prepare(images_train), torch.from_numpy(y_train)), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    x_val, t_val = prepare(images_val), torch.from_numpy(y_val)
    losses, accuracies = [], []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        losses.append(total / len(loader.dataset))
        model.eval()
        with torch.inference_mode():
            accuracies.append((model(x_val).argmax(dim=1) == t_val).float().mean().item())
    return losses, accuracies'''

TODO_14 = r'''raw_acc_14 = ...         # a) pixels 0 to 255 (float32, NOT divided): the validation accuracy after 3 epochs is
                         #    "A" below 50 %, "B" from 50 to 70 %, "C" above 70 %
raw_nan_14 = ...         # b) pixels 0 to 255: does a loss become nan during the 3 epochs? True or False
raw_first_loss_14 = ...  # c) is the mean training loss of the 1st epoch with pixels 0 to 255 more than 10 times the one
                         #    with normalized pixels? True or False
gap_14 = ...             # d) after 3 epochs, does the accuracy of the normalized pixels beat the one of the pixels divided by
                         #    255 only by more than 3 points? True or False'''

SOLUTION_14 = r'''raw_acc_14, raw_nan_14, raw_first_loss_14, gap_14 = "C", False, True, False'''

EXPERIMENT_14 = r'''x_tr_14, y_tr_14 = X_TRAIN_ALL[:12_000], Y_TRAIN_ALL[:12_000]       # 12 000 images to train...
x_va_14, y_va_14 = X_TRAIN_ALL[-6_000:], Y_TRAIN_ALL[-6_000:]       # ... 6 000 others to validate (the same in both modes)
mu_14, sigma_14 = float((x_tr_14 / 255.0).mean()), float((x_tr_14 / 255.0).std())
results_14 = {"normalisés ((x/255 - μ)/σ)": train_variant_14(x_tr_14, y_tr_14, x_va_14, y_va_14, 255.0, (mu_14, sigma_14)),
              "divisés par 255 seulement": train_variant_14(x_tr_14, y_tr_14, x_va_14, y_va_14, 255.0, (0.0, 1.0)),
              "bruts, de 0 à 255 (en float32)": train_variant_14(x_tr_14, y_tr_14, x_va_14, y_va_14, 1.0, (0.0, 1.0))}
for name_14, (losses_14, accs_14) in results_14.items():
    print(f"{name_14:32s} loss d'entraînement par epoch {[round(v, 3) for v in losses_14]} · "
          f"accuracy de validation {[round(v, 3) for v in accs_14]}")
try:                                                                 # and the uint8 pixels, without any conversion
    nn.Linear(784, 10)(torch.from_numpy(x_tr_14[:2].reshape(2, -1)))
except RuntimeError as error_14:
    print("uint8 sans conversion :", type(error_14).__name__, "-", str(error_14).splitlines()[0])
wb.check("23.14a", raw_acc_14)
wb.check("23.14b", raw_nan_14)
wb.check("23.14c", raw_first_loss_14)
wb.check("23.14d", gap_14)'''

RECORD_14 = (r'''wb.record("23.14a", "C", mistakes=''' + "{" +
             '"Adam règle la taille de chaque pas : l\'entraînement survit-il à des entrées 255 fois trop grandes ?": "A", '
             '"regarde l\'accuracy de la 3e epoch, pas celle de la 1re : où en est-on après trois passages ?": "B"' + "}" + r''')
wb.record("23.14b", False, mistakes={"une loss nan vient d'un calcul qui déborde : avec Adam, les pas restent-ils de la taille du learning rate ?": True})
wb.record("23.14c", True, mistakes={"compare les deux premières valeurs de loss d'entraînement : quel est leur rapport ?": False})
wb.record("23.14d", False, mistakes={"l'écart existe, mais combien de points mesure-t-il après 3 epochs ?": True})''')

TODO_15 = r'''def split_15(X, y, val_size=0.1, seed=0):
    """(X_tr, X_val, y_tr, y_val): a STRATIFIED split (the same share of each class on both sides), reproducible with
    `seed`; val_size is the share of the validation."""
    raise NotImplementedError("split_15")


def fit_stats_15(images):
    """(mean, std): the mean and the standard deviation (divisor n) of the pixels divided by 255, as Python floats."""
    raise NotImplementedError("fit_stats_15")


def preprocess_15(images, mean, std):
    """uint8 images (N, 28, 28) -> a float32 tensor (N, 1, 28, 28): (x / 255 - mean) / std. The input is not modified,
    and the result is float32 even when mean and std are NumPy float64 numbers."""
    raise NotImplementedError("preprocess_15")


def make_loaders_15(X_tr, y_tr, X_val, y_val, mean, std, batch_size=256, seed=0):
    """(train_loader, val_loader): mini-batches of (images (B, 1, 28, 28) float32, labels (B,) int64). The training
    loader shuffles at each epoch with its own torch.Generator seeded by `seed`; the validation loader does not
    shuffle and uses mini-batches of 1024."""
    raise NotImplementedError("make_loaders_15")'''

SOLUTION_15 = r'''def split_15(X, y, val_size=0.1, seed=0):
    """(X_tr, X_val, y_tr, y_val): a STRATIFIED split (the same share of each class on both sides), reproducible with
    `seed`; val_size is the share of the validation."""
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed)


def fit_stats_15(images):
    """(mean, std): the mean and the standard deviation (divisor n) of the pixels divided by 255, as Python floats."""
    x = np.asarray(images) / 255.0
    return float(x.mean()), float(x.std())


def preprocess_15(images, mean, std):
    """uint8 images (N, 28, 28) -> a float32 tensor (N, 1, 28, 28): (x / 255 - mean) / std. The input is not modified,
    and the result is float32 even when mean and std are NumPy float64 numbers."""
    x = (np.asarray(images, dtype=np.float32) / 255.0 - np.float32(mean)) / np.float32(std)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1)          # a new array: the input is untouched


def make_loaders_15(X_tr, y_tr, X_val, y_val, mean, std, batch_size=256, seed=0):
    """(train_loader, val_loader): mini-batches of (images (B, 1, 28, 28) float32, labels (B,) int64). The training
    loader shuffles at each epoch with its own torch.Generator seeded by `seed`; the validation loader does not
    shuffle and uses mini-batches of 1024."""
    train_ds = TensorDataset(preprocess_15(X_tr, mean, std), torch.as_tensor(np.asarray(y_tr), dtype=torch.int64))
    val_ds = TensorDataset(preprocess_15(X_val, mean, std), torch.as_tensor(np.asarray(y_val), dtype=torch.int64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=1024)
    return train_loader, val_loader'''

CHECK_15 = r'''with checked("23.15"):
    X_tr_15, X_val_15, y_tr_15, y_val_15 = pair("23.15", "split_15(X_TRAIN_ALL, Y_TRAIN_ALL)",
                                                split_15(X_TRAIN_ALL, Y_TRAIN_ALL),
                                                names=("X_tr", "X_val", "y_tr", "y_val"))
    verdict("23.15", len(X_tr_15) == 54000 and len(X_val_15) == 6000 and len(y_val_15) == 6000,
            "54 000 images d'entraînement et 6 000 de validation.",
            f"attendu 54 000 + 6 000 images, reçu {len(X_tr_15)} + {len(X_val_15)}.")
    want_val_15 = train_test_split(X_TRAIN_ALL, Y_TRAIN_ALL, test_size=0.1, stratify=Y_TRAIN_ALL, random_state=0)[3]
    verdict("23.15", np.array_equal(np.asarray(y_val_15), want_val_15),
            "c'est le découpage de train_test_split(..., stratify=y, random_state=0).",
            "ton découpage n'est pas celui de train_test_split(X, y, test_size=val_size, stratify=y, random_state=seed) "
            "que demande le contrat : les statistiques b) et c) en dépendent.")
    again_15 = split_15(X_TRAIN_ALL, Y_TRAIN_ALL)
    verdict("23.15", np.array_equal(again_15[3], y_val_15), "le découpage est reproductible (même graine, mêmes images).",
            "deux appels avec la même graine doivent donner le même découpage (random_state).")
    wb.check("23.15a", int(np.sum(np.asarray(y_val_15) == 6)), computed=True)
    mean_15, std_15 = pair("23.15", "fit_stats_15(X_tr)", fit_stats_15(X_tr_15), names=("mean", "std"),
                           floats=("mean", "std"))
    wb.check("23.15b", mean_15, computed=True)
    wb.check("23.15c", std_15, computed=True)
    copy_15 = X_tr_15[:500].copy()
    x_15 = preprocess_15(X_tr_15[:500], np.float64(mean_15), np.float64(std_15))   # NumPy float64 statistics on purpose
    if returned("23.15", "preprocess_15", x_15):
        verdict("23.15", isinstance(x_15, torch.Tensor) and x_15.dtype == torch.float32 and tuple(x_15.shape) == (500, 1, 28, 28),
                "preprocess_15 rend un tenseur float32 (500, 1, 28, 28), même avec des statistiques float64.",
                f"attendu un tenseur float32 de forme (500, 1, 28, 28), reçu {type(x_15).__name__} "
                f"{getattr(x_15, 'dtype', '')} {tuple(getattr(x_15, 'shape', ()))} : sous NumPy 2, un tableau float32 "
                "moins un scalaire NumPy float64 devient float64.")
        same("23.15", "preprocess_15 contre (x / 255 - mean) / std", x_15[:, 0],
             (X_tr_15[:500] / 255.0 - mean_15) / std_15, rtol=1e-5, atol=1e-5)
        verdict("23.15", np.array_equal(copy_15, X_tr_15[:500]), "les images d'entrée ne sont pas modifiées.",
                "preprocess_15 a modifié les images qu'on lui a données : travaille sur une copie.")
    train_15, val_15 = pair("23.15", "make_loaders_15(...)", make_loaders_15(X_tr_15, y_tr_15, X_val_15, y_val_15,
                                                                                mean_15, std_15), names=("train_loader", "val_loader"))
    xb_15, yb_15 = next(iter(train_15))
    verdict("23.15", xb_15.dtype == torch.float32 and yb_15.dtype == torch.int64 and tuple(xb_15.shape) == (256, 1, 28, 28),
            "un mini-batch d'entraînement : images float32 (256, 1, 28, 28) et labels int64.",
            f"attendu des images float32 (256, 1, 28, 28) et des labels int64, reçu {xb_15.dtype} {tuple(xb_15.shape)} et {yb_15.dtype}.")
    first_epoch_15, second_epoch_15 = next(iter(train_15))[1], next(iter(train_15))[1]
    same_seed_15 = next(iter(make_loaders_15(X_tr_15, y_tr_15, X_val_15, y_val_15, mean_15, std_15)[0]))[1]
    verdict("23.15", torch.equal(yb_15, same_seed_15) and not torch.equal(first_epoch_15, second_epoch_15),
            "le mélange est reproductible d'un appel à l'autre, et change d'une epoch à la suivante.",
            "le loader d'entraînement doit mélanger à chaque epoch, avec son propre générateur à graine (generator=...).")
    xv_15, yv_15 = next(iter(val_15))
    verdict("23.15", len(yv_15) == 1024 and torch.equal(yv_15, torch.as_tensor(np.asarray(y_val_15[:1024]))),
            "la validation n'est pas mélangée, par mini-batches de 1024.",
            "le loader de validation ne mélange pas (shuffle=False) et prend des mini-batches de 1024.")'''

RECORD_15 = r'''X_tr_15, X_val_15, y_tr_15, y_val_15 = split_15(X_TRAIN_ALL, Y_TRAIN_ALL)
plain_15 = train_test_split(X_TRAIN_ALL, Y_TRAIN_ALL, test_size=0.1, random_state=0)[3]
wb.record("23.15a", int(np.sum(y_val_15 == 6)), mistakes={
    "le découpage doit être stratifié : stratify=y garde 10 % de chaque classe": int(np.sum(plain_15 == 6))})
mean_15, std_15 = fit_stats_15(X_tr_15)
mean_all_15, std_all_15 = fit_stats_15(X_TRAIN_ALL)
wb.record("23.15b", mean_15, decimals=4, mistakes={"les pixels divisés par 255, pas de 0 à 255": mean_15 * 255})
wb.record("23.15c", std_15, decimals=4, mistakes={"les pixels divisés par 255, pas de 0 à 255": std_15 * 255,
          "l'écart-type, pas la variance": std_15 ** 2})'''

TODO_16 = r'''def make_mlp_16(n_hidden=784, n_classes=10):
    """The book's MLP: nn.Flatten, nn.Linear(784, n_hidden), nn.ReLU, nn.Linear(n_hidden, n_classes), in an
    nn.Sequential; it returns logits (no softmax)."""
    raise NotImplementedError("make_mlp_16")


def make_lazy_mlp_16(n_hidden=784, n_classes=10):
    """The same MLP with nn.LazyLinear layers: no input size to write."""
    raise NotImplementedError("make_lazy_mlp_16")


def count_params_16(model):
    """The number of learned numbers (parameters) of a torch module."""
    raise NotImplementedError("count_params_16")'''

SOLUTION_16 = r'''def make_mlp_16(n_hidden=784, n_classes=10):
    """The book's MLP: nn.Flatten, nn.Linear(784, n_hidden), nn.ReLU, nn.Linear(n_hidden, n_classes), in an
    nn.Sequential; it returns logits (no softmax)."""
    return nn.Sequential(nn.Flatten(), nn.Linear(784, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_classes))


def make_lazy_mlp_16(n_hidden=784, n_classes=10):
    """The same MLP with nn.LazyLinear layers: no input size to write."""
    return nn.Sequential(nn.Flatten(), nn.LazyLinear(n_hidden), nn.ReLU(), nn.LazyLinear(n_classes))


def count_params_16(model):
    """The number of learned numbers (parameters) of a torch module."""
    return sum(p.numel() for p in model.parameters())'''

CHECK_16 = r'''with checked("23.16"):
    torch.manual_seed(2316)
    mlp_16 = make_mlp_16()
    if returned("23.16", "make_mlp_16", mlp_16):
        print(mlp_16)
        kinds_16 = [type(m).__name__ for m in mlp_16.children()] if isinstance(mlp_16, nn.Module) else []
        verdict("23.16", kinds_16 == ["Flatten", "Linear", "ReLU", "Linear"],
                "make_mlp_16 : Flatten, Linear, ReLU, Linear, sans softmax.",
                f"attendu les couches Flatten, Linear, ReLU, Linear (sans softmax), reçu {kinds_16}.")
        verdict("23.16", count_params_16(mlp_16) == n_params(mlp_16) == 623_290,
                "count_params_16 retrouve les 623 290 paramètres du livre.",
                f"count_params_16 rend {count_params_16(mlp_16)} : attendu {n_params(mlp_16)} (le livre en compte 623 290).")
        wb.check("23.16a", count_params_16(make_mlp_16(n_hidden=100)), computed=True)
        with torch.inference_mode():
            out_16 = mlp_16(torch.zeros(7, 1, 28, 28))
        wb.check("23.16b", shape_text(tuple(out_16.shape)), computed=True)
        x_16 = (torch.from_numpy(X_TRAIN_ALL[:2000]).float() / 255.0 - 0.286) / 0.353
        with torch.inference_mode():
            start_16 = nn.CrossEntropyLoss()(mlp_16(x_16.unsqueeze(1)), torch.from_numpy(Y_TRAIN_ALL[:2000])).item()
        verdict("23.16", abs(start_16 - math.log(10)) < 0.3,
                f"loss du modèle non entraîné : {fr(start_16)}, proche de ln 10 ≈ 2,303 : le modèle part sans préférence.",
                f"loss du modèle non entraîné : {fr(start_16)}, loin de ln 10 ≈ 2,303 : des poids initialisés à la main, ou des entrées mal mises à l'échelle ?")
    lazy_16 = make_lazy_mlp_16()
    if returned("23.16", "make_lazy_mlp_16", lazy_16):
        lazy_layers_16 = [m for m in lazy_16.modules() if isinstance(m, nn.modules.lazy.LazyModuleMixin)]
        verdict("23.16", len(lazy_layers_16) == 2, "deux couches nn.LazyLinear avant le premier passage.",
                f"attendu deux couches nn.LazyLinear, reçu {len(lazy_layers_16)} couche(s) paresseuse(s).")
        with torch.inference_mode():
            lazy_16(torch.zeros(1, 1, 28, 28))                 # the first pass fixes the input sizes
        verdict("23.16", n_params(lazy_16) == 623_290, "après un premier passage, le modèle paresseux a aussi 623 290 paramètres.",
                f"après un premier passage, attendu 623 290 paramètres, reçu {n_params(lazy_16)}.")'''

RECORD_16 = r'''wb.record("23.16a", count_params_16(make_mlp_16(n_hidden=100)), mistakes={
    "n'oublie pas les biais : (n_in + 1) × n_out par couche": 784 * 100 + 100 * 10,
    "la couche de sortie reçoit 100 valeurs, pas 784": (784 + 1) * 100 + (784 + 1) * 10})
wb.record("23.16b", "(7, 10)", mistakes={"Flatten garde l'axe des exemples : une ligne de 10 logits par image": "(70,)",
          "la sortie a 10 logits par image, sans axe de canal": "(7, 1, 10)"})'''

PART_A = Part("A", "L'environnement, les données, le prétraitement et le MLP du livre",
              "Tu vérifies ton environnement et manipules des tenseurs d'images (23.12) ; tu charges et regardes "
              "Fashion-MNIST (23.13) ; tu prédis ce qui arrive quand on oublie la normalisation (23.14) ; tu réunis "
              "tout le prétraitement en quatre fonctions (23.15) ; puis tu construis le MLP du livre (23.16).\n\n"
              "La cellule d'outils importe PyTorch, choisit le device, et charge Fashion-MNIST dans l'ordre des "
              "fichiers (`X_TRAIN_ALL`, `Y_TRAIN_ALL`, `X_TEST`, `Y_TEST` : les vérifications s'en servent). "
              "`N_TRAIN` est le nombre d'images d'entraînement des exercices qui entraînent longtemps (12 000 en mode "
              "rapide, toutes sinon). Les fichiers que les exercices écrivent vont dans un dossier temporaire, "
              "`WORK_DIR`.",
              given=TOOLS, exercises=[
    Ex("23.12", "📦", 2, 20, "Environnement et tenseurs : versions, device, reshape, view, permute",
       "vérifier les versions et le device, convertir des images de NHWC à NCHW, et attraper les avertissements.",
       "ch. 20 (tenseurs) · fiche §23.1.3, §23.2.1, §23.3.2 à §23.3.5, §23.5.1", thread="Fashion-MNIST", tracks="R, C",
       body=r"""Quatre petites fonctions qu'on retrouve en tête de tout projet. `IMAGES_12` contient 4 images en couleur de 5 lignes et 6 colonnes, au format channels_last `(N, H, W, C)`, en `uint8`.

**Contrat**
- `library_versions_12()` rend un dictionnaire `{"python": ..., "numpy": ..., "torch": ..., "torchvision": ...}` : les versions, en chaînes, comme les écrit chaque bibliothèque (Python : `platform.python_version()`) ;
- `pick_device_12()` rend `"cuda"` si un GPU NVIDIA (ou AMD) est disponible, sinon `"mps"` si un GPU d'Apple l'est, sinon `"cpu"` ;
- `to_nchw_12(images)` transforme un tableau NumPy `(N, H, W, C)` en tenseur `float32` `(N, C, H, W)`, **contigu**, avec les mêmes valeurs (de 0 à 255) ; un tableau qui n'a pas 4 axes lève `ValueError` ;
- `caught_warnings_12(fn)` appelle `fn()` et rend la liste des noms des classes des avertissements émis (par exemple `["UserWarning"]`), dans l'ordre, **sans les afficher** ; un avertissement émis deux fois figure deux fois.

La vérification compare tes résultats à ceux de la machine, contrôle la conversion pixel par pixel, puis vérifie deux valeurs : a) `to_nchw_12(IMAGES_12)[1, 2, 3, 4]` ; b) les avertissements émis par `nn.MSELoss()(torch.zeros(4, 1), torch.zeros(4))`, une prédiction `(4, 1)` comparée à une cible `(4,)`.

Dans tes notes : pourquoi `x.permute(0, 3, 1, 2).view(4, -1)` lève-t-il une erreur alors que `reshape` passe ? Que dit l'avertissement de b), et quel calcul faux aurait-il évité ?""",
       given=GIVEN_12, todo=TODO_12, check=CHECK_12, solution=SOLUTION_12 + "\n\n\n" + CHECK_12, record=RECORD_12,
       note="`permute` ne déplace rien : le tenseur rendu lit la même mémoire dans un autre ordre, il n'est plus "
            "contigu, et `view` refuse ; `.contiguous()` (ou `reshape`) recopie les valeurs dans le nouvel ordre. "
            "L'avertissement de b) prévient que la cible `(4,)` va être diffusée contre la prédiction `(4, 1)` : "
            "la loss comparerait chaque prédiction à **toutes** les cibles, une matrice `(4, 4)`, sans erreur."),

    Ex("23.13", "📦", 2, 20, "Charger et regarder Fashion-MNIST",
       "charger un dataset d'images, vérifier sa structure et regarder ses images classe par classe.",
       "Ex 23.12 · fiche §23.4.1, §23.5.2, §23.5.3", thread="Fashion-MNIST", tracks="R, C",
       body=r"""Avant tout entraînement, on charge et on regarde (fiche §23.5.3).

**Contrat**
- `X_train_13, y_train_13` et `X_test_13, y_test_13` : les images et les labels de Fashion-MNIST, dans l'ordre des fichiers (`wb.datasets.load_fashion_mnist(split, seed=None)` ; `seed=None` garde cet ordre) ;
- a) `first_class_13` : le **nom** de la classe de la première image d'entraînement (`CLASSES` donne les noms dans l'ordre des labels) ;
- b) `per_class_13` : le nombre d'images d'entraînement de chaque classe (elles en ont toutes autant : un entier) ;
- c) `darkest_13` et d) `brightest_13` : le nom de la classe dont les images d'entraînement ont la plus petite, puis la plus grande valeur moyenne des pixels ;
- `show_grid_13(images, labels, per_class=8)` affiche une grille de 10 lignes, une par classe, avec `per_class` images de la classe et son nom en début de ligne, et finit par `plt.show()`.

Affiche aussi, pour toi, la forme, le type, le minimum et le maximum des images. La vérification contrôle tes tableaux, tes quatre réponses, puis appelle ta grille sur les images d'entraînement.

Dans tes notes : quelles classes se ressemblent le plus, à l'œil ? Pourquoi les sandales sont-elles les images les plus sombres ? Relève un problème de **contenu** et un problème de **structure** que ces contrôles auraient pu révéler sur un autre dataset.""",
       todo=TODO_13, check=CHECK_13, solution=SOLUTION_13 + "\n\n\n" + CHECK_13, record=RECORD_13,
       note="La première image d'entraînement est une bottine (*Ankle boot*), et chaque classe a 6 000 images "
            "d'entraînement : le dataset est équilibré, l'accuracy est donc une mesure raisonnable. Les sandales "
            "sont les plus sombres en moyenne (de fines lanières sur un fond noir : peu de pixels allumés), les "
            "manteaux les plus clairs. Les classes du haut du corps (T-shirt, pull, manteau, chemise) se "
            "ressemblent : elles feront l'essentiel des confusions (23.20)."),

    Ex("23.14", "🔮", 1, 10, "Oublier la conversion en float ou la normalisation : erreur ou entraînement lent ?",
       "prédire ce que coûte, en chiffres, un prétraitement oublié.",
       "Ex 23.13 · fiche §23.5.5, §23.5.6", thread="Fashion-MNIST", tracks="C", hypothesis=True,
       body=r"""La fiche le dit : des pixels `uint8` donnés à une couche dense lèvent une erreur de type, et la normalisation aide l'optimisation. Mais de combien ? On entraîne trois fois le MLP du livre, 3 epochs avec Adam (learning rate 0,001, mini-batches de 256), sur les mêmes 12 000 images, avec des pixels en `float32` : (1) normalisés, $(x/255 - \mu)/\sigma$ ; (2) divisés par 255 seulement ; (3) **bruts**, de 0 à 255, sans aucune division. La fonction `train_variant_14`, donnée ci-dessous, rend la loss d'entraînement moyenne de chaque epoch et l'accuracy de validation après chaque epoch.

Prédis, **avant** d'exécuter :
a) `raw_acc_14` : avec les pixels bruts, l'accuracy de validation après 3 epochs est `"A"` sous 50 %, `"B"` entre 50 et 70 %, `"C"` au-dessus de 70 % ;
b) `raw_nan_14` : avec les pixels bruts, une loss devient-elle `nan` pendant les 3 epochs ? (`True` ou `False`) ;
c) `raw_first_loss_14` : la loss d'entraînement moyenne de la 1ʳᵉ epoch avec les pixels bruts est-elle plus de 10 fois celle des pixels normalisés ? (`True` ou `False`) ;
d) `gap_14` : après 3 epochs, l'accuracy des pixels normalisés dépasse-t-elle celle des pixels divisés par 255 de **plus de 3 points** ? (`True` ou `False`).

L'expérience affiche les trois entraînements, puis essaie les pixels `uint8` sans conversion.""",
       given=GIVEN_14, todo=TODO_14, solution=SOLUTION_14,
       after=[("code", guarded(EXPERIMENT_14, ["raw_acc_14", "raw_nan_14", "raw_first_loss_14", "gap_14"],
                               "⏳ Ex 23.14 : écris d'abord tes quatre prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Pourquoi Adam "
                     "encaisse-t-il des entrées 255 fois trop grandes, et que se passerait-il avec SGD et un learning "
                     "rate de 0,1 ? Si l'écart de d) est modeste, pourquoi normalise-t-on quand même toujours ?")],
       record=RECORD_14,
       note="Avec des pixels bruts, la première epoch est catastrophique (une loss moyenne d'une vingtaine de "
            "nats, plus de trente fois celle des pixels normalisés : les premiers logits sont énormes), mais rien ne "
            "plante et aucune loss ne devient `nan` : Adam divise chaque pas par une moyenne de la taille des "
            "gradients récents, si bien que ses pas restent de l'ordre du learning rate même quand les gradients "
            "sont 255 fois trop grands. Après 3 epochs, le réseau dépasse déjà 75 %, derrière les deux autres "
            "versions. La normalisation complète devance la simple division par 255 de quelques points après la "
            "première epoch, mais d'à peine un point après la troisième : ce MLP est petit et Adam compense. On normalise quand même toujours : avec SGD, un learning rate "
            "réglé pour des entrées de taille 1 diverge sur des entrées 255 fois plus grandes, et les réseaux "
            "profonds sont bien plus sensibles à l'échelle de leurs entrées."),

    Ex("23.15", "📦", 2, 30, "Tout le prétraitement au même endroit : split stratifié, float32, normalisation, DataLoader",
       "écrire le prétraitement d'un projet une seule fois, en quatre fonctions réutilisables.",
       "Ex 23.13 · ch. 8 (stratification) · ch. 12 (statistiques apprises sur le train) · fiche §23.5.4 à §23.5.8",
       thread="Fashion-MNIST", tracks="R, C",
       body=r"""Le livre réunit tout son prétraitement dans un seul listing (fiche §23.5.8). Tu l'écris en quatre fonctions, que les exercices suivants réutilisent.

**Contrat**
- `split_15(X, y, val_size=0.1, seed=0)` rend `(X_tr, X_val, y_tr, y_val)` : le découpage **stratifié** (la même part de chaque classe des deux côtés) de `train_test_split` de scikit-learn, avec `random_state=seed` : les parties suivantes reprennent exactement ce découpage ;
- `fit_stats_15(images)` rend `(mean, std)` : la moyenne et l'écart-type (diviseur $n$) des pixels **divisés par 255**, en `float` Python ;
- `preprocess_15(images, mean, std)` transforme des images `uint8` `(N, 28, 28)` en tenseur `float32` `(N, 1, 28, 28)`, $(x/255 - \text{mean})/\text{std}$ ; il ne modifie pas son entrée et rend du `float32` **même si** `mean` et `std` sont des nombres NumPy `float64` ;
- `make_loaders_15(X_tr, y_tr, X_val, y_val, mean, std, batch_size=256, seed=0)` rend `(train_loader, val_loader)`, dont les mini-batches sont des couples (images `float32` `(B, 1, 28, 28)`, labels `int64` `(B,)`) ; le loader d'entraînement mélange à chaque epoch avec son propre `torch.Generator` initialisé par `seed` ; celui de validation ne mélange pas et prend des mini-batches de 1024.

La vérification découpe les 60 000 images d'entraînement, puis contrôle : a) le nombre d'images de la classe 6 (*Shirt*) dans ta validation ; b) et c) la moyenne et l'écart-type que rend `fit_stats_15` sur ta partie d'entraînement ; le type et la forme de ton prétraitement, appelé exprès avec des statistiques `np.float64` ; le mélange de tes loaders.

Dans tes notes : pourquoi calcule-t-on `mean` et `std` sur `X_tr` seulement ? Pourquoi un générateur propre au loader plutôt que la graine globale de `torch.manual_seed` ?""",
       todo=TODO_15, check=CHECK_15, solution=SOLUTION_15 + "\n\n\n" + CHECK_15, record=RECORD_15,
       note="Sous NumPy 2, un tableau `float32` combiné à un **scalaire NumPy** `float64` (ce que rend `.mean()`) "
            "devient `float64`, et `nn.Linear` refuserait ce tenseur : on convertit les statistiques en `float` "
            "Python (ou en `np.float32`) et l'on rend explicitement du `float32`. Les statistiques viennent du "
            "train seul : la validation simule des données jamais vues. Un générateur propre au loader rend "
            "l'ordre des mini-batches indépendant de tout ce qui a tiré des nombres avant (l'initialisation du "
            "modèle, un dropout) : changer le modèle ne change pas l'ordre des données."),

    Ex("23.16", "📦", 1, 15, "Le MLP du livre avec nn.Sequential et nn.Flatten",
       "construire le MLP du livre en PyTorch et compter ses paramètres.",
       "Ex 23.15 · ✏️ 23.3 · fiche §23.6 à §23.6.2", thread="Fashion-MNIST", tracks="R, C",
       body=r"""Le modèle du livre, traduit (fiche §23.6.2).

**Contrat**
- `make_mlp_16(n_hidden=784, n_classes=10)` rend un `nn.Sequential` : `nn.Flatten`, `nn.Linear(784, n_hidden)`, `nn.ReLU`, `nn.Linear(n_hidden, n_classes)` ; il rend des **logits** (pas de softmax) ;
- `make_lazy_mlp_16(n_hidden=784, n_classes=10)` rend le même réseau avec des `nn.LazyLinear`, qui découvrent leur taille d'entrée au premier passage ;
- `count_params_16(model)` rend le nombre de paramètres d'un module.

La vérification affiche ton modèle (`print`), compare ton compte à celui du livre, puis vérifie : a) le nombre de paramètres de `make_mlp_16(n_hidden=100)` ; b) la forme de la sortie pour un mini-batch de 7 images `(7, 1, 28, 28)`, en chaîne (`"(3, 5, 1)"`). Elle mesure aussi la loss de ton modèle **non entraîné**, et regarde ton modèle paresseux avant et après un premier passage.

Dans tes notes : pourquoi la loss d'un modèle non entraîné est-elle proche de $\ln 10$ ? Quel avantage et quel inconvénient vois-tu à `nn.LazyLinear` ?""",
       todo=TODO_16, check=CHECK_16, solution=SOLUTION_16 + "\n\n\n" + CHECK_16, record=RECORD_16,
       note="À l'initialisation, les logits sont petits et presque égaux : la softmax donne environ 1/10 à chaque "
            "classe, et la loss vaut à peu près $-\\ln(1/10) = \\ln 10 \\approx 2{,}30$ ; une première loss très "
            "différente signale un problème, comme des entrées non normalisées (une softmax en trop, elle, ne se voit "
            "pas au départ, mais dans une loss qui ne descend presque plus, fiche §23.6.3). "
            "`nn.LazyLinear` évite de calculer les tailles d'entrée (utile après des convolutions), mais le modèle "
            "n'a pas de paramètres avant le premier passage : on ne peut ni les compter, ni créer l'optimiseur, ni "
            "charger un `state_dict` avant d'y avoir fait passer un exemple."),
])

# ---------------------------------------------------------------------------
# Part B: inference bugs, fit(), the history, the errors (23.17 to 23.20)
# ---------------------------------------------------------------------------
PIPELINE = r'''# The pipeline of parts B to D (exercises 23.15 and 23.16, written once for all)
X_TR, X_VAL, Y_TR, Y_VAL = train_test_split(X_TRAIN_ALL, Y_TRAIN_ALL, test_size=0.1, stratify=Y_TRAIN_ALL, random_state=0)
MEAN, STD = float((X_TR / 255.0).mean()), float((X_TR / 255.0).std())   # learned on the 54 000 training images only


def preprocess(images, mean=MEAN, std=STD):
    """uint8 images (N, 28, 28) -> float32 tensor (N, 1, 28, 28), normalized with the TRAIN statistics."""
    x = (np.asarray(images, dtype=np.float32) / 255.0 - np.float32(mean)) / np.float32(std)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1)


X_VAL_T, Y_VAL_T = preprocess(X_VAL), torch.from_numpy(Y_VAL)


def make_loaders(n_train=N_TRAIN, batch_size=256, seed=0):
    """(train_loader, val_loader): the first n_train training images, shuffled with their own generator; the 6 000
    validation images, in order, by 1024."""
    train = TensorDataset(preprocess(X_TR[:n_train]), torch.from_numpy(Y_TR[:n_train]))
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    return loader, DataLoader(TensorDataset(X_VAL_T, Y_VAL_T), batch_size=1024)


def make_mlp(n_hidden=784, n_classes=10, dropout=0.0):
    """The book's MLP (a dropout after the ReLU if dropout > 0): it returns logits."""
    layers = [nn.Flatten(), nn.Linear(784, n_hidden), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers, nn.Linear(n_hidden, n_classes))


print(f"{len(X_TR)} training images (the exercises use the first {N_TRAIN}), {len(X_VAL)} validation images · "
      f"mean {MEAN:.4f}, std {STD:.4f}")'''

GIVEN_17 = r'''torch.manual_seed(2317)
model_17 = make_mlp(dropout=0.5)                     # a model with dropout, in training mode (the default)
torch.save(model_17.state_dict(), WORK_DIR / "mlp_17.pt")
TRAP_17 = {"ran": False}


def trap_17(message):
    """What a booby-trapped file could run while it is loaded (here: harmless, it only raises a flag)."""
    TRAP_17["ran"] = True
    print(f"⚠️ {message} : ce code vient de s'exécuter pendant le chargement du fichier !")
    return {}


class Trap17:
    def __reduce__(self):                            # pickle: "to rebuild me, call trap_17(...)"
        return (trap_17, ("code caché dans un fichier de poids",))


torch.save({"weight": torch.zeros(2), "extra": Trap17()}, WORK_DIR / "downloaded_17.pt")   # a file "found online"
print("model_17 ready (training mode) · files:", sorted(p.name for p in WORK_DIR.glob("*_17.pt")))'''

TODO_17 = r'''# Fix the THREE bugs of these two functions in place (same signatures).
def predict_17(model, images):
    """Predicted classes (N,) int64 of uint8 images (N, 28, 28): the model in EVALUATION mode, computed inside
    torch.inference_mode() (no autograd graph)."""
    x = preprocess(images)
    logits = model(x)
    return logits.argmax(dim=1)


def load_weights_17(path):
    """The state_dict saved at `path`, loaded SAFELY: tensors and simple types only, no code executed."""
    return torch.load(path, weights_only=False)'''

SOLUTION_17 = r'''def predict_17(model, images):
    """Predicted classes (N,) int64 of uint8 images (N, 28, 28): the model in EVALUATION mode, computed inside
    torch.inference_mode() (no autograd graph)."""
    model.eval()                                  # bug 1: dropout off (and batchnorm on its running statistics)
    with torch.inference_mode():                  # bug 2: no graph, no gradient
        return model(preprocess(images)).argmax(dim=1)


def load_weights_17(path):
    """The state_dict saved at `path`, loaded SAFELY: tensors and simple types only, no code executed."""
    return torch.load(path, weights_only=True)   # bug 3: never unpickle arbitrary objects'''

CHECK_17 = r'''def point_17(k, ok, success, failure):
    print(f"{'✅' if ok else '❌'} Ex 23.17 · point {k} : {success if ok else failure}")
    return ok


with checked("23.17"):
    images_17 = X_VAL[:200]
    model_17.eval()
    with torch.inference_mode():
        want_17 = model_17(preprocess(images_17)).argmax(dim=1)     # the expected predictions
    model_17.train()                                                # back to the state of the given cell
    modes_17 = []                                                   # how each forward pass of model_17 runs
    hook_17 = model_17.register_forward_hook(
        lambda module, args, output: modes_17.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled())))
    try:
        first_17, second_17 = predict_17(model_17, images_17), predict_17(model_17, images_17)
    finally:
        hook_17.remove()
    model_17.train()
    print(f"symptôme : {int((first_17 != second_17).sum())} prédictions sur 200 changent entre deux appels identiques")
    ok_17 = point_17(1, torch.equal(first_17, want_17) and torch.equal(second_17, want_17),
                     "les prédictions sont celles du mode évaluation, identiques d'un appel à l'autre.",
                     "les prédictions changent d'un appel à l'autre : quelle couche de model_17 tire au hasard, et "
                     "quand le fait-elle ?")
    if ok_17:
        inference_17 = bool(modes_17) and all(inference for inference, _ in modes_17)
        no_grad_17 = bool(modes_17) and all(not grad for _, grad in modes_17)
        ok_17 = point_17(2, inference_17,
                         "la passe avant du modèle se fait sous torch.inference_mode().",
                         "la passe avant se fait sous torch.no_grad() : les gradients sont bien coupés, mais le contrat "
                         "demande torch.inference_mode(), plus strict et un peu plus rapide." if no_grad_17 else
                         "la passe avant enregistre le graphe de calcul (les gradients sont activés) : PyTorch garde "
                         "en mémoire de quoi rétropropager, pour rien.")
    else:
        print("⏳ Ex 23.17 · point 2 : corrige d'abord le point 1.")
    if ok_17:
        TRAP_17["ran"] = False
        try:
            load_weights_17(WORK_DIR / "downloaded_17.pt")
            refused_17 = False
        except Exception as error_17:                               # the safe loader refuses the unknown object
            refused_17 = True
            print("   chargement refusé :", type(error_17).__name__, "-", str(error_17).splitlines()[0][:120])
        good_17 = load_weights_17(WORK_DIR / "mlp_17.pt")
        ok_17 = point_17(3, refused_17 and not TRAP_17["ran"] and isinstance(good_17, dict)
                         and all(torch.equal(good_17[k], v) for k, v in model_17.state_dict().items()),
                         "le fichier piégé est refusé sans rien exécuter, et les vrais poids se chargent.",
                         f"le fichier piégé a {'exécuté son code' if TRAP_17['ran'] else 'été chargé'} : "
                         "quel argument de torch.load n'accepte que des tenseurs et des types simples ?")
    else:
        print("⏳ Ex 23.17 · point 3 : corrige d'abord les points précédents.")'''

TODO_18 = r'''def evaluate_18(model, loader, loss_fn, device="cpu"):
    """(mean loss, accuracy) over all the examples of `loader`, as Python floats: the model in evaluation mode, no
    autograd graph; the mean is weighted by the size of each mini-batch."""
    raise NotImplementedError("evaluate_18")


def fit_18(model, train_loader, val_loader, loss_fn, optimizer, epochs, callbacks=(), device="cpu"):
    """Train `model` in place for at most `epochs` epochs and return the history
    {"train_loss": [...], "train_acc": [...], "val_loss": [...], "val_acc": [...]} (one Python float per epoch).

    Each epoch: model.train(), one optimizer step per mini-batch; the training loss and accuracy of the epoch are the
    means over its mini-batches, weighted by their sizes; then evaluate_18 on val_loader; then each callback is called
    as callback(epoch, history, model, optimizer), with the epoch counted from 1 and the history already updated; if
    one of them returns True, training stops after this epoch."""
    raise NotImplementedError("fit_18")'''

SOLUTION_18 = r'''@torch.inference_mode()
def evaluate_18(model, loader, loss_fn, device="cpu"):
    """(mean loss, accuracy) over all the examples of `loader`, as Python floats: the model in evaluation mode, no
    autograd graph; the mean is weighted by the size of each mini-batch."""
    model.eval()
    total, correct, count = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += loss_fn(logits, yb).item() * len(xb)           # loss_fn gives the MEAN of the mini-batch
        correct += (logits.argmax(dim=1) == yb).sum().item()
        count += len(xb)
    return total / count, correct / count


def fit_18(model, train_loader, val_loader, loss_fn, optimizer, epochs, callbacks=(), device="cpu"):
    """Train `model` in place for at most `epochs` epochs and return the history
    {"train_loss": [...], "train_acc": [...], "val_loss": [...], "val_acc": [...]} (one Python float per epoch).

    Each epoch: model.train(), one optimizer step per mini-batch; the training loss and accuracy of the epoch are the
    means over its mini-batches, weighted by their sizes; then evaluate_18 on val_loader; then each callback is called
    as callback(epoch, history, model, optimizer), with the epoch counted from 1 and the history already updated; if
    one of them returns True, training stops after this epoch."""
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()                                            # evaluate_18 left the model in evaluation mode
        total, correct, count = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            count += len(xb)
        val_loss, val_acc = evaluate_18(model, val_loader, loss_fn, device)
        for key, value in (("train_loss", total / count), ("train_acc", correct / count),
                           ("val_loss", val_loss), ("val_acc", val_acc)):
            history[key].append(float(value))
        stop = [bool(callback(epoch, history, model, optimizer)) for callback in callbacks]   # call them ALL
        if any(stop):
            break
    return history'''

CHECK_18 = r'''history_18 = None
with checked("23.18"):
    torch.manual_seed(2318)
    probe_18 = make_mlp(dropout=0.5)
    small_18 = DataLoader(TensorDataset(X_VAL_T[:1000], Y_VAL_T[:1000]), batch_size=300)   # 300, 300, 300, 100
    probe_18.eval()
    with torch.inference_mode():
        logits_18 = probe_18(X_VAL_T[:1000])
    want_18 = (nn.CrossEntropyLoss()(logits_18, Y_VAL_T[:1000]).item(),
               (logits_18.argmax(dim=1) == Y_VAL_T[:1000]).float().mean().item())
    probe_18.train()
    got_18 = pair("23.18", "evaluate_18", evaluate_18(probe_18, small_18, nn.CrossEntropyLoss()), names=("loss", "accuracy"),
                  floats=("loss", "accuracy"))
    same("23.18", "evaluate_18 sur 1 000 images en mini-batches de 300, 300, 300 et 100 (modèle avec dropout)", got_18,
         want_18, rtol=1e-5)
    train_18, val_18 = make_loaders()
    torch.manual_seed(2318)
    model_18 = make_mlp()
    history_18 = fit_18(model_18, train_18, val_18, nn.CrossEntropyLoss(), torch.optim.Adam(model_18.parameters(), lr=1e-3), 3)
    if returned("23.18", "fit_18", history_18):
        keys_18 = ["train_loss", "train_acc", "val_loss", "val_acc"]
        ok_18 = isinstance(history_18, dict) and all(isinstance(history_18.get(k), list) and len(history_18[k]) == 3
                                                     and all(isinstance(v, float) for v in history_18[k]) for k in keys_18)
        verdict("23.18", ok_18, "l'historique a quatre listes de 3 floats.",
                f"attendu un dict {keys_18} de listes de 3 floats Python, reçu {history_18!r:.200}.")
        if ok_18:
            for epoch_18 in range(3):
                print(f"epoch {epoch_18 + 1} · train loss {history_18['train_loss'][epoch_18]:.4f} acc "
                      f"{history_18['train_acc'][epoch_18]:.4f} · val loss {history_18['val_loss'][epoch_18]:.4f} acc "
                      f"{history_18['val_acc'][epoch_18]:.4f}")
            verdict("23.18", history_18["train_loss"][2] < history_18["train_loss"][0] and history_18["val_acc"][2] > 0.8,
                    "la loss d'entraînement baisse et l'accuracy de validation dépasse 80 % en 3 epochs.",
                    "la loss d'entraînement devrait baisser et l'accuracy de validation dépasser 80 % : chaque mini-batch "
                    "fait-il bien zero_grad, backward et step ?")
            last_18 = evaluate_18(model_18, val_18, nn.CrossEntropyLoss())
            same("23.18", "la dernière val_loss et val_acc de l'historique contre evaluate_18 après l'entraînement",
                 [history_18["val_loss"][-1], history_18["val_acc"][-1]], list(last_18), rtol=1e-6)
    torch.manual_seed(2318)
    frozen_18 = make_mlp()
    ordered_18 = DataLoader(TensorDataset(X_VAL_T[:1000], Y_VAL_T[:1000]), batch_size=300)
    frozen_history_18 = fit_18(frozen_18, ordered_18, ordered_18, nn.CrossEntropyLoss(),
                               torch.optim.SGD(frozen_18.parameters(), lr=0.0), 1)
    same("23.18", "avec un learning rate nul, la loss et l'accuracy d'entraînement pondérées par la taille des mini-batches",
         [frozen_history_18["train_loss"][0], frozen_history_18["train_acc"][0]],
         [frozen_history_18["val_loss"][0], frozen_history_18["val_acc"][0]], rtol=1e-6)
    calls_18 = []

    def stop_at_two_18(epoch, history, model, optimizer):
        calls_18.append((epoch, len(history["val_loss"])))
        return epoch >= 2

    stop_model_18 = make_mlp()
    stopped_18 = fit_18(stop_model_18, ordered_18, ordered_18, nn.CrossEntropyLoss(),
                        torch.optim.SGD(stop_model_18.parameters(), lr=0.0), 5, callbacks=[stop_at_two_18])
    verdict("23.18", calls_18 == [(1, 1), (2, 2)] and len(stopped_18["val_loss"]) == 2,
            "les callbacks reçoivent l'epoch (comptée à partir de 1) et l'historique à jour, et un True arrête l'entraînement.",
            f"un callback qui rend True à l'epoch 2 : attendu les appels [(1, 1), (2, 2)] (epoch, longueur de l'historique) "
            f"et 2 epochs, reçu {calls_18} et {len(stopped_18['val_loss'])} epochs.")'''

TODO_19 = r'''best_loss_epoch_19 = ...   # a) the epoch (counted from 1) of the smallest validation loss
best_acc_epoch_19 = ...    # b) the epoch of the largest validation accuracy
loss_ratio_19 = ...        # c) the validation loss of epoch 50 divided by the smallest one (2 decimals)
acc_change_19 = ...        # d) val accuracy of epoch 50 minus val accuracy at the epoch of a), in points (1 decimal)
first_95_19 = ...          # e) the first epoch whose TRAINING accuracy reaches 95 %'''

SOLUTION_19 = r'''val_loss_19, val_acc_19 = np.array(HISTORY_19["val_loss"]), np.array(HISTORY_19["val_acc"])
best_loss_epoch_19 = int(np.argmin(val_loss_19)) + 1          # epochs are counted from 1, indices from 0
best_acc_epoch_19 = int(np.argmax(val_acc_19)) + 1
loss_ratio_19 = float(val_loss_19[-1] / val_loss_19.min())
acc_change_19 = float((val_acc_19[-1] - val_acc_19[best_loss_epoch_19 - 1]) * 100)
first_95_19 = int(np.argmax(np.array(HISTORY_19["train_acc"]) >= 0.95)) + 1'''

CHECK_19 = r'''wb.check("23.19a", best_loss_epoch_19)
wb.check("23.19b", best_acc_epoch_19)
wb.check("23.19c", loss_ratio_19)
wb.check("23.19d", acc_change_19)
wb.check("23.19e", first_95_19)'''

TODO_20 = r'''def predict_all_20(model, loader, device="cpu"):
    """(y_true, y_pred): two int64 NumPy arrays (N,) over all the examples of `loader`, in its order; the model in
    evaluation mode, no autograd graph."""
    raise NotImplementedError("predict_all_20")


def top_confusions_20(C, k=3):
    """The k largest OFF-diagonal cells of the confusion matrix C, as a list of (true_class, predicted_class, count)
    tuples of Python ints, by decreasing count (ties: by true_class, then predicted_class)."""
    raise NotImplementedError("top_confusions_20")


def show_confusions_20(images, y_true, y_pred, true_class, pred_class, n=8):
    """Show (in one row) the first n images whose label is true_class and that the model predicted as pred_class,
    with a title that names both classes; ends with plt.show()."""
    raise NotImplementedError("show_confusions_20")'''

SOLUTION_20 = r'''@torch.inference_mode()
def predict_all_20(model, loader, device="cpu"):
    """(y_true, y_pred): two int64 NumPy arrays (N,) over all the examples of `loader`, in its order; the model in
    evaluation mode, no autograd graph."""
    model.eval()
    truths, preds = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(dim=1).cpu())
        truths.append(yb)
    return torch.cat(truths).numpy().astype(np.int64), torch.cat(preds).numpy().astype(np.int64)


def top_confusions_20(C, k=3):
    """The k largest OFF-diagonal cells of the confusion matrix C, as a list of (true_class, predicted_class, count)
    tuples of Python ints, by decreasing count (ties: by true_class, then predicted_class)."""
    C = np.asarray(C)
    cells = [(int(C[i, j]), i, j) for i in range(C.shape[0]) for j in range(C.shape[1]) if i != j]
    cells.sort(key=lambda cell: (-cell[0], cell[1], cell[2]))
    return [(i, j, count) for count, i, j in cells[:k]]


def show_confusions_20(images, y_true, y_pred, true_class, pred_class, n=8):
    """Show (in one row) the first n images whose label is true_class and that the model predicted as pred_class,
    with a title that names both classes; ends with plt.show()."""
    chosen = np.flatnonzero((np.asarray(y_true) == true_class) & (np.asarray(y_pred) == pred_class))[:n]
    fig, axes = plt.subplots(1, n, figsize=(n * 1.1, 1.6))
    for ax, index in zip(np.atleast_1d(axes), list(chosen) + [None] * n):
        if index is not None:
            ax.imshow(images[index], cmap="gray_r")
        ax.axis("off")
    fig.suptitle(f"{CLASSES[true_class]} prédit {CLASSES[pred_class]} ({len(chosen)} montrées)", fontsize=9)
    plt.show()'''

CHECK_20 = r'''with checked("23.20"):
    scripted_20 = np.array([[50, 3, 9], [2, 40, 9], [9, 1, 30]])
    top_20 = top_confusions_20(scripted_20, k=3)
    verdict("23.20", top_20 == [(0, 2, 9), (1, 2, 9), (2, 0, 9)],
            "top_confusions_20 trie les cases hors diagonale, ex aequo par classe réelle puis prédite.",
            f"sur [[50, 3, 9], [2, 40, 9], [9, 1, 30]], attendu [(0, 2, 9), (1, 2, 9), (2, 0, 9)], reçu {top_20!r}.")
    train_20, val_20 = make_loaders()
    torch.manual_seed(2320)
    model_20 = make_mlp()
    fit_18(model_20, train_20, val_20, nn.CrossEntropyLoss(), torch.optim.Adam(model_20.parameters(), lr=1e-3), 5)
    y_true_20, y_pred_20 = pair("23.20", "predict_all_20", predict_all_20(model_20, val_20), names=("y_true", "y_pred"))
    model_20.eval()
    with torch.inference_mode():
        want_pred_20 = model_20(X_VAL_T).argmax(dim=1).numpy()
    same("23.20", "y_pred contre les prédictions du modèle sur les 6 000 images de validation", y_pred_20, want_pred_20)
    same("23.20", "y_true contre Y_VAL", y_true_20, Y_VAL)
    C_20 = mylearn.metrics.confusion_matrix(y_true_20, y_pred_20, labels=list(range(10)))
    print(f"accuracy de validation : {np.trace(C_20) / C_20.sum():.4f}")
    off_20 = C_20.astype(float).copy()
    np.fill_diagonal(off_20, np.nan)                               # the errors only, as in the book's heat map
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image_20 = ax.imshow(off_20, cmap="Reds")
    ax.set_xticks(range(10), CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(10), CLASSES, fontsize=8)
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("classe réelle")
    fig.colorbar(image_20, ax=ax, label="nombre d'images")
    ax.set_title("Erreurs de validation (diagonale retirée)")
    plt.show()
    worst_20 = top_confusions_20(C_20, k=3)
    for true_20, pred_20, count_20 in worst_20:
        print(f"{CLASSES[true_20]:>12s} prédit {CLASSES[pred_20]:<12s} {count_20} fois")
    show_confusions_20(X_VAL, y_true_20, y_pred_20, worst_20[0][0], worst_20[0][1])'''

# 23.19 and 23.27: the given history is built from tools/chapters/history_ch23.json (precomputed by history_ch23.py)
_HISTORY_19 = json.loads(Path(__file__).with_name("history_ch23.json").read_text(encoding="utf-8"))
GIVEN_19 = ("HISTORY_19 = {   # the book's MLP, 50 epochs of Adam on 54 000 images (tools/chapters/history_ch23.py)\n"
            + "".join(f'    "{key}": {_HISTORY_19[key]},\n' for key in ("train_loss", "train_acc", "val_loss", "val_acc"))
            + "}\n" + r'''
fig, (ax_loss_19, ax_acc_19) = plt.subplots(1, 2, figsize=(11, 3.3))
epochs_19 = range(1, 51)
ax_loss_19.plot(epochs_19, HISTORY_19["train_loss"], label="entraînement (50 epochs)")
ax_loss_19.plot(epochs_19, HISTORY_19["val_loss"], label="validation (50 epochs)")
ax_acc_19.plot(epochs_19, HISTORY_19["train_acc"], label="entraînement (50 epochs)")
ax_acc_19.plot(epochs_19, HISTORY_19["val_acc"], label="validation (50 epochs)")
mine_19 = globals().get("history_18")
if isinstance(mine_19, dict) and len(mine_19.get("val_loss", [])) > 0:     # your 3 epochs of 23.18, if you have them
    ax_loss_19.plot(range(1, len(mine_19["val_loss"]) + 1), mine_19["val_loss"], "o--", label="ta validation (23.18)")
    ax_acc_19.plot(range(1, len(mine_19["val_acc"]) + 1), mine_19["val_acc"], "o--", label="ta validation (23.18)")
ax_loss_19.set_title("loss")
ax_acc_19.set_title("accuracy")
for ax_19 in (ax_loss_19, ax_acc_19):
    ax_19.set_xlabel("epoch")
    ax_19.legend(fontsize=7)
plt.show()''')

_VL_19, _VA_19 = _HISTORY_19["val_loss"], _HISTORY_19["val_acc"]
_TL_19, _TA_19 = _HISTORY_19["train_loss"], _HISTORY_19["train_acc"]
RECORD_19 = (f'''wb.record("23.19a", best_loss_epoch_19, mistakes={{"les epochs se comptent à partir de 1 : l'indice 0 est l'epoch 1": best_loss_epoch_19 - 1,
          "la loss de VALIDATION, pas celle d'entraînement": {_TL_19.index(min(_TL_19)) + 1}}})
wb.record("23.19b", best_acc_epoch_19, mistakes={{"les epochs se comptent à partir de 1 : l'indice 0 est l'epoch 1": best_acc_epoch_19 - 1,
          "l'accuracy de VALIDATION, pas celle d'entraînement": {_TA_19.index(max(_TA_19)) + 1},
          "b) porte sur l'accuracy de validation : np.argmax de val_acc": best_loss_epoch_19}})
wb.record("23.19c", loss_ratio_19, decimals=2, mistakes={{"la loss de l'epoch 50 au numérateur, la plus petite au dénominateur": 1 / loss_ratio_19,
          "la loss de VALIDATION, pas celle d'entraînement": {_TL_19[-1] / min(_TL_19)!r}}})
wb.record("23.19d", acc_change_19, decimals=1, mistakes={{"en points : multiplie la différence d'accuracy par 100": acc_change_19 / 100,
          "l'accuracy de l'epoch 50 MOINS celle de l'epoch de a)": -acc_change_19}})
wb.record("23.19e", first_95_19, mistakes={{"les epochs se comptent à partir de 1 : l'indice 0 est l'epoch 1": first_95_19 - 1}})''')


PART_B = Part("B", "Inférence, boucle d'entraînement, historique et analyse des erreurs",
              "Tu répares trois bugs d'inférence (23.17) ; tu écris **ta** fonction `fit()`, qui garde l'historique "
              "et accepte des callbacks (23.18) ; tu lis un historique de 50 epochs (23.19) ; puis tu analyses les "
              "erreurs d'un modèle (23.20).\n\n"
              "La cellule suivante installe le pipeline des parties B à D, celui que tu as écrit en 23.15 et 23.16 : "
              "le découpage stratifié `X_TR`, `X_VAL`, `Y_TR`, `Y_VAL` (54 000 + 6 000 images), les statistiques "
              "`MEAN` et `STD`, `preprocess`, `make_loaders` et `make_mlp` (avec un dropout en option).",
              given=PIPELINE, exercises=[
    Ex("23.17", "🐛", 2, 25, "Trois bugs d'inférence : eval() oublié, gradients calculés pour rien (inference_mode), chargement non sûr (weights_only)",
       "réparer une fonction de prédiction et une fonction de chargement qui marchent en apparence.",
       "Ex 23.16 · ✏️ 23.4 · fiche §23.7 (encadré « model.eval() ne coupe pas les gradients »), §23.8.2, §23.9",
       thread="Fashion-MNIST", tracks="R, C",
       body=r"""Un collègue a écrit deux fonctions pour servir un modèle avec dropout, `model_17`. Elles tournent sans erreur, et pourtant elles contiennent **trois** bugs : `predict_17` n'est pas fiable et gaspille de la mémoire, et `load_weights_17` peut exécuter du code caché dans un fichier téléchargé. La cellule suivante crée le modèle (en mode entraînement, comme après sa construction), sauve ses poids dans `WORK_DIR / "mlp_17.pt"`, et fabrique un fichier `downloaded_17.pt` « trouvé en ligne », piégé : le charger sans précaution exécute une fonction (ici inoffensive : elle lève un drapeau et affiche un message).

Corrige les deux fonctions **sur place**, sans changer leurs signatures. Le contrat est dans leurs docstrings.

La vérification affiche d'abord le symptôme, puis trois diagnostics enchaînés (tant qu'un point échoue, les suivants attendent). Elle tourne dès le premier « Tout exécuter » : les messages du code fautif sont ton point de départ.

Dans tes notes : lequel des trois bugs fausse les résultats, lequel coûte seulement du temps et de la mémoire, et lequel est une faille de sécurité ? Avec un modèle sans dropout ni batchnorm, le premier bug se verrait-il ?""",
       given=GIVEN_17, todo=TODO_17, check=CHECK_17, solution=SOLUTION_17 + "\n\n\n" + CHECK_17,
       note="Sans `model.eval()`, le dropout éteint au hasard la moitié des neurones à chaque appel : les "
            "prédictions changent d'un appel à l'autre, et sont moins bonnes. Sans `torch.inference_mode()` (ou "
            "`torch.no_grad()`), PyTorch garde le graphe de calcul de chaque prédiction : de la mémoire et du temps "
            "pour rien, et une sortie qui `requires_grad`. `weights_only=False` laisse `pickle` reconstruire "
            "n'importe quel objet, en appelant n'importe quelle fonction : un fichier de poids téléchargé devient un "
            "programme. Le premier bug fausse les résultats, le deuxième coûte des ressources, le troisième est une "
            "faille. Sans dropout ni batchnorm, le premier ne se verrait pas… jusqu'au jour où l'on en ajoute."),

    Ex("23.18", "📦", 3, 45, "Ta fonction fit() : boucle, validation et historique",
       "écrire une fonction d'entraînement réutilisable, avec validation, historique et callbacks.",
       "Ex 23.16 · ch. 20 (boucle d'entraînement) · ✏️ 23.5 · 23.R2 · fiche §23.6.3, §23.6.4, §23.7, §23.8",
       thread="Fashion-MNIST", tracks="R, C",
       body=r"""Le `fit()` de Keras, en PyTorch, écrit par toi : les exercices suivants s'en servent.

**Contrat**
- `evaluate_18(model, loader, loss_fn, device="cpu")` rend `(loss moyenne, accuracy)` sur **tous** les exemples du loader, en `float` Python : le modèle en mode évaluation, sans graphe de calcul ; la moyenne est pondérée par la taille de chaque mini-batch (le dernier peut être plus petit) ;
- `fit_18(model, train_loader, val_loader, loss_fn, optimizer, epochs, callbacks=(), device="cpu")` entraîne `model` sur place pendant au plus `epochs` epochs et rend l'historique `{"train_loss": [...], "train_acc": [...], "val_loss": [...], "val_acc": [...]}`, un `float` Python par epoch ;
- à chaque epoch : `model.train()`, un pas d'optimiseur par mini-batch ; la loss et l'accuracy d'entraînement de l'epoch sont les moyennes sur ses mini-batches, pondérées par leur taille (calculées pendant l'epoch, comme Keras) ; puis `evaluate_18` sur `val_loader` ; puis chaque callback est appelé, `callback(epoch, history, model, optimizer)`, avec l'epoch comptée **à partir de 1** et l'historique déjà à jour ; si l'un d'eux rend `True`, l'entraînement s'arrête après cette epoch.

La vérification essaie `evaluate_18` sur des mini-batches de tailles inégales et un modèle avec dropout, entraîne le MLP du livre 3 epochs (Adam, learning rate 0,001) et affiche ton historique, contrôle la pondération avec un learning rate nul, puis un callback qui arrête l'entraînement à l'epoch 2.

Dans tes notes : pourquoi la loss d'entraînement affichée pour une epoch est-elle un peu plus haute que la loss du modèle de fin d'epoch sur ces mêmes images ? Pourquoi `fit_18` doit-il rappeler `model.train()` au début de **chaque** epoch ?""",
       todo=TODO_18, check=CHECK_18, solution=SOLUTION_18 + "\n\n\n" + CHECK_18,
       note="La loss d'entraînement d'une epoch est une moyenne prise pendant que les poids changent : ses premiers "
            "mini-batches ont été vus par un modèle moins bon que celui de la fin, d'où une valeur un peu plus haute "
            "que la loss du modèle final sur ces images. `evaluate_18` met le modèle en mode évaluation : sans "
            "`model.train()` au début de l'epoch suivante, un dropout ou une batchnorm resteraient en mode "
            "évaluation pendant tout le reste de l'entraînement, sans aucune erreur."),

    Ex("23.19", "📈", 2, 20, "Lire l'historique : 3 epochs, puis 50",
       "lire un historique d'entraînement : meilleure epoch, overfitting, loss contre accuracy.",
       "Ex 23.18 · ch. 9 (overfitting) · fiche §23.8.1, §23.8.3", thread="Fashion-MNIST", tracks="R, C",
       body=r"""Le livre lance 3 epochs, puis 100. Ta `fit_18` vient d'en faire 3 (`history_18`). `HISTORY_19`, calculé à l'avance par `tools/chapters/history_ch23.py`, en contient **50** : le MLP du livre (Adam, learning rate 0,001, mini-batches de 256) entraîné sur les 54 000 images d'entraînement et validé sur les 6 000 autres. La cellule suivante trace les deux historiques.

Réponds en calculant sur `HISTORY_19` (des listes Python, une valeur par epoch, l'epoch 1 à l'indice 0) :
a) `best_loss_epoch_19` : l'epoch de la plus petite loss de validation ;
b) `best_acc_epoch_19` : l'epoch de la plus grande accuracy de validation ;
c) `loss_ratio_19` : la loss de validation de l'epoch 50 divisée par la plus petite (2 décimales) ;
d) `acc_change_19` : l'accuracy de validation de l'epoch 50 moins celle de l'epoch de a), en points (1 décimale) ;
e) `first_95_19` : la première epoch dont l'accuracy d'**entraînement** atteint 95 %.

Dans tes notes : que t'aurait fait conclure la 3ᵉ epoch seule (regarde ta courbe et celle de `HISTORY_19`) ? Le modèle fait-il de l'overfitting, et à partir de quand ? Si tu devais garder **une** epoch, laquelle, et selon quel critère ?""",
       given=GIVEN_19, todo=TODO_19, check=CHECK_19, solution=SOLUTION_19 + "\n\n\n" + CHECK_19, record=RECORD_19,
       note="La loss de validation est au plus bas dès l'epoch 8, puis remonte jusqu'à presque doubler, pendant que l'accuracy de validation continue de monter un peu (meilleure à l'epoch 29) avant de plafonner vers 90 % : le modèle devient de plus en plus sûr de lui, y compris sur ses erreurs (💼 23.E4). C'est de l'overfitting au sens de la loss dès la 9ᵉ epoch ; l'accuracy, plus grossière, ne le montre presque pas. La 3ᵉ epoch seule aurait trompé : sa loss de validation remonte, un soubresaut, pas encore l'overfitting. On garderait l'epoch 8 si l'on se sert des probabilités, ou une epoch autour de 20 à 30 si seule la classe prédite compte, en choisissant la mesure avant de regarder les courbes."),

    Ex("23.20", "📦", 2, 25, "Analyser les erreurs : matrice de confusion et grille des confusions",
       "trouver où et pourquoi un classifieur se trompe, avec une matrice de confusion et les images mal classées.",
       "Ex 23.18 · ch. 3 (matrice de confusion, `mylearn.metrics`) · fiche §23.8.1", thread="Fashion-MNIST", tracks="C",
       body=r"""L'accuracy dit **combien** le modèle se trompe ; la matrice de confusion dit **où**, et les images mal classées disent souvent **pourquoi** (fiche §23.8.1). La vérification entraîne le MLP du livre 5 epochs avec **ta** `fit_18`, puis se sert de tes trois fonctions.

**Contrat**
- `predict_all_20(model, loader, device="cpu")` rend `(y_true, y_pred)`, deux tableaux NumPy `int64` `(N,)`, sur tous les exemples du loader, dans son ordre ; le modèle en mode évaluation, sans graphe de calcul ;
- `top_confusions_20(C, k=3)` rend les $k$ plus grandes cases **hors diagonale** de la matrice de confusion `C` (lignes : classe réelle ; colonnes : classe prédite), en liste de triplets `(classe_réelle, classe_prédite, effectif)` d'entiers Python, par effectif décroissant (ex aequo : par classe réelle, puis par classe prédite) ;
- `show_confusions_20(images, y_true, y_pred, true_class, pred_class, n=8)` affiche sur une ligne les $n$ premières images de la classe `true_class` prédites `pred_class`, avec un titre qui nomme les deux classes, et finit par `plt.show()`.

La matrice se calcule avec **ta** `mylearn.metrics.confusion_matrix` du ch. 3 (ou la référence, si tu as sauté ce chapitre). La vérification affiche la carte des erreurs (la diagonale retirée, comme celle du livre), les trois confusions les plus fréquentes et les images de la première.

Dans tes notes : quelles classes se confondent le plus, et dans quel sens ? En regardant les images, dirais-tu que le modèle se trompe « bêtement », ou que certaines images sont ambiguës, voire ont un label faux ?""",
       todo=TODO_20, check=CHECK_20, solution=SOLUTION_20 + "\n\n\n" + CHECK_20,
       note="Les confusions viennent presque toutes des vêtements du haut du corps : chemise (*Shirt*) prise pour "
            "un T-shirt, un pull ou un manteau, et l'inverse ; puis pull et manteau entre eux. En 28 × 28 pixels "
            "gris, une chemise sans boutons visibles ressemble à un T-shirt à manches longues : beaucoup de ces "
            "images sont ambiguës même pour un humain, et le label de quelques-unes semble faux. Ce diagnostic dit "
            "où chercher : plus de résolution, un CNN (ch. 21 et 24), ou des classes regroupées si l'usage le "
            "permet."),
])

# ---------------------------------------------------------------------------
# Part C: real-world images, prediction, checkpoints, a pretrained model (23.21 to 23.24)
# ---------------------------------------------------------------------------
REFERENCE_C = r'''# The reference model of part C: the book's MLP, 5 epochs of Adam on the 54 000 training images (about 10 s on a CPU)
def train_reference(epochs=5, seed=2321):
    torch.manual_seed(seed)
    model = make_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader, _ = make_loaders(n_train=len(X_TR), seed=seed)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


started_c = time.perf_counter()
MODEL_C = train_reference()
with torch.inference_mode():
    VAL_ACC_C = (MODEL_C(X_VAL_T).argmax(dim=1) == Y_VAL_T).float().mean().item()
print(f"MODEL_C trained in {time.perf_counter() - started_c:.0f} s · validation accuracy {VAL_ACC_C:.4f}")'''

GIVEN_21 = r'''def shift_right_21(images, pixels=2):
    """The images moved `pixels` columns to the right; the empty columns on the left are black (0)."""
    out = np.zeros_like(images)
    out[:, :, pixels:] = images[:, :, :-pixels]
    return out


TRANSFORMS_21 = {"originales": lambda a: a,
                 "décalées de 2 pixels à droite": shift_right_21,
                 "inversées (255 - x : fond clair)": lambda a: 255 - a,
                 "miroir gauche-droite": lambda a: a[:, :, ::-1].copy()}
fig, axes = plt.subplots(1, 4, figsize=(7, 2))
for ax, (name_21, transform_21) in zip(axes, TRANSFORMS_21.items()):
    ax.imshow(transform_21(X_VAL[:1])[0], cmap="gray_r", vmin=0, vmax=255)
    ax.set_title(name_21, fontsize=7)
    ax.axis("off")
plt.show()'''

TODO_21 = r'''shift_acc_21 = ...       # a) validation images shifted 2 pixels to the right: MODEL_C's accuracy is
                         #    "A" above 80 %, "B" from 60 to 80 %, "C" from 40 to 60 %, "D" below 40 %
invert_chance_21 = ...   # b) inverted images (255 - x): is the accuracy below 10 % (the chance level)? True or False
invert_sure_21 = ...     # c) inverted images: is the mean confidence (the largest probability) above 0.8? True or False
mirror_acc_21 = ...      # d) left-right mirror: is the accuracy above 80 %? True or False'''

SOLUTION_21 = r'''shift_acc_21, invert_chance_21, invert_sure_21, mirror_acc_21 = "C", True, True, False'''

EXPERIMENT_21 = r'''for name_21, transform_21 in TRANSFORMS_21.items():
    with torch.inference_mode():
        probs_21 = MODEL_C(preprocess(transform_21(X_VAL))).softmax(dim=1)
    accuracy_21 = (probs_21.argmax(dim=1) == Y_VAL_T).float().mean().item()
    confidence_21 = probs_21.max(dim=1).values.mean().item()
    print(f"{name_21:34s} accuracy {accuracy_21:6.1%} · confiance moyenne {confidence_21:.2f}")
wb.check("23.21a", shift_acc_21)
wb.check("23.21b", invert_chance_21)
wb.check("23.21c", invert_sure_21)
wb.check("23.21d", mirror_acc_21)'''

RECORD_21 = (r'''wb.record("23.21a", "C", mistakes=''' + "{" +
             '"2 pixels sur 28, c\'est peu pour un humain : mais une couche dense relie chaque poids à UNE position de pixel": "A", '
             '"le décalage coûte davantage : chaque pixel tombe sur des poids appris pour une autre position": "B", '
             '"le modèle garde une bonne part de ses réponses : la plupart des images restent proches de leur position": "D"' + "}" + r''')
wb.record("23.21b", True, mistakes={"le fond devient clair et le vêtement sombre : le modèle a-t-il jamais vu une telle image ?": False})
wb.record("23.21c", True, mistakes={"une softmax donne toujours des probabilités : rien n'oblige un modèle perdu à hésiter": False})
wb.record("23.21d", False, mistakes={"beaucoup de vêtements sont presque symétriques, mais leurs détails (fermetures, poches, talons) ne le sont pas": True})''')

TODO_22 = r'''def predict_proba_22(model, images, mean=MEAN, std=STD, device="cpu"):
    """The class probabilities (N, 10), a NumPy float array whose rows sum to 1, of uint8 images (N, 28, 28) or of ONE
    image (28, 28) (then the shape is (1, 10)): the training preprocessing, the model in evaluation mode, no graph."""
    raise NotImplementedError("predict_proba_22")


def top_k_22(probs, k=2):
    """For each row of probs (N, 10): the list of the k most probable (class name, probability) pairs, by decreasing
    probability (the probabilities as Python floats)."""
    raise NotImplementedError("top_k_22")


def save_preprocessing_22(path, mean, std, classes):
    """Write the preprocessing that travels with the model to a JSON file: {"mean": float, "std": float,
    "classes": [10 names], "input_shape": [1, 28, 28]}."""
    raise NotImplementedError("save_preprocessing_22")


def load_preprocessing_22(path):
    """The dict written by save_preprocessing_22."""
    raise NotImplementedError("load_preprocessing_22")'''

SOLUTION_22 = r'''def predict_proba_22(model, images, mean=MEAN, std=STD, device="cpu"):
    """The class probabilities (N, 10), a NumPy float array whose rows sum to 1, of uint8 images (N, 28, 28) or of ONE
    image (28, 28) (then the shape is (1, 10)): the training preprocessing, the model in evaluation mode, no graph."""
    images = np.asarray(images)
    if images.ndim == 2:                                  # one image: a batch of one
        images = images[None]
    model.eval()
    with torch.inference_mode():
        logits = model(preprocess(images, mean, std).to(device))
    return logits.softmax(dim=1).cpu().numpy()


def top_k_22(probs, k=2):
    """For each row of probs (N, 10): the list of the k most probable (class name, probability) pairs, by decreasing
    probability (the probabilities as Python floats)."""
    probs = np.asarray(probs)
    order = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return [[(CLASSES[j], float(row[j])) for j in idx] for row, idx in zip(probs, order)]


def save_preprocessing_22(path, mean, std, classes):
    """Write the preprocessing that travels with the model to a JSON file: {"mean": float, "std": float,
    "classes": [10 names], "input_shape": [1, 28, 28]}."""
    config = {"mean": float(mean), "std": float(std), "classes": list(classes), "input_shape": [1, 28, 28]}
    Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_preprocessing_22(path):
    """The dict written by save_preprocessing_22."""
    return json.loads(Path(path).read_text(encoding="utf-8"))'''

CHECK_22 = r'''with checked("23.22"):
    MODEL_C.train()                                       # the function must put the model in evaluation mode itself
    probs_22 = predict_proba_22(MODEL_C, X_VAL[:500])
    MODEL_C.eval()
    with torch.inference_mode():
        want_22 = MODEL_C(preprocess(X_VAL[:500])).softmax(dim=1).numpy()
    same("23.22", "predict_proba_22 sur 500 images de validation (MODEL_C laissé en mode entraînement avant l'appel)",
         probs_22, want_22, rtol=1e-5, atol=1e-6)
    verdict("23.22", isinstance(probs_22, np.ndarray), "predict_proba_22 rend un tableau NumPy.",
            f"attendu un tableau NumPy, reçu {type(probs_22).__name__} (.numpy() après .cpu()).")
    one_22 = predict_proba_22(MODEL_C, X_VAL[0])
    same("23.22", "une seule image (28, 28) : des probabilités de forme (1, 10)", one_22, want_22[:1], rtol=1e-5, atol=1e-6)
    top_22 = top_k_22(np.array([[0.1, 0.6, 0, 0, 0, 0, 0.3, 0, 0, 0]]), k=2)
    verdict("23.22", top_22 == [[("Trouser", 0.6), ("Shirt", 0.3)]],
            "top_k_22 rend les noms et les probabilités, de la plus probable à la moins probable.",
            f"sur [0.1, 0.6, 0, 0, 0, 0, 0.3, 0, 0, 0], attendu [[('Trouser', 0.6), ('Shirt', 0.3)]], reçu {top_22!r}.")
    path_22 = WORK_DIR / "preprocessing.json"
    save_preprocessing_22(path_22, np.float32(MEAN), np.float32(STD), CLASSES)    # NumPy float32 numbers on purpose
    config_22 = load_preprocessing_22(path_22)
    verdict("23.22", config_22 == {"mean": float(np.float32(MEAN)), "std": float(np.float32(STD)), "classes": CLASSES,
                                   "input_shape": [1, 28, 28]},
            f"le prétraitement fait l'aller-retour par {path_22.name} : {json.dumps(config_22)[:90]}…",
            f"après l'aller-retour, attendu {{'mean': {MEAN:.6f}, 'std': {STD:.6f}, 'classes': [...], 'input_shape': [1, 28, 28]}}, "
            f"reçu {config_22!r:.200}.")
    new_22 = X_VAL[-8:]                                    # eight images the model has never seen
    best_22 = top_k_22(predict_proba_22(MODEL_C, new_22, config_22["mean"], config_22["std"]), k=2)
    fig, axes = plt.subplots(1, 8, figsize=(12, 2.2))
    for ax, image_22, label_22, guesses_22 in zip(axes, new_22, Y_VAL[-8:], best_22):
        ax.imshow(image_22, cmap="gray_r")
        ax.set_title("\n".join(f"{name} {p:.0%}" for name, p in guesses_22), fontsize=7,
                     color="black" if guesses_22[0][0] == CLASSES[label_22] else "red")
        ax.set_xlabel(CLASSES[label_22], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Les deux classes les plus probables (en rouge : erreur) ; sous l'image, la vraie classe", fontsize=8)
    plt.show()'''

TODO_23 = r'''def save_checkpoint_23(path, model, optimizer, epoch, generator, config):
    """Save everything a resumed training needs, with tensors and simple types only (torch.load with weights_only=True
    must read it): the model and optimizer state_dicts, the epoch, the state of torch's global random generator, the
    state of `generator` (the DataLoader's), and `config`, the dict of the arguments of make_mlp."""
    raise NotImplementedError("save_checkpoint_23")


def resume_23(path, lr=1e-3):
    """(model, optimizer, generator, epoch): a NEW model make_mlp(**config), a new Adam(lr) on its parameters and a new
    torch.Generator, the three loaded from the checkpoint (read with weights_only=True); the global random generator
    of torch is restored too, LAST (building the model draws random numbers)."""
    raise NotImplementedError("resume_23")'''

SOLUTION_23 = r'''def save_checkpoint_23(path, model, optimizer, epoch, generator, config):
    """Save everything a resumed training needs, with tensors and simple types only (torch.load with weights_only=True
    must read it): the model and optimizer state_dicts, the epoch, the state of torch's global random generator, the
    state of `generator` (the DataLoader's), and `config`, the dict of the arguments of make_mlp."""
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": int(epoch),
                "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(), "config": dict(config)}, path)


def resume_23(path, lr=1e-3):
    """(model, optimizer, generator, epoch): a NEW model make_mlp(**config), a new Adam(lr) on its parameters and a new
    torch.Generator, the three loaded from the checkpoint (read with weights_only=True); the global random generator
    of torch is restored too."""
    checkpoint = torch.load(path, weights_only=True)
    model = make_mlp(**checkpoint["config"])               # the architecture is code: rebuilt from its arguments
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint["optimizer"])     # Adam's moments and step counts
    generator = torch.Generator()
    generator.set_state(checkpoint["loader_rng"])          # the order of the next mini-batches
    torch.set_rng_state(checkpoint["torch_rng"])           # the next dropout masks
    return model, optimizer, generator, checkpoint["epoch"]'''

CHECK_23 = r'''def run_epochs_23(model, optimizer, generator, epochs):
    """A few epochs on 2 048 training images, mini-batches of 128, in the order drawn by `generator`."""
    loader = DataLoader(TensorDataset(preprocess(X_TR[:2048]), torch.from_numpy(Y_TR[:2048])), batch_size=128,
                        shuffle=True, generator=generator)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


with checked("23.23"):
    config_23 = {"n_hidden": 256, "dropout": 0.2}
    torch.manual_seed(2323)                                  # run 1: four epochs in one go
    straight_23 = make_mlp(**config_23)
    run_epochs_23(straight_23, torch.optim.Adam(straight_23.parameters(), lr=1e-3), torch.Generator().manual_seed(23), 4)
    torch.manual_seed(2323)                                  # run 2: two epochs, a checkpoint, ...
    model_23 = make_mlp(**config_23)
    optimizer_23, generator_23 = torch.optim.Adam(model_23.parameters(), lr=1e-3), torch.Generator().manual_seed(23)
    run_epochs_23(model_23, optimizer_23, generator_23, 2)
    path_23 = WORK_DIR / "checkpoint_23.pt"
    save_checkpoint_23(path_23, model_23, optimizer_23, 2, generator_23, config_23)
    del model_23, optimizer_23, generator_23
    torch.manual_seed(999)                                   # ... "the next day": everything else has changed
    stored_23 = torch.load(path_23, weights_only=True)
    print("checkpoint :", sorted(stored_23), f"· {path_23.stat().st_size / 1e6:.1f} Mo")
    resumed_23 = resume_23(path_23)
    model_r_23, optimizer_r_23, generator_r_23, epoch_23 = pair("23.23", "resume_23", resumed_23,
                                                                names=("model", "optimizer", "generator", "epoch"))
    verdict("23.23", epoch_23 == 2, "resume_23 rend l'epoch sauvegardée (2).", f"attendu l'epoch 2, reçu {epoch_23!r}.")
    run_epochs_23(model_r_23, optimizer_r_23, generator_r_23, 2)         # ... and two more epochs
    gaps_23 = [(name, (a - b).abs().max().item()) for (name, a), b in
               zip(straight_23.state_dict().items(), model_r_23.state_dict().values())]
    worst_23 = max(gaps_23, key=lambda item: item[1])
    verdict("23.23", worst_23[1] == 0.0,
            "reprise exacte : après 2 + 2 epochs, les poids sont identiques, au bit près, à ceux de 4 epochs d'affilée.",
            f"après la reprise, les poids diffèrent de ceux d'un entraînement d'affilée (écart maximal {worst_23[1]:.2e} "
            f"sur {worst_23[0]}) : que manque-t-il au checkpoint, que n'as-tu pas rechargé, ou qu'est-ce qui tire des nombres "
            "au hasard APRÈS la restauration du générateur global ?")'''

GIVEN_24 = r'''N_FEAT_24 = 2_000                         # training images of the new head (the validation: 1 000 images)
SIZE_24 = 112 if FAST_MODE else 224       # the size given to the ResNet (224 is the size of ImageNet; 112 is 4 times cheaper)
try:
    from torchvision.models import ResNet18_Weights, resnet18
    WEIGHTS_24 = ResNet18_Weights.DEFAULT
    resnet18(weights=WEIGHTS_24)          # downloads about 45 MB the first time (then: the local cache)
    print("ResNet-18 weights ready:", WEIGHTS_24, "· its transforms:", WEIGHTS_24.transforms())
except Exception as error_24:             # no network: the exercise is done on Colab
    WEIGHTS_24 = None
    print("⏳ Ex 23.24 : poids de ResNet-18 indisponibles (réseau ?) :", type(error_24).__name__)
rng_24 = np.random.default_rng(2324)
IDX_24 = rng_24.choice(len(X_TR), N_FEAT_24, replace=False)       # a random subset of the training images'''

TODO_24 = r'''def make_transfer_24(weights, n_classes=10):
    """torchvision's resnet18(weights=weights) with ALL its parameters frozen, and a new last layer model.fc =
    nn.Linear(512, n_classes), the only trainable part."""
    raise NotImplementedError("make_transfer_24")


def to_resnet_input_24(images, weights, size):
    """uint8 images (N, 28, 28) -> a float32 tensor (N, 3, size, size): the pixels divided by 255, the gray channel
    copied into the 3 channels, resized to size x size (antialias=True), then normalized with the ImageNet mean and std
    of weights.transforms()."""
    raise NotImplementedError("to_resnet_input_24")


def extract_features_24(model, x, batch_size=250):
    """(N, 512) float32 tensor: the output of the frozen network BEFORE model.fc, in evaluation mode, no graph, by
    mini-batches of batch_size; model.fc is unchanged afterwards."""
    raise NotImplementedError("extract_features_24")'''

SOLUTION_24 = r'''from torchvision.transforms import v2


def make_transfer_24(weights, n_classes=10):
    """torchvision's resnet18(weights=weights) with ALL its parameters frozen, and a new last layer model.fc =
    nn.Linear(512, n_classes), the only trainable part."""
    model = resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad_(False)                                  # freeze the backbone
    model.fc = nn.Linear(model.fc.in_features, n_classes)        # a new head: its parameters are trainable
    return model


def to_resnet_input_24(images, weights, size):
    """uint8 images (N, 28, 28) -> a float32 tensor (N, 3, size, size): the pixels divided by 255, the gray channel
    copied into the 3 channels, resized to size x size (antialias=True), then normalized with the ImageNet mean and std
    of weights.transforms()."""
    reference = weights.transforms()
    x = torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    resize = v2.Resize((size, size), antialias=True)
    return v2.Normalize(reference.mean, reference.std)(resize(x)).contiguous()


def extract_features_24(model, x, batch_size=250):
    """(N, 512) float32 tensor: the output of the frozen network BEFORE model.fc, in evaluation mode, no graph, by
    mini-batches of batch_size; model.fc is unchanged afterwards."""
    head, model.fc = model.fc, nn.Identity()                     # everything but the head
    model.eval()
    try:
        with torch.inference_mode():
            return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    finally:
        model.fc = head'''

CHECK_24 = r'''if WEIGHTS_24 is None:
    print("⏳ Ex 23.24 : sans les poids de ResNet-18, cet exercice se fait sur Colab.")
else:
    with checked("23.24"):
        transfer_24 = make_transfer_24(WEIGHTS_24)
        trainable_24 = sum(p.numel() for p in transfer_24.parameters() if p.requires_grad)
        frozen_24 = sum(p.numel() for p in transfer_24.parameters() if not p.requires_grad)
        verdict("23.24", trainable_24 == 5_130 and frozen_24 == 11_176_512 and isinstance(transfer_24.fc, nn.Linear),
                f"{trainable_24:_} paramètres entraînables (la nouvelle tête) et {frozen_24:_} gelés.".replace("_", " "),
                f"attendu 5 130 paramètres entraînables (la tête) et 11 176 512 gelés, reçu {trainable_24} et {frozen_24}.")
        x_24 = to_resnet_input_24(X_TR[IDX_24[:4]], WEIGHTS_24, 64)
        ref_24 = WEIGHTS_24.transforms()
        base_24 = torch.from_numpy(X_TR[IDX_24[:4]].astype(np.float32) / 255.0).unsqueeze(1).expand(-1, 3, -1, -1)
        want_x_24 = (F.interpolate(base_24, size=(64, 64), mode="bilinear", antialias=True)
                     - torch.tensor(ref_24.mean).view(1, 3, 1, 1)) / torch.tensor(ref_24.std).view(1, 3, 1, 1)
        same("23.24", "to_resnet_input_24 sur 4 images, en 64 x 64", x_24, want_x_24, rtol=1e-4, atol=1e-4)
        started_24 = time.perf_counter()
        feats_tr_24 = extract_features_24(transfer_24, to_resnet_input_24(X_TR[IDX_24], WEIGHTS_24, SIZE_24))
        feats_val_24 = extract_features_24(transfer_24, to_resnet_input_24(X_VAL[:1000], WEIGHTS_24, SIZE_24))
        print(f"features : {tuple(feats_tr_24.shape)} et {tuple(feats_val_24.shape)}, en {time.perf_counter() - started_24:.0f} s")
        verdict("23.24", isinstance(transfer_24.fc, nn.Linear) and tuple(feats_tr_24.shape) == (N_FEAT_24, 512),
                "extract_features_24 rend (N, 512) et laisse la tête en place.",
                f"attendu des features (N, 512) et model.fc inchangée, reçu {tuple(feats_tr_24.shape)} et {type(transfer_24.fc).__name__}.")
        body_24 = nn.Sequential(*list(transfer_24.children())[:-1], nn.Flatten()).eval()
        with torch.inference_mode():
            want_f_24 = body_24(to_resnet_input_24(X_VAL[:20], WEIGHTS_24, SIZE_24))
        same("23.24", "les features de 20 images contre le réseau sans sa dernière couche", feats_val_24[:20], want_f_24,
             rtol=1e-4, atol=1e-4)
        torch.manual_seed(2324)                              # train ONLY the head, on the stored features: 20 epochs
        head_opt_24 = torch.optim.Adam(transfer_24.fc.parameters(), lr=1e-3)
        y_tr_24, y_val_24 = torch.from_numpy(Y_TR[IDX_24]), torch.from_numpy(Y_VAL[:1000])
        for _ in range(20):
            for i_24 in torch.randperm(N_FEAT_24).split(64):
                loss_24 = nn.CrossEntropyLoss()(transfer_24.fc(feats_tr_24[i_24]), y_tr_24[i_24])
                head_opt_24.zero_grad()
                loss_24.backward()
                head_opt_24.step()
        with torch.no_grad():
            acc_head_24 = (transfer_24.fc(feats_val_24).argmax(dim=1) == y_val_24).float().mean().item()
        torch.manual_seed(2324)                              # the book's MLP on the SAME 2 000 images, for comparison
        mlp_24 = make_mlp()
        mlp_opt_24 = torch.optim.Adam(mlp_24.parameters(), lr=1e-3)
        x_mlp_24 = preprocess(X_TR[IDX_24])
        for _ in range(15):
            for i_24 in torch.randperm(N_FEAT_24).split(64):
                loss_24 = nn.CrossEntropyLoss()(mlp_24(x_mlp_24[i_24]), y_tr_24[i_24])
                mlp_opt_24.zero_grad()
                loss_24.backward()
                mlp_opt_24.step()
        with torch.inference_mode():
            acc_mlp_24 = (mlp_24.eval()(X_VAL_T[:1000]).argmax(dim=1) == y_val_24).float().mean().item()
        print(f"sur 1 000 images de validation : tête de ResNet-18 gelé {acc_head_24:.3f} "
              f"(5 130 paramètres appris) · MLP du livre {acc_mlp_24:.3f} (623 290 paramètres appris)")
        verdict("23.24", acc_head_24 > 0.75, "la nouvelle tête, seule entraînée, dépasse 75 % d'accuracy.",
                f"la nouvelle tête ne fait que {acc_head_24:.3f} : la normalisation d'ImageNet est-elle appliquée ?")'''

PART_C = Part("C", "Prédire, sauvegarder, reprendre et réutiliser",
              "Tu prédis la réaction du modèle à des images qui ne ressemblent pas à celles du train (23.21) ; tu "
              "écris les fonctions de prédiction et le prétraitement qui voyage avec le modèle (23.22) ; tu reprends "
              "un entraînement interrompu au bit près (23.23) ; puis tu réutilises un ResNet-18 pré-entraîné "
              "(23.24).\n\nLa cellule suivante entraîne le modèle de référence de cette partie, `MODEL_C` : le MLP "
              "du livre, 5 epochs d'Adam sur les 54 000 images d'entraînement, dans les deux modes (une dizaine de "
              "secondes). Les prédictions de 23.21 ne dépendent donc pas de ton code.",
              given=REFERENCE_C, exercises=[
    Ex("23.21", "🔮", 1, 10, "Prédire la réaction du MLP à des images « du monde réel »",
       "prédire la robustesse d'un MLP à de petits changements de ses images, avant de la mesurer.",
       "Ex 23.18 · fiche §23.8.2", thread="Fashion-MNIST", tracks="R, C", hypothesis=True,
       body=r"""Le livre photographie des chiffres dans la rue et les prépare comme MNIST (fiche §23.8.2). Ici, on fabrique des images « du monde réel » reproductibles à partir des 6 000 images de validation : (1) les originales ; (2) décalées de **2 pixels** vers la droite (un vêtement mal centré) ; (3) inversées, $255 - x$ (un vêtement sombre sur fond clair) ; (4) en miroir gauche-droite. La cellule suivante en montre un exemple. On mesure l'accuracy de `MODEL_C` (88 à 89 % sur les originales) et sa confiance moyenne, la plus grande des 10 probabilités.

Prédis, **avant** d'exécuter :
a) `shift_acc_21` : sur les images décalées de 2 pixels, l'accuracy est `"A"` au-dessus de 80 %, `"B"` de 60 à 80 %, `"C"` de 40 à 60 %, `"D"` sous 40 % ;
b) `invert_chance_21` : sur les images inversées, l'accuracy est-elle sous 10 %, le niveau du hasard ? (`True` ou `False`) ;
c) `invert_sure_21` : sur les images inversées, la confiance moyenne dépasse-t-elle 0,8 ? (`True` ou `False`) ;
d) `mirror_acc_21` : en miroir, l'accuracy dépasse-t-elle 80 % ? (`True` ou `False`).""",
       given=GIVEN_21, todo=TODO_21, solution=SOLUTION_21,
       after=[("code", guarded(EXPERIMENT_21, ["shift_acc_21", "invert_chance_21", "invert_sure_21", "mirror_acc_21"],
                               "⏳ Ex 23.21 : écris d'abord tes quatre prédictions, puis relance cette cellule.")),
              ("md", "Dans tes notes : pour chaque prédiction ratée, qu'est-ce qui t'a trompé ? Que t'apprend c) sur "
                     "la confiance d'un réseau ? Que faudrait-il changer, dans les données ou dans le modèle, pour "
                     "qu'un décalage de 2 pixels ne coûte presque rien ?")],
       record=RECORD_21,
       note="Un décalage de 2 pixels fait tomber l'accuracy à un peu plus de 50 % : une couche dense relie chaque "
            "poids à **une** position de pixel, et rien ne lui dit qu'une image décalée est la même image. Les "
            "images inversées tombent **sous** le hasard (3 à 5 %) avec une confiance moyenne de 0,95 : la softmax "
            "rend toujours des probabilités, et un réseau qui n'a jamais vu de telles images peut être sûr de lui "
            "et faux ; une confiance élevée ne garantit rien hors des données d'entraînement. Le miroir coûte plus "
            "de 20 points : les vêtements ne sont symétriques qu'en apparence (fermetures, poches, talons et "
            "pointes des chaussures, toutes tournées du même côté dans Fashion-MNIST). Remèdes : des CNN, qui "
            "partagent leurs poids sur toutes les positions (ch. 21), et l'augmentation de données, qui montre ces "
            "variations à l'entraînement (ch. 24)."),

    Ex("23.22", "📦", 2, 25, "Prédire sur de nouvelles images : eval, inference_mode, probabilités",
       "prédire des probabilités sur de nouvelles images, et sauvegarder le prétraitement qui va avec le modèle.",
       "Ex 23.17 · Ex 23.21 · fiche §23.8.2, §23.9.5", thread="Fashion-MNIST", tracks="R, C",
       body=r"""Les fonctions qu'appellera une application qui utilise `MODEL_C`.

**Contrat**
- `predict_proba_22(model, images, mean=MEAN, std=STD, device="cpu")` rend les probabilités des classes, un tableau **NumPy** `(N, 10)` dont chaque ligne somme à 1, pour des images `uint8` `(N, 28, 28)` ou **une seule** image `(28, 28)` (alors `(1, 10)`) : le prétraitement de l'entraînement, le modèle en mode évaluation (la fonction l'y met elle-même), sans graphe de calcul ;
- `top_k_22(probs, k=2)` rend, pour chaque ligne, la liste des $k$ couples `(nom de la classe, probabilité)` les plus probables, par probabilité décroissante (des `float` Python) ;
- `save_preprocessing_22(path, mean, std, classes)` écrit un fichier JSON `{"mean": ..., "std": ..., "classes": [...], "input_shape": [1, 28, 28]}` (des `float` Python, même si on lui donne des nombres NumPy), et `load_preprocessing_22(path)` le relit en dictionnaire.

La vérification compare tes probabilités à celles du modèle (après avoir remis `MODEL_C` en mode entraînement), essaie une image seule, `top_k_22` sur un exemple, l'aller-retour du prétraitement par un fichier JSON, puis affiche les deux classes les plus probables de 8 images jamais vues.

Dans tes notes : pourquoi le fichier JSON doit-il accompagner le fichier de poids ? Que se passerait-il si l'application donnait au modèle des pixels de 0 à 255 ?""",
       todo=TODO_22, check=CHECK_22, solution=SOLUTION_22 + "\n\n\n" + CHECK_22,
       note="Sans le prétraitement, le fichier de poids ne sert à rien : l'application doit diviser par 255, "
            "retirer la même moyenne et diviser par le même écart-type que pendant l'entraînement, et connaître "
            "l'ordre des classes pour nommer la sortie. Avec des pixels de 0 à 255, le modèle répondrait quand même, "
            "sans erreur, avec des prédictions fausses (23.14 et 23.21 en montrent l'ampleur). `json.dumps` refuse "
            "un `np.float32` : on convertit avec `float(...)`."),

    Ex("23.23", "📦", 2, 30, "Reprendre un entraînement interrompu : checkpoint complet (modèle, optimiseur, epoch, générateurs aléatoires)",
       "sauvegarder un entraînement et le reprendre plus tard exactement là où il s'était arrêté.",
       "Ex 23.18 · ✏️ 23.8 · fiche §23.9 à §23.9.3, « Au-delà du livre : la reproductibilité »", thread="Fashion-MNIST",
       tracks="R, C",
       body=r"""Un entraînement de plusieurs heures s'interrompt (une coupure, une session Colab qui expire). Avec un bon checkpoint, on le reprend comme s'il ne s'était jamais arrêté : mêmes poids, même état d'Adam, même ordre des mini-batches, mêmes masques de dropout.

**Contrat**
- `save_checkpoint_23(path, model, optimizer, epoch, generator, config)` sauve, avec `torch.save`, **tout** ce qu'il faut pour reprendre, en tenseurs et types simples seulement (`torch.load(path, weights_only=True)` doit pouvoir le lire) : les `state_dict` du modèle et de l'optimiseur, l'epoch, l'état du générateur aléatoire global de PyTorch, l'état de `generator` (celui du `DataLoader`), et `config`, le dictionnaire des arguments de `make_mlp` (l'architecture, fiche §23.9.3) ;
- `resume_23(path, lr=1e-3)` rend `(model, optimizer, generator, epoch)` : un **nouveau** modèle `make_mlp(**config)`, un nouvel `Adam(lr)` sur ses paramètres et un nouveau `torch.Generator`, tous trois chargés depuis le checkpoint (lu avec `weights_only=True`), et l'epoch sauvegardée ; le générateur global de PyTorch est restauré lui aussi, **en dernier** : construire le modèle tire des nombres au hasard.

La vérification entraîne un MLP avec dropout 4 epochs d'affilée ; puis, depuis la même graine, 2 epochs, un checkpoint, un « lendemain » où tout a changé (`torch.manual_seed(999)`), ta reprise et 2 epochs de plus. Les poids doivent être **identiques au bit près**.

Dans tes notes : enlève (dans une copie de ta fonction) l'état de l'optimiseur, puis celui de `generator`, puis celui du générateur global : chaque oubli change-t-il les poids ? Lequel aurait le plus d'effet sur un vrai entraînement ?""",
       todo=TODO_23, check=CHECK_23, solution=SOLUTION_23 + "\n\n\n" + CHECK_23,
       note="Chaque oubli casse l'égalité au bit près, mais pas de la même façon. Sans l'état d'Adam, ses moments "
            "repartent de zéro et ses premiers pas, corrigés du biais, sont de la taille du learning rate pour "
            "chaque poids : une secousse visible sur la loss, l'oubli le plus coûteux. Sans le générateur du "
            "`DataLoader`, l'ordre des mini-batches change, et sans le générateur global, les masques de dropout : "
            "l'entraînement reste bon, mais n'est plus reproductible. Les deux générateurs se sauvent comme des "
            "tenseurs d'octets (`get_rng_state`, `get_state`), lisibles avec `weights_only=True`."),

    Ex("23.24", "📦", 3, 40, "Un modèle pré-entraîné torchvision : transforms, gel et nouvelle tête",
       "réutiliser un ResNet-18 entraîné sur ImageNet : préparer ses entrées, le geler et entraîner une nouvelle tête.",
       "Ex 23.23 · ch. 21 (ResNet, `weights.transforms()`) · ⚖️ 23.10 · fiche §23.9.4, §23.9.5", thread="Fashion-MNIST",
       tracks="C",
       body=r"""Un ResNet-18 entraîné sur ImageNet (1,2 million de photos en couleur, 1 000 classes) sait-il quelque chose d'utile sur de petites images grises de vêtements ? On le **gèle** et on n'entraîne qu'une nouvelle dernière couche (fiche §23.9.4). Comme le réseau gelé ne change pas, on calcule une fois pour toutes ses **features** (les 512 valeurs qui entrent dans sa dernière couche) sur 2 000 images d'entraînement et 1 000 de validation, puis on entraîne la tête sur ces features.

**Contrat**
- `make_transfer_24(weights, n_classes=10)` rend `resnet18(weights=weights)` dont **tous** les paramètres sont gelés, avec une nouvelle dernière couche `model.fc = nn.Linear(512, n_classes)`, la seule partie entraînable ;
- `to_resnet_input_24(images, weights, size)` transforme des images `uint8` `(N, 28, 28)` en tenseur `float32` `(N, 3, size, size)` : les pixels divisés par 255, le canal gris recopié dans les 3 canaux, redimensionnés en `size × size` (`antialias=True`), puis normalisés avec la moyenne et l'écart-type d'ImageNet de `weights.transforms()` (ses attributs `mean` et `std`) ;
- `extract_features_24(model, x, batch_size=250)` rend un tenseur `float32` `(N, 512)` : la sortie du réseau gelé **avant** `model.fc`, en mode évaluation, sans graphe de calcul, par mini-batches ; `model.fc` est inchangée après l'appel.

`SIZE_24` vaut 112 en mode rapide (4 fois moins de calculs) et 224, la taille d'ImageNet, sinon. La vérification compte tes paramètres, contrôle ton prétraitement et tes features, entraîne la tête 20 epochs sur les features, puis le MLP du livre sur les **mêmes** 2 000 images, et compare.

Dans tes notes : le modèle pré-entraîné fait-il beaucoup mieux que le MLP ? Pourquoi, d'après toi (pense à ce que sont les images d'ImageNet) ? Dans quel cas l'apprentissage par transfert donne-t-il un grand avantage ?""",
       given=GIVEN_24, todo=TODO_24, check=CHECK_24, solution=SOLUTION_24 + "\n\n\n" + CHECK_24,
       note="La tête du ResNet gelé (5 130 paramètres appris) fait un peu mieux que le MLP du livre (623 290 "
            "paramètres appris) sur les mêmes 2 000 images : environ 86 % contre 83 à 84 % en mode rapide, un "
            "avantage modeste. Les features "
            "d'ImageNet ont été apprises sur des photos en couleur d'objets du monde, à 224 pixels ; des vêtements "
            "gris de 28 pixels agrandis en sont loin (un déplacement de domaine). Le transfert donne un grand "
            "avantage quand les nouvelles images ressemblent à celles du pré-entraînement et qu'on en a peu "
            "(quelques centaines de photos d'un produit), et davantage encore quand on dégèle les dernières couches "
            "(le fine-tuning, chapitre bonus B1)."),
])

# ---------------------------------------------------------------------------
# Part D: callbacks, schedules, patience, the model card, the challenge (23.25 to 23.29)
# ---------------------------------------------------------------------------
TODO_25 = r'''class BestCheckpoint_25:
    """Callback for fit_18: when history[monitor][-1] is STRICTLY better than every earlier value ("min": smaller,
    "max": larger), save model.state_dict() to `path` with torch.save, and set best_ and best_epoch_ (the epoch
    received). Returns False. Before the first call, best_ is inf ("min") or -inf ("max") and best_epoch_ is None."""

    def __init__(self, path, monitor="val_loss", mode="min"):
        raise NotImplementedError("BestCheckpoint_25")

    def __call__(self, epoch, history, model, optimizer):
        raise NotImplementedError("BestCheckpoint_25.__call__")


class SchedulerCallback_25:
    """Callback for fit_18: append the learning rate used during the epoch (optimizer.param_groups[0]["lr"], read
    BEFORE stepping) to history["lr"] (created at the first call), then step the scheduler once:
    scheduler.step(history["val_loss"][-1]) for a ReduceLROnPlateau, scheduler.step() otherwise. Returns False."""

    def __init__(self, scheduler):
        raise NotImplementedError("SchedulerCallback_25")

    def __call__(self, epoch, history, model, optimizer):
        raise NotImplementedError("SchedulerCallback_25.__call__")


class EarlyStoppingCallback_25:
    """Callback for fit_18: mylearn.nn.regularization.EarlyStopping(patience, min_delta) (mode "min") stepped with
    history[monitor][-1]; returns True (stop) when it says so. The EarlyStopping object is the attribute `stopper`."""

    def __init__(self, patience, min_delta=0.0, monitor="val_loss"):
        raise NotImplementedError("EarlyStoppingCallback_25")

    def __call__(self, epoch, history, model, optimizer):
        raise NotImplementedError("EarlyStoppingCallback_25.__call__")'''

SOLUTION_25 = r'''class BestCheckpoint_25:
    """Callback for fit_18: when history[monitor][-1] is STRICTLY better than every earlier value ("min": smaller,
    "max": larger), save model.state_dict() to `path` with torch.save, and set best_ and best_epoch_ (the epoch
    received). Returns False. Before the first call, best_ is inf ("min") or -inf ("max") and best_epoch_ is None."""

    def __init__(self, path, monitor="val_loss", mode="min"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.path, self.monitor, self.mode = Path(path), monitor, mode
        self.best_ = math.inf if mode == "min" else -math.inf
        self.best_epoch_ = None

    def __call__(self, epoch, history, model, optimizer):
        value = history[self.monitor][-1]
        better = value < self.best_ if self.mode == "min" else value > self.best_
        if better:
            self.best_, self.best_epoch_ = value, epoch
            torch.save(model.state_dict(), self.path)
        return False


class SchedulerCallback_25:
    """Callback for fit_18: append the learning rate used during the epoch (optimizer.param_groups[0]["lr"], read
    BEFORE stepping) to history["lr"] (created at the first call), then step the scheduler once:
    scheduler.step(history["val_loss"][-1]) for a ReduceLROnPlateau, scheduler.step() otherwise. Returns False."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, epoch, history, model, optimizer):
        history.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(history["val_loss"][-1])
        else:
            self.scheduler.step()
        return False


class EarlyStoppingCallback_25:
    """Callback for fit_18: mylearn.nn.regularization.EarlyStopping(patience, min_delta) (mode "min") stepped with
    history[monitor][-1]; returns True (stop) when it says so. The EarlyStopping object is the attribute `stopper`."""

    def __init__(self, patience, min_delta=0.0, monitor="val_loss"):
        self.stopper = mylearn.nn.regularization.EarlyStopping(patience=patience, min_delta=min_delta)
        self.monitor = monitor

    def __call__(self, epoch, history, model, optimizer):
        return bool(self.stopper.step(history[self.monitor][-1]))'''

CHECK_25 = r'''with checked("23.25"):
    tiny_25 = nn.Linear(2, 1)
    tiny_opt_25 = torch.optim.SGD(tiny_25.parameters(), lr=1.0)
    path_25 = WORK_DIR / "best_25.pt"
    best_25 = BestCheckpoint_25(path_25)
    history_25 = {"val_loss": [], "val_acc": []}
    for epoch_25, loss_25 in enumerate([0.50, 0.40, 0.45, 0.30, 0.30, 0.35], start=1):
        with torch.no_grad():
            tiny_25.weight.fill_(epoch_25)                     # the weights "remember" the epoch
        history_25["val_loss"].append(loss_25)
        verdict_25 = best_25(epoch_25, history_25, tiny_25, tiny_opt_25)
    saved_25 = torch.load(path_25, weights_only=True)["weight"][0, 0].item()
    verdict("23.25", best_25.best_epoch_ == 4 and best_25.best_ == 0.30 and saved_25 == 4.0 and verdict_25 is False,
            "BestCheckpoint_25 garde l'epoch 4 (0,30), pas l'ex aequo de l'epoch 5, et rend False.",
            f"sur les val_loss [0.50, 0.40, 0.45, 0.30, 0.30, 0.35], attendu best_epoch_ 4, best_ 0.3 et les poids de "
            f"l'epoch 4 dans le fichier ; reçu {best_25.best_epoch_!r}, {best_25.best_!r} et ceux de l'epoch {saved_25:g}.")
    best_acc_25 = BestCheckpoint_25(WORK_DIR / "best_acc_25.pt", monitor="val_acc", mode="max")
    for epoch_25, acc_25 in enumerate([0.80, 0.85, 0.84], start=1):
        best_acc_25(epoch_25, {"val_acc": [acc_25]}, tiny_25, tiny_opt_25)
    verdict("23.25", best_acc_25.best_epoch_ == 2, "en mode « max », la meilleure accuracy (0,85, epoch 2).",
            f"mode='max' sur [0.80, 0.85, 0.84] : attendu best_epoch_ 2, reçu {best_acc_25.best_epoch_!r}.")
    with warnings.catch_warnings():                       # the scripted tests never call optimizer.step()
        warnings.simplefilter("ignore", UserWarning)
        for make_25, losses_25 in ((lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=2, gamma=0.5), [1.0] * 5),
                                   (lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o, factor=0.5, patience=0),
                                    [1.0, 0.9, 0.95, 0.96, 0.8])):
            lr_opt_25 = torch.optim.SGD(tiny_25.parameters(), lr=1.0)
            callback_25 = SchedulerCallback_25(make_25(lr_opt_25))
            hist_25 = {"val_loss": []}
            for epoch_25, loss_25 in enumerate(losses_25, start=1):
                hist_25["val_loss"].append(loss_25)
                callback_25(epoch_25, hist_25, tiny_25, lr_opt_25)
            ref_opt_25 = torch.optim.SGD(tiny_25.parameters(), lr=1.0)    # the same schedule, stepped by hand
            ref_sched_25, want_lr_25 = make_25(ref_opt_25), []
            for loss_25 in losses_25:
                want_lr_25.append(ref_opt_25.param_groups[0]["lr"])
                ref_sched_25.step(loss_25) if isinstance(ref_sched_25, torch.optim.lr_scheduler.ReduceLROnPlateau) else ref_sched_25.step()
            verdict("23.25", hist_25.get("lr") == want_lr_25,
                    f"{type(callback_25.scheduler).__name__} : learning rates {want_lr_25}.",
                    f"{type(make_25(torch.optim.SGD(tiny_25.parameters(), lr=1.0))).__name__} : attendu history['lr'] = "
                    f"{want_lr_25}, reçu {hist_25.get('lr')!r} (le learning rate de l'epoch, lu AVANT le pas du planning).")
    stop_cb_25 = EarlyStoppingCallback_25(patience=2)
    hist_25, answers_25 = {"val_loss": []}, []
    for loss_25 in [0.9, 0.8, 0.82, 0.79, 0.85, 0.86, 0.70]:
        hist_25["val_loss"].append(loss_25)
        answers_25.append(stop_cb_25(len(hist_25["val_loss"]), hist_25, tiny_25, tiny_opt_25))
    verdict("23.25", answers_25[:6] == [False, False, False, False, False, True],
            "EarlyStoppingCallback_25 (patience 2) demande l'arrêt à l'epoch 6.",
            f"patience 2 sur [0.9, 0.8, 0.82, 0.79, 0.85, 0.86, ...] : attendu [False] * 5 + [True], reçu {answers_25[:6]}.")
    train_25, val_25 = make_loaders()                          # a real run, with your fit_18 and the three callbacks
    torch.manual_seed(2325)
    model_25 = make_mlp()
    optimizer_25 = torch.optim.SGD(model_25.parameters(), lr=0.05, momentum=0.9)
    schedule_25 = torch.optim.lr_scheduler.LambdaLR(optimizer_25, lambda e: max(0.1, 1 - 0.05 * e))   # the book's shape
    checkpoint_25 = BestCheckpoint_25(WORK_DIR / "best_model_25.pt")
    stopping_25 = EarlyStoppingCallback_25(patience=4)
    started_25 = time.perf_counter()
    run_25 = fit_18(model_25, train_25, val_25, nn.CrossEntropyLoss(), optimizer_25, 40,
                    callbacks=[checkpoint_25, SchedulerCallback_25(schedule_25), stopping_25])
    model_25.load_state_dict(torch.load(WORK_DIR / "best_model_25.pt", weights_only=True))
    reloaded_25 = evaluate_18(model_25, val_25, nn.CrossEntropyLoss())
    print(f"{len(run_25['val_loss'])} epochs en {time.perf_counter() - started_25:.0f} s · meilleure epoch "
          f"{checkpoint_25.best_epoch_} (val_loss {checkpoint_25.best_:.4f}) · après rechargement : val_loss "
          f"{reloaded_25[0]:.4f}, accuracy {reloaded_25[1]:.4f}")
    verdict("23.25", abs(reloaded_25[0] - checkpoint_25.best_) < 1e-6,
            "le modèle rechargé est bien celui de la meilleure epoch (même val_loss).",
            f"le modèle rechargé a la val_loss {reloaded_25[0]:.6f}, attendu {checkpoint_25.best_:.6f} : le fichier "
            "contient-il les poids de la meilleure epoch ?")
    print("arrêt par l'early stopping :", "oui" if stopping_25.stopper.stopped_ else "non (les 40 epochs ont été faites)")
    fig, (ax1_25, ax2_25) = plt.subplots(1, 2, figsize=(10, 3))
    epochs_25 = range(1, len(run_25["val_loss"]) + 1)
    ax1_25.plot(epochs_25, run_25["train_loss"], label="entraînement")
    ax1_25.plot(epochs_25, run_25["val_loss"], label="validation")
    ax1_25.axvline(checkpoint_25.best_epoch_, color="0.5", ls="--", label="meilleure epoch")
    ax1_25.set_xlabel("epoch")
    ax1_25.set_ylabel("loss")
    ax1_25.legend()
    ax2_25.plot(epochs_25, run_25.get("lr", []), marker=".")
    ax2_25.set_xlabel("epoch")
    ax2_25.set_ylabel("learning rate")
    plt.show()'''

TODO_26 = r'''def make_scheduler_26(name, optimizer, epochs):
    """The learning-rate schedule `name` for `optimizer`, over `epochs` epochs:
    "constant": LambdaLR with a factor of 1 · "step": StepLR(step_size=max(1, epochs // 3), gamma=0.3)
    "cosine": CosineAnnealingLR(T_max=epochs) · "plateau": ReduceLROnPlateau(factor=0.3, patience=1).
    Any other name raises ValueError."""
    raise NotImplementedError("make_scheduler_26")'''

SOLUTION_26 = r'''def make_scheduler_26(name, optimizer, epochs):
    """The learning-rate schedule `name` for `optimizer`, over `epochs` epochs:
    "constant": LambdaLR with a factor of 1 · "step": StepLR(step_size=max(1, epochs // 3), gamma=0.3)
    "cosine": CosineAnnealingLR(T_max=epochs) · "plateau": ReduceLROnPlateau(factor=0.3, patience=1).
    Any other name raises ValueError."""
    schedules = torch.optim.lr_scheduler
    if name == "constant":
        return schedules.LambdaLR(optimizer, lambda e: 1.0)
    if name == "step":
        return schedules.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    if name == "cosine":
        return schedules.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return schedules.ReduceLROnPlateau(optimizer, factor=0.3, patience=1)
    raise ValueError(f"unknown schedule: {name!r}")'''

CHECK_26 = r'''EPOCHS_26 = 12 if FAST_MODE else 20
NAMES_26 = ["constant", "step", "cosine", "plateau"]
with checked("23.26"):
    dummy_26 = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.05)
    kinds_26 = {name: type(make_scheduler_26(name, torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.05), 12)).__name__
                for name in NAMES_26}
    verdict("23.26", kinds_26 == {"constant": "LambdaLR", "step": "StepLR", "cosine": "CosineAnnealingLR",
                                  "plateau": "ReduceLROnPlateau"},
            "make_scheduler_26 rend les quatre plannings demandés.", f"types reçus : {kinds_26}.")
    step_26 = make_scheduler_26("step", dummy_26, 12)
    verdict("23.26", step_26.step_size == 4 and step_26.gamma == 0.3, "par paliers : × 0,3 toutes les 4 epochs (12 // 3).",
            f"StepLR : attendu step_size 4 et gamma 0.3 pour 12 epochs, reçu {step_26.step_size} et {step_26.gamma}.")
    refuses("23.26", "un nom inconnu", lambda: make_scheduler_26("linear", dummy_26, 12))
    runs_26 = {}
    started_26 = time.perf_counter()
    for name_26 in NAMES_26:
        train_26, val_26 = make_loaders(seed=2326)
        torch.manual_seed(2326)
        model_26 = make_mlp()
        optimizer_26 = torch.optim.SGD(model_26.parameters(), lr=0.05, momentum=0.9)
        callback_26 = SchedulerCallback_25(make_scheduler_26(name_26, optimizer_26, EPOCHS_26))
        runs_26[name_26] = fit_18(model_26, train_26, val_26, nn.CrossEntropyLoss(), optimizer_26, EPOCHS_26,
                                  callbacks=[callback_26])
        h_26 = runs_26[name_26]
        print(f"{name_26:9s} val_acc finale {h_26['val_acc'][-1]:.4f} · meilleure {max(h_26['val_acc']):.4f} · "
              f"val_loss finale {h_26['val_loss'][-1]:.4f} · learning rate final {h_26['lr'][-1]:.4f}")
    print(f"({time.perf_counter() - started_26:.0f} s)")
    fig, axes_26 = plt.subplots(1, 3, figsize=(13, 3))
    for name_26, h_26 in runs_26.items():
        epochs_26 = range(1, len(h_26["val_acc"]) + 1)
        axes_26[0].plot(epochs_26, h_26["lr"], marker=".", label=name_26)
        axes_26[1].plot(epochs_26, h_26["val_loss"], label=name_26)
        axes_26[2].plot(epochs_26, h_26["val_acc"], label=name_26)
    for ax_26, title_26 in zip(axes_26, ["learning rate", "loss de validation", "accuracy de validation"]):
        ax_26.set_title(title_26)
        ax_26.set_xlabel("epoch")
    axes_26[0].legend()
    plt.show()'''

TODO_27 = r'''def simulate_27(val_losses, patience, min_delta=0.0):
    """(stop_epoch, best_epoch), epochs counted from 1: replay val_losses (one per epoch) in
    mylearn.nn.regularization.EarlyStopping(patience, min_delta); stop_epoch is the epoch whose step returned True
    (None if training would never stop), best_epoch the epoch whose weights would be reloaded."""
    raise NotImplementedError("simulate_27")'''

SOLUTION_27 = r'''def simulate_27(val_losses, patience, min_delta=0.0):
    """(stop_epoch, best_epoch), epochs counted from 1: replay val_losses (one per epoch) in
    mylearn.nn.regularization.EarlyStopping(patience, min_delta); stop_epoch is the epoch whose step returned True
    (None if training would never stop), best_epoch the epoch whose weights would be reloaded."""
    stopper = mylearn.nn.regularization.EarlyStopping(patience=patience, min_delta=min_delta)
    for epoch, loss in enumerate(val_losses, start=1):
        if stopper.step(loss):
            return epoch, stopper.best_epoch_ + 1                # best_epoch_ is a 0-based index
    return None, stopper.best_epoch_ + 1'''

CHECK_27 = r'''with checked("23.27"):
    losses_27 = HISTORY_19["val_loss"]
    acc_27 = HISTORY_19["val_acc"]
    verdict("23.27", tuple(simulate_27([0.5, 0.4, 0.45, 0.46], patience=2)) == (4, 2),
            "simulate_27 rejoue la règle : (arrêt, meilleure epoch) comptées à partir de 1.",
            f"sur [0.5, 0.4, 0.45, 0.46] avec une patience de 2, attendu (4, 2), reçu {simulate_27([0.5, 0.4, 0.45, 0.46], patience=2)!r}.")
    verdict("23.27", tuple(simulate_27([0.5, 0.4, 0.3], patience=2)) == (None, 3),
            "sans arrêt, stop_epoch vaut None.",
            f"sur [0.5, 0.4, 0.3] (toujours mieux), attendu (None, 3), reçu {simulate_27([0.5, 0.4, 0.3], patience=2)!r}.")
    print(" patience | min_delta | arrêt | epoch rechargée | val_acc rechargée | epochs entraînées après la meilleure")
    for patience_27, delta_27 in [(1, 0.0), (2, 0.0), (3, 0.0), (3, 0.01), (5, 0.0), (10, 0.0), (20, 0.0)]:
        stop_27, best_27 = simulate_27(losses_27, patience_27, delta_27)
        trained_27 = stop_27 if stop_27 is not None else len(losses_27)
        print(f"{patience_27:9d} | {delta_27:9.2f} | {str(stop_27):>5s} | {best_27:15d} | {acc_27[best_27 - 1]:17.4f} | "
              f"{trained_27 - best_27:d}")
    wb.check("23.27a", simulate_27(losses_27, 1)[0], computed=True)
    wb.check("23.27b", simulate_27(losses_27, 1)[1], computed=True)
    wb.check("23.27c", simulate_27(losses_27, 10)[0], computed=True)
    wb.check("23.27d", simulate_27(losses_27, 3, 0.01)[1], computed=True)'''

RECORD_27 = r'''losses_27 = HISTORY_19["val_loss"]
wb.record("23.27a", simulate_27(losses_27, 1)[0], mistakes={"compte les epochs à partir de 1": simulate_27(losses_27, 1)[0] - 1})
wb.record("23.27b", simulate_27(losses_27, 1)[1], mistakes={"best_epoch_ est un indice compté à partir de 0 : ajoute 1": simulate_27(losses_27, 1)[1] - 1,
          "b) demande l'epoch dont on recharge les poids, celle de la meilleure loss": simulate_27(losses_27, 1)[0]})
wb.record("23.27c", simulate_27(losses_27, 10)[0], mistakes={"compte les epochs à partir de 1": simulate_27(losses_27, 10)[0] - 1})
wb.record("23.27d", simulate_27(losses_27, 3, 0.01)[1], mistakes={"best_epoch_ est un indice compté à partir de 0 : ajoute 1": simulate_27(losses_27, 3, 0.01)[1] - 1,
          "avec min_delta = 0,01, une baisse plus petite que 0,01 ne compte pas comme une amélioration": simulate_27(losses_27, 3)[1]})'''

GIVEN_28 = r'''CARD_FACTS_28 = {                                     # what you need to know about MODEL_C (part C)
    "architecture": "Flatten, Linear(784, 784), ReLU, Linear(784, 10) (623 290 paramètres)",
    "training": "Adam (learning rate 0,001), mini-batches de 256, 5 epochs, graine 2321, PyTorch " + torch.__version__,
    "data": "Fashion-MNIST (Zalando, licence MIT) : 54 000 images d'entraînement, 6 000 de validation (découpage stratifié, random_state=0)",
    "preprocessing": f"pixels / 255, puis (x - {MEAN:.4f}) / {STD:.4f} (moyenne et écart-type du train)",
    "val_accuracy": round(VAL_ACC_C, 4),
    "classes": CLASSES,
}
REQUIRED_28 = ["Détails du modèle", "Usage prévu", "Facteurs", "Métriques", "Données d'évaluation",
               "Données d'entraînement", "Analyses quantitatives", "Considérations éthiques",
               "Mises en garde et recommandations"]     # the nine sections of Mitchell et al. (2019)
for key_28, value_28 in CARD_FACTS_28.items():
    print(f"{key_28:14s} {value_28}")'''

TODO_28 = r'''MODEL_CARD_28 = """# Fiche modèle : ...
"""   # your model card in Markdown: the nine sections of REQUIRED_28 as "## " headings, in this order'''

SOLUTION_28 = r'''with torch.inference_mode():                                       # per-class accuracy, for the quantitative analyses
    pred_28 = MODEL_C(X_VAL_T).argmax(dim=1).numpy()
per_class_28 = {CLASSES[k]: float(np.mean(pred_28[Y_VAL == k] == k)) for k in range(10)}
worst_28 = sorted(per_class_28.items(), key=lambda item: item[1])[:3]
MODEL_CARD_28 = f"""# Fiche modèle : MLP Fashion-MNIST (workbook, ch. 23)

## Détails du modèle
Perceptron multicouche : {CARD_FACTS_28["architecture"]}. Entraînement : {CARD_FACTS_28["training"]}. Sortie : 10 logits ; la classe prédite est leur argmax. Auteur : un apprenant du workbook, à des fins d'apprentissage.

## Usage prévu
Classer des images de vêtements de 28 × 28 pixels en niveaux de gris, centrées, sur fond noir, comme celles de Fashion-MNIST, parmi 10 classes : {", ".join(CLASSES)}. Usage pédagogique. Hors du champ : des photos prises au téléphone, des images en couleur, décentrées ou sur fond clair, toute décision qui concerne une personne.

## Facteurs
La classe du vêtement (les hauts se confondent entre eux), le cadrage (un décalage de 2 pixels fait perdre plus de 30 points d'accuracy), le contraste (des images inversées tombent sous le hasard) et l'orientation (le miroir gauche-droite coûte plus de 20 points).

## Métriques
Accuracy sur la validation (les 10 classes sont équilibrées), et accuracy par classe ; la matrice de confusion montre où se trouvent les erreurs.

## Données d'évaluation
Les 6 000 images de validation tirées du train de Fashion-MNIST (600 par classe). Le jeu de test (10 000 images) n'a pas servi.

## Données d'entraînement
{CARD_FACTS_28["data"]}. Prétraitement : {CARD_FACTS_28["preprocessing"]} ; il est sauvegardé avec le modèle (preprocessing.json) et doit être appliqué tel quel.

## Analyses quantitatives
Accuracy de validation : {CARD_FACTS_28["val_accuracy"]:.4f}. Les trois classes les moins bien reconnues : {", ".join(f"{name} ({acc:.1%})" for name, acc in worst_28)}.

## Considérations éthiques
Les images viennent du catalogue d'un vendeur européen : des vêtements surtout occidentaux, photographiés en studio ; un modèle de ce genre reconnaîtrait mal d'autres styles de vêtements. Aucune personne n'apparaît dans les images, mais un tel classifieur ne doit pas servir à décrire ou juger des personnes.

## Mises en garde et recommandations
Une confiance élevée ne garantit rien hors des données d'entraînement : des images inversées sont classées faux avec une confiance de 0,95. Pour un usage réel : un CNN, de l'augmentation de données, une évaluation sur des photos du terrain et une mesure finale sur le jeu de test, une seule fois.
"""'''

CHECK_28 = r'''def check_card_28(text):
    """Problems found in a model card (an empty list: none)."""
    problems = []
    if not isinstance(text, str):
        return ["MODEL_CARD_28 doit être une chaîne de caractères (du Markdown)."]
    headings = re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)
    missing = [h for h in REQUIRED_28 if h not in headings]
    if missing:
        problems.append(f"sections « ## » manquantes : {missing}")
    elif [h for h in headings if h in REQUIRED_28] != REQUIRED_28:
        problems.append("les neuf sections ne sont pas dans l'ordre de REQUIRED_28")
    words = len(re.findall(r"\w+", text))
    if words < 200:
        problems.append(f"{words} mots : une fiche utile en compte au moins 200")
    for label, needle in [("la moyenne du prétraitement", f"{MEAN:.4f}"), ("l'écart-type du prétraitement", f"{STD:.4f}"),
                          ("l'accuracy de validation", f"{VAL_ACC_C:.4f}"), ("le nom du dataset", "Fashion-MNIST"),
                          ("la licence des données", "MIT")]:
        if needle not in text:
            problems.append(f"{label} ({needle}) n'apparaît pas")
    return problems


if MODEL_CARD_28 is ... or len(re.findall(r"\w+", str(MODEL_CARD_28))) < 10:
    print("⏳ Ex 23.28 : pas encore fait (rédige MODEL_CARD_28).")
else:
    problems_28 = check_card_28(MODEL_CARD_28)
    verdict("23.28", not problems_28, "la fiche a ses neuf sections, les chiffres du modèle et au moins 200 mots.",
            "à compléter : " + " ; ".join(problems_28) + ".")
    if not problems_28:
        card_path_28 = (Path("model_card_mlp_fashion.md") if "mon_travail" in str(Path.cwd())   # next to your notebook
                        else WORK_DIR / "model_card_mlp_fashion.md")
        card_path_28.write_text(MODEL_CARD_28, encoding="utf-8")
        print(f"écrite dans {card_path_28.resolve()}")'''

CHECK_28_SOLUTION = CHECK_28                     # outside mon_travail, the card goes to WORK_DIR

GIVEN_29 = r'''X_TR_T29, Y_TR_T29 = preprocess(X_TR), torch.from_numpy(Y_TR)     # the 54 000 training images, preprocessed
X_TEST_T29, Y_TEST_T29 = preprocess(X_TEST), torch.from_numpy(Y_TEST)
GOAL_29, SPREAD_29, BUDGET_29 = 0.88, 0.005, 180.0                 # each run >= 88 %, max - min < 0.5 point, <= 3 min
ALLOWED_29 = (nn.Sequential, nn.Flatten, nn.Linear, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Dropout,
              nn.BatchNorm1d, nn.LayerNorm, nn.Identity)


def accuracy_29(model, x, y):
    model.eval()
    with torch.inference_mode():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def grade_29(train_fn, seeds=(0, 1, 2, 3, 4)):
    """Train one model per seed with train_fn(seed, X_TR_T29, Y_TR_T29); check that each one is an MLP; return the
    models, the validation accuracies and the total time."""
    started, models = time.perf_counter(), []
    for seed in seeds:
        model = train_fn(seed, X_TR_T29, Y_TR_T29)
        foreign = sorted({type(m).__name__ for m in model.modules() if not isinstance(m, ALLOWED_29)})
        if foreign:
            raise ValueError(f"only MLP layers are allowed, found: {foreign}")
        models.append(model)
    seconds = time.perf_counter() - started
    return models, [accuracy_29(m, X_VAL_T, Y_VAL_T) for m in models], seconds'''

TODO_29 = r'''def train_29(seed, x, y):
    """Build an MLP (allowed layers: ALLOWED_29) and train it on x (N, 1, 28, 28) float32 and y (N,) int64, everything
    random drawn from `seed`; return the trained model."""
    raise NotImplementedError("train_29")'''

SOLUTION_29 = r'''def train_29(seed, x, y, epochs=8, batch_size=256):
    """Build an MLP (allowed layers: ALLOWED_29) and train it on x (N, 1, 28, 28) float32 and y (N,) int64, everything
    random drawn from `seed`; return the trained model."""
    torch.manual_seed(seed)                                   # the weights, and the dropout if any
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)        # targets 0.91 / 0.01: less overconfident, more stable
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()                                    # OneCycleLR is stepped after each mini-batch
    return model'''

CHECK_29 = r'''with checked("23.29"):
    models_29, val_29, seconds_29 = grade_29(train_29)
    spread_29 = max(val_29) - min(val_29)
    print(f"validation : {[round(a, 4) for a in val_29]} · écart {spread_29 * 100:.2f} point(s) · {seconds_29:.0f} s")
    if seconds_29 > BUDGET_29:
        print(f"❌ Ex 23.29 : {seconds_29:.0f} s pour les cinq entraînements : il faut au plus {BUDGET_29:.0f} s.")
    elif min(val_29) < GOAL_29 or spread_29 >= SPREAD_29:
        print(f"❌ Ex 23.29 : sur la validation, il faut chaque accuracy ≥ {fr(GOAL_29 * 100, 0)} % et un écart < "
              f"{fr(SPREAD_29 * 100, 1)} point avant de révéler le test.")
    else:
        TEST_29 = globals().get("TEST_29")
        if TEST_29 is None:                                    # revealed once per session (an honour rule)
            TEST_29 = [accuracy_29(m, X_TEST_T29, Y_TEST_T29) for m in models_29]
        test_spread_29 = max(TEST_29) - min(TEST_29)
        print(f"test (révélé une fois par session) : {[round(a, 4) for a in TEST_29]} · écart {test_spread_29 * 100:.2f} point(s)")
        verdict("23.29", min(TEST_29) >= GOAL_29 and test_spread_29 < SPREAD_29,
                f"défi réussi : cinq graines, toutes ≥ {fr(GOAL_29 * 100, 0)} % sur le test, à moins de "
                f"{fr(SPREAD_29 * 100, 1)} point l'une de l'autre, en {seconds_29:.0f} s.",
                f"pas encore : sur le test, minimum {min(TEST_29):.4f}, écart {test_spread_29 * 100:.2f} point(s). Le "
                "résultat est figé pour cette session : rends ta recette plus stable sur la validation, sans la "
                "régler au plus près de ces cinq graines.")'''

PART_D = Part("D", "Callbacks, plannings, patience, fiche modèle et défi",
              "Tu écris en PyTorch les trois callbacks de Keras, pour ta `fit_18` (23.25) ; tu compares quatre "
              "plannings de learning rate (23.26) et plusieurs patiences d'early stopping (23.27) ; tu rédiges la "
              "fiche du modèle (23.28) ; puis tu relèves le défi de reproductibilité (23.29).",
              exercises=[
    Ex("23.25", "📦", 3, 45, "Les callbacks en PyTorch : meilleur checkpoint, planning de LR, early stopping",
       "écrire les trois callbacks de Keras pour ta fonction fit_18, et les combiner dans un vrai entraînement.",
       "Ex 23.18 · Ex 23.23 · ✏️ 23.6, 23.7 · ch. 20 (`EarlyStopping` de mylearn) · fiche §23.10 à §23.10.3",
       thread="Fashion-MNIST", tracks="R, C",
       body=r"""PyTorch n'a pas de callbacks ; ta `fit_18` en accepte (23.18) : des objets appelés `callback(epoch, history, model, optimizer)` à la fin de chaque epoch, dont un `True` arrête l'entraînement. Tu écris les trois du livre sous forme de classes : un objet garde son état (la meilleure valeur, le compteur) d'un appel à l'autre.

**Contrat**
- `BestCheckpoint_25(path, monitor="val_loss", mode="min")` : quand `history[monitor][-1]` est **strictement** meilleur que toutes les valeurs précédentes (plus petit en mode `"min"`, plus grand en `"max"`), il sauve `model.state_dict()` dans `path` avec `torch.save`, et met à jour `best_` et `best_epoch_` (l'epoch reçue) ; il rend `False`. Avant le premier appel, `best_` vaut `inf` (`"min"`) ou `-inf` (`"max"`) et `best_epoch_` vaut `None` ;
- `SchedulerCallback_25(scheduler)` ajoute à `history["lr"]` (créée au premier appel) le learning rate **utilisé pendant l'epoch**, `optimizer.param_groups[0]["lr"]` lu **avant** le pas du planning, puis fait un pas du planning : `scheduler.step(history["val_loss"][-1])` pour un `ReduceLROnPlateau`, `scheduler.step()` sinon ; il rend `False` ;
- `EarlyStoppingCallback_25(patience, min_delta=0.0, monitor="val_loss")` fait avancer un `mylearn.nn.regularization.EarlyStopping(patience, min_delta)` (ta classe du ch. 20, mode « min »), rangé dans l'attribut `stopper`, avec `history[monitor][-1]`, et rend `True` quand il demande l'arrêt.

La vérification essaie chaque callback sur des historiques écrits à la main, puis lance un vrai entraînement (SGD avec momentum, un planning linéaire avec plancher comme celui du livre, au plus 40 epochs) avec les trois, recharge le meilleur checkpoint et trace les courbes.

Dans tes notes : dans quel ordre faut-il passer les callbacks à `fit_18`, et cet ordre compte-t-il ici ? Après l'arrêt, pourquoi recharger le fichier du meilleur checkpoint plutôt que garder le modèle tel quel ?""",
       todo=TODO_25, check=CHECK_25, solution=SOLUTION_25 + "\n\n\n" + CHECK_25,
       note="L'ordre compte peu ici : chaque callback lit l'historique, déjà à jour, et `fit_18` les appelle tous "
            "avant de s'arrêter. Il compterait si l'un d'eux modifiait l'historique ou le modèle avant les autres "
            "(un callback qui rechargerait les meilleurs poids avant que le checkpoint ne soit écrit). À l'arrêt, "
            "le modèle a les poids de la **dernière** epoch, `patience` epochs après la meilleure : on recharge le "
            "fichier pour revenir à la meilleure epoch (ce que Keras appelle `restore_best_weights`)."),

    Ex("23.26", "🔬", 2, 30, "Planning de learning rate : constant, par paliers, cosinus ou plateau",
       "comparer quatre plannings de learning rate sur le même entraînement.",
       "Ex 23.25 · ✏️ 23.7 · ch. 19 (plannings) · fiche §23.10.2", thread="Fashion-MNIST", tracks="C",
       body=r"""Quatre fois le même entraînement (le MLP du livre, SGD avec momentum 0,9 et learning rate de départ 0,05, la même graine, les mêmes données), avec quatre plannings ; 12 epochs en mode rapide, 20 sinon (`EPOCHS_26`).

**Contrat** : `make_scheduler_26(name, optimizer, epochs)` rend le planning `name` pour `optimizer` :
- `"constant"` : un `LambdaLR` de facteur 1 (le learning rate ne bouge pas) ;
- `"step"` : `StepLR(step_size=max(1, epochs // 3), gamma=0.3)` ;
- `"cosine"` : `CosineAnnealingLR(T_max=epochs)` ;
- `"plateau"` : `ReduceLROnPlateau(factor=0.3, patience=1)` ;
- tout autre nom lève `ValueError`.

La vérification contrôle tes plannings, puis lance les quatre entraînements avec ta `fit_18` et ton `SchedulerCallback_25`, et trace le learning rate, la loss et l'accuracy de validation.

Dans tes notes : quels plannings finissent le mieux, et de combien ? Regarde la loss de validation des dernières epochs : que fait la baisse du learning rate ? Le planning « plateau » réagit-il au bon moment ? Avec un écart de l'ordre d'un point, que faudrait-il faire avant de conclure qu'un planning est meilleur (pense au défi 23.29) ?""",
       todo=TODO_26, check=CHECK_26, solution=SOLUTION_26 + "\n\n\n" + CHECK_26,
       note="Les trois plannings qui baissent le learning rate finissent environ un point au-dessus du learning rate "
            "constant : en fin d'entraînement, de petits pas laissent les poids se poser au fond d'une vallée au lieu "
            "d'osciller autour, et la loss de validation baisse nettement quand le learning rate tombe. Le cosinus et "
            "les paliers se valent ici ; le plateau ne réagit qu'après quelques epochs sans progrès. Un écart d'un "
            "point sur une seule graine ne prouve rien : il faut répéter avec plusieurs graines et comparer "
            "moyennes et écarts (23.29)."),

    Ex("23.27", "🔬", 2, 20, "Patience de l'early stopping : 1, 3 ou 10",
       "mesurer l'effet de la patience et de min_delta sur l'early stopping, en rejouant un historique de 50 epochs.",
       "Ex 23.19 · Ex 23.25 · ✏️ 23.6 · fiche §23.8.3, §23.10.3", thread="Fashion-MNIST", tracks="C",
       body=r"""Plutôt que de réentraîner le modèle pour chaque patience, on **rejoue** la loss de validation de `HISTORY_19` (23.19, 50 epochs) dans ton `EarlyStopping` : l'arrêt et la meilleure epoch ne dépendent que de cette suite de nombres.

**Contrat** : `simulate_27(val_losses, patience, min_delta=0.0)` rend `(stop_epoch, best_epoch)`, comptées **à partir de 1** : on donne chaque loss, dans l'ordre, à `mylearn.nn.regularization.EarlyStopping(patience, min_delta)` ; `stop_epoch` est l'epoch dont le `step` a rendu `True` (`None` si l'entraînement ne s'arrêterait jamais), `best_epoch` l'epoch dont on rechargerait les poids. Attention : l'attribut `best_epoch_` de ta classe est un **indice** compté à partir de 0 (sa docstring).

La vérification essaie ta fonction sur deux petits cas, affiche un tableau pour plusieurs patiences, puis vérifie : a) l'epoch d'arrêt avec une patience de 1 ; b) l'epoch rechargée avec une patience de 1 ; c) l'epoch d'arrêt avec une patience de 10 ; d) l'epoch rechargée avec une patience de 3 et `min_delta=0.01`.

Dans tes notes : que coûte une patience trop petite ici, et une patience trop grande ? Pourquoi `min_delta=0.01` change-t-il l'epoch rechargée ? Quelle patience choisirais-tu pour ce modèle ?""",
       todo=TODO_27, check=CHECK_27, solution=SOLUTION_27 + "\n\n\n" + CHECK_27, record=RECORD_27,
       note="Une patience de 1 s'arrête sur le premier soubresaut de la loss de validation (à l'epoch 3) et "
            "recharge l'epoch 2, deux points d'accuracy en dessous de la meilleure. Dès une patience de 2, l'arrêt "
            "trouve l'epoch 8 ; au-delà, une patience plus grande ne change rien au modèle rechargé et coûte "
            "seulement des epochs (10 de plus avec une patience de 10). `min_delta=0.01` refuse les petites "
            "améliorations : après l'epoch 4 (0,315), les baisses suivantes (0,315 puis 0,306 et 0,305) sont plus "
            "petites que 0,01 et ne comptent pas, le compteur atteint 3 à l'epoch 7, et l'on recharge l'epoch 4. Une patience de 3 à 5, sans `min_delta`, est un "
            "bon choix pour ce modèle."),

    Ex("23.28", "🛠️", 2, 20, "Rédiger la fiche du modèle (model card)",
       "documenter un modèle dans une fiche modèle complète, avec ses chiffres et ses limites.",
       "Ex 23.22 · Ex 23.23 · MP3 (fiche modèle) · fiche §23.9.5", thread="Fashion-MNIST", tracks="C",
       body=r"""Le livre conseille de documenter le prétraitement et de faire suivre cette documentation au modèle (fiche §23.9.5). La forme standard est la **fiche modèle** (*model card*, Mitchell et coll., 2019), que tu as rencontrée au MP3. Rédige celle de `MODEL_C`, en Markdown, dans la chaîne `MODEL_CARD_28`. La cellule suivante affiche ce qu'il faut savoir du modèle (`CARD_FACTS_28`) et la liste des neuf sections de l'article (`REQUIRED_28`).

**Contrat** : `MODEL_CARD_28` contient les neuf sections de `REQUIRED_28`, comme titres `## ...`, dans cet ordre, et au moins 200 mots ; on y trouve la moyenne et l'écart-type du prétraitement (4 décimales), l'accuracy de validation (4 décimales, celle de `CARD_FACTS_28`), le nom du dataset et sa licence. Une f-string à triples guillemets (`f` devant trois guillemets) t'évite de recopier les nombres à la main. Sers-toi de ce que tu as mesuré : les confusions de 23.20, les images « du monde réel » de 23.21.

La vérification relit ta fiche, puis l'écrit dans `model_card_mlp_fashion.md`, à côté de ton notebook (dans `mon_travail`) : elle a sa place dans ton portfolio.""",
       given=GIVEN_28, todo=TODO_28,
       after=[("check", CHECK_28), ("solution", SOLUTION_28 + "\n\n\n" + CHECK_28_SOLUTION)],
       note="Une bonne fiche ne se contente pas de l'accuracy globale : elle dit pour quelles images le modèle est "
            "fait, lesquelles il ne sait pas traiter (mesuré en 23.21), où il se trompe (23.20) et quel prétraitement "
            "appliquer. Les sections « Facteurs » et « Mises en garde » sont celles qui évitent les mauvaises "
            "surprises en production."),

    Ex("23.29", "🏆", 4, 90, "Défi reproductibilité : un MLP Fashion-MNIST à au moins 88 % dont l'accuracy varie de moins de 0,5 point sur 5 graines, en moins de 3 min de CPU",
       "trouver une recette d'entraînement de MLP à la fois bonne, stable d'une graine à l'autre et rapide.",
       "Ex 23.25 · Ex 23.26 · ch. 19 (plannings) · fiche §23.6.2, §23.8, « Au-delà du livre : la reproductibilité »",
       thread="Fashion-MNIST", tracks="C",
       body=r"""Un modèle n'est vraiment bon que si sa performance ne dépend pas de la chance d'une graine. Le défi : une recette `train_29(seed, x, y)` qui construit et entraîne un **MLP** (les couches de `ALLOWED_29` seulement : linéaires, activations, dropout, normalisations) sur les 54 000 images d'entraînement prétraitées, avec tout son hasard tiré de `seed`.

**Objectif**, pour les graines 0 à 4 :
- chaque accuracy **≥ 88 %**, et l'écart entre la meilleure et la moins bonne **< 0,5 point** ;
- les cinq entraînements **en 3 minutes au plus**, en tout, sur le CPU (celui de Colab ou d'un portable), mesurés en temps réel : lance la cellule sur une machine au repos ;
- d'abord sur la validation ; si c'est atteint, la cellule révèle le **test**, une seule fois par session (relancer la cellule montre le même résultat figé) : le défi est réussi si le test tient aussi les deux seuils.

Point de départ : le MLP du livre avec Adam à learning rate constant, 8 epochs, fait environ 88 % en moyenne sur le test, mais une graine sur cinq tombe vers 87,4 % : un écart de plus d'un point entre graines. À toi de jouer sur l'architecture, l'optimiseur, le planning (23.26), le nombre d'epochs, la taille des mini-batches, la régularisation… en ne regardant que la validation.

Dans tes notes : ta recette, les essais qui ont échoué, et ce qui a réduit l'écart entre graines.""",
       given=GIVEN_29, todo=TODO_29, check=CHECK_29, solution=SOLUTION_29 + "\n\n\n" + CHECK_29,
       note="La recette de référence : un MLP 784-512-256-10, Adam avec un planning `OneCycleLR` (le learning rate "
            "monte jusqu'à 0,003 puis redescend presque à zéro), 8 epochs de mini-batches de 256 et un label "
            "smoothing de 0,1 ; environ une minute pour les cinq graines sur 2 CPU, autour de 89,7 % sur le test "
            "avec moins de 0,2 point d'écart. La fin du planning, à learning rate très faible, est ce qui réduit le "
            "plus l'écart entre graines : chaque modèle finit posé au fond de sa vallée au lieu d'osciller autour, là "
            "où un learning rate constant laisse l'accuracy finale dépendre de l'endroit où s'arrête l'oscillation "
            "(sans planning, l'écart dépasse 0,7 point sur la validation). Le label smoothing, qui vise 0,91 au lieu de 1 pour la bonne "
            "classe, empêche le modèle de devenir trop sûr de lui et stabilise encore les résultats."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]
NEXT_SESSION: list = []


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 23.Q1 à 23.Q12, 23.R1 à 23.R3, 23.1 à 23.8 | vérifier tes réponses courtes | 🧠 🔁 ✏️ 🧮 | "
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
        title = "# 23 · PyTorch en pratique 1 : du dataset au modèle sauvegardé — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch23_pytorch_1/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 23`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 23 · PyTorch en pratique 1 : du dataset au modèle sauvegardé — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Préparer un dataset d'images pour PyTorch : formes `(N, C, H, W)`, type `float32`, normalisation "
               "avec les statistiques du train, `Dataset` et `DataLoader`.\n"
               "- Écrire ta propre fonction `fit()` (boucle, validation, historique), puis analyser les erreurs du "
               "modèle et prédire sur de nouvelles images.\n"
               "- Sauvegarder et recharger un modèle, reprendre un entraînement, réutiliser un modèle "
               "pré-entraîné, et écrire en PyTorch les callbacks de Keras (meilleur checkpoint, planning de "
               "learning rate, early stopping).\n\n"
               "**Rappel express.** Indice plat d'un élément $(i, j, k)$ de forme $(D_1, D_2, D_3)$ : "
               "$i\\,D_2 D_3 + j\\,D_3 + k$. Normalisation : $x' = (x/255 - \\mu_{\\text{train}})/"
               "\\sigma_{\\text{train}}$. Couche dense : $(n_{in} + 1)\\,n_{out}$ paramètres. Cross-entropy sur des "
               "logits : $L = -z_y + \\ln \\sum_j e^{z_j}$. Mini-batches par epoch : $\\lceil N/B \\rceil$. "
               "`LambdaLR` : $\\eta_e = \\eta_0 \\cdot \\lambda(e)$. Coût d'une epoch : $\\approx 6PN$ opérations.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code 23.12 à 23.29 (parties A à D : le pipeline Fashion-MNIST, ta fonction "
             "`fit()`, la sauvegarde, le modèle pré-entraîné et les callbacks) seront ajoutés à la prochaine session "
             "de génération ; relance alors `python tools/start_chapter.py 23` pour obtenir le notebook complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu préparer des images pour PyTorch (forme, type, normalisation, `DataLoader`) et "
               "expliquer pourquoi `reshape` ne remplace pas `permute` ?\n"
               "2. Sais-tu écrire une boucle d'entraînement avec validation, et dire ce que font `model.eval()` et "
               "`torch.inference_mode()` ?\n"
               "3. Sais-tu sauvegarder un modèle pour le reprendre ou le livrer, avec son prétraitement, et "
               "programmer un meilleur checkpoint, un planning et un early stopping ?\n\n"
               "**Pour aller plus loin** : les tutoriels « Learn the Basics » de PyTorch, cités dans la fiche. La "
               "suite : le ch. 24, PyTorch en pratique 2 (CNN, RNN, augmentation de données)." + later)]


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
