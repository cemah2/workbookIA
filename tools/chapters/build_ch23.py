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
         r'''mistakes={"le premier nombre de mat1 est la taille du mini-batch ; le second, le nombre de features": 784,
          "748 est la première dimension de mat2, la matrice de poids": 748,
          "10 est le nombre de sorties de la dernière couche": 10}'''),
        ("d", WHOLE, "784",
         r'''mistakes={"748 est justement la faute de frappe": 748,
          "in_features est le nombre de valeurs reçues par image, pas la taille du mini-batch": 32,
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
          "c'est channels_last : en channels_first, les canaux viennent d'abord": "(48, 64, 3)"}'''),
        ("b", SHAPE, '"(48, 64, 3)"',
         r'''mistakes={"la hauteur (le nombre de lignes) vient avant la largeur": "(64, 48, 3)",
          "c'est channels_first : en channels_last, les canaux viennent à la fin": "(3, 48, 64)"}'''),
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
          "la loss calcule elle-même la log-softmax": "D"}'''),
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
         r'''mistakes={"additionne les deux couches": dense(784, 784), "n'oublie pas les biais": 784 * 784 + 784 * 10}'''),
        ("d", DEC3, "dense(784, 784) / (dense(784, 784) + dense(784, 10))",
         r'''decimals=3, mistakes={"a) divisé par c), pas l'inverse": (dense(784, 784) + dense(784, 10)) / dense(784, 784),
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
         r'''decimals=3, mistakes={"L = −z_y + ln Σ e^z : n'oublie pas le terme −z_y": cross_entropy(Z_234, 1) + 2,
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
         r'''mistakes={"avec drop_last, on arrondit en dessous": math.ceil(N_235 / 256)}'''),
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
         r'''mistakes={"l'epoch 11 est une amélioration, qui remet le compteur à 0": 12}'''),
        ("f", WHOLE, "BEST_236E",
         r'''mistakes={"on recharge les poids de la dernière amélioration": 16,
          "l'epoch 11 (0,37) bat 0,379": 7}'''),
        ("g", WHOLE, "STOP_236A",
         r'''mistakes={"Keras 2.0 ne recharge rien : il garde les poids de la dernière epoch entraînée": 7}'''),
    ]),
    Paper("23.7", "Plannings de learning rate : le planning du livre avec LambdaLR", [
        ("a", DEC2, "1.0 * book_lambda(25)",
         r'''decimals=2, mistakes={"1 − 0,01 × 25 = 0,75": 0.25,
          "pendant l'epoch 25, le planning a reçu 25 appels (epochs comptées depuis 0)": book_lambda(24)}'''),
        ("b", WHOLE, "min(e for e in range(200) if book_lambda(e) <= 0.1)",
         r'''mistakes={"les epochs se comptent à partir de 0 dans cet exercice": 91}'''),
        ("c", DEC4, "0.01 * book_lambda(50)",
         r'''decimals=4, mistakes={"LambdaLR multiplie λ par le learning rate de l'optimiseur": book_lambda(50),
          "pendant l'epoch 50, le planning a reçu 50 appels : λ(50)": 0.01 * book_lambda(49),
          "les epochs se comptent à partir de 0 : c'est λ(50) qu'il faut": 0.01 * book_lambda(51)}'''),
        ("d", WHOLE, "round(book_lambda(0) / 0.05)",
         r'''mistakes={"il faut 0,05 × λ(0) = 1 : λ(0) = 1 / 0,05": 1}'''),
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
# Parts A to D (exercises 23.12 to 23.29): next generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "23.12 à 23.16", "l'environnement et les tenseurs, Fashion-MNIST regardé, oublier la conversion ou la "
                           "normalisation, tout le prétraitement au même endroit, le MLP du livre"),
    ("B", "23.17 à 23.20", "trois bugs d'inférence, ta fonction fit() et son historique, lire l'historique, "
                           "l'analyse des erreurs"),
    ("C", "23.21 à 23.24", "des images « du monde réel », prédire, reprendre un entraînement interrompu, un modèle "
                           "pré-entraîné de torchvision"),
    ("D", "23.25 à 23.29", "les callbacks en PyTorch, les plannings de learning rate, la patience de l'early "
                           "stopping, la fiche modèle, le défi reproductibilité"),
]


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
