#!/usr/bin/env python
"""Build the two notebooks of bonus chapter B1 from a single source (used by Claude).

    python tools/chapters/build_chB1.py
    python tools/run_all_notebooks.py chapitres/b1_transfer/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/b1_transfer/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q11, of the recalls R1 to R3 and of the ✏️ paper exercises
B1.1 to B1.5 (02_exercices.md). The derivation ∂ B1.6, the estimate 🧮 B1.7, the reflection (🗣️ B1.8, ⚖️ B1.9,
📄 B1.10) and the interview are corrected with 05_solutions.md. The answers of the ✏️ are computed by hand formulas
and checked against PyTorch, torchvision and scikit-learn in the context cell of the solutions notebook (no download:
`resnet18(weights=None)`).
Parts A to D (exercises B1.11 to B1.28) come at the second generation session (NEXT_SESSION): load and probe,
fine-tune, when transfer helps or fails, foundation models, cost and delivery.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Paper, badge, check_unique_keys, code, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "B1"
FOLDER = "chapitres/b1_transfer"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import math

import numpy as np


# ✏️ B1.1: side of a map after a convolution or a pooling, floor((n + 2p - k) / s) + 1 (ch. 21)
def out_side(n, k, s, p):
    return (n + 2 * p - k) // s + 1


def resnet18_sides(n):
    """Sides after the stem, layer1, layer2, layer3 and layer4 of a ResNet-18 for an n x n image."""
    stem = out_side(out_side(n, 7, 2, 3), 3, 2, 1)          # conv1, then maxpool
    l2 = out_side(stem, 3, 2, 1)                             # layer1 keeps the size; layer2 to 4 start with stride 2
    l3 = out_side(l2, 3, 2, 1)
    return {"stem": stem, "layer1": stem, "layer2": l2, "layer3": l3, "layer4": out_side(l3, 3, 2, 1)}


def shape(*axes):
    return "(" + ", ".join(str(a) for a in axes) + ")"


S128_B11 = resnet18_sides(128)

# ✏️ B1.4: parameters of ResNet-18 (convolutions without bias; gamma and beta in each batchnorm)
def conv_weights(c_in, c_out, k):
    return c_in * c_out * k * k


def bn_params(c):
    return 2 * c


BLOCK0_B14 = (conv_weights(256, 512, 3) + bn_params(512) + conv_weights(512, 512, 3) + bn_params(512)
              + conv_weights(256, 512, 1) + bn_params(512))
BLOCK1_B14 = 2 * (conv_weights(512, 512, 3) + bn_params(512))
LAYER4_B14 = BLOCK0_B14 + BLOCK1_B14
IMAGENET_B14 = 11_689_512                                    # weights.meta["num_params"], ImageNet head included
HEAD10_B14 = 512 * 10 + 10
NET10_B14 = IMAGENET_B14 - (512 * 1000 + 1000) + HEAD10_B14
TRAIN_B14 = LAYER4_B14 + HEAD10_B14

# ✏️ B1.2: six embeddings; ties of votes go to the first class in alphabetical order (KNeighborsClassifier)
E_B12 = np.array([[3, 0, 4], [0, 3, 4], [16, 2, 8], [2, 1, 2], [1, 0, 0], [0, 1, 1]], dtype=float)
Y_B12 = np.array(["oiseau", "avion", "avion", "oiseau", "avion", "oiseau"])
Q_B12 = np.array([1, 2, 2], dtype=float)
COS_B12 = E_B12 @ Q_B12 / (np.linalg.norm(E_B12, axis=1) * np.linalg.norm(Q_B12))


def knn_cosine_b12(k):
    nearest = np.argsort(-COS_B12, kind="stable")[:k]
    classes, votes = np.unique(Y_B12[nearest], return_counts=True)      # classes in alphabetical order
    return str(classes[np.argmax(votes)])                                # argmax: the first of the tied classes


# ✏️ B1.3: discriminative learning rates (head, layer4, layer3, layer2), warmup then cosine, in units of 1e-5
HEAD_LR_B13, GAMMA_B13, T_WARMUP_B13, T_TOTAL_B13 = 100.0, 0.3, 5, 105


def factor_b13(t):
    if t < T_WARMUP_B13:
        return t / T_WARMUP_B13
    progress = (t - T_WARMUP_B13) / (T_TOTAL_B13 - T_WARMUP_B13)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def lr_b13(depth, t):
    """depth 0 = the head, 1 = layer4, 2 = layer3, 3 = layer2 (that is, L - l)."""
    return HEAD_LR_B13 * GAMMA_B13 ** depth * factor_b13(t)


# ✏️ B1.5: zero-shot CLIP, classes (A) airplane, (B) bird, (C) ship, (D) truck
COS_B15 = np.array([0.24, 0.30, 0.28, 0.19])


def zero_shot_b15(s, cos=COS_B15):
    z = s * cos
    e = np.exp(z - z.max())
    return e / e.sum()


P100_B15 = zero_shot_b15(math.exp(4.6052))
PINIT_B15 = zero_shot_b15(math.exp(2.6592))
'''

LIBRARY_CHECKS = '''
# The same answers from the libraries (no download: weights=None gives the architecture)
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchvision.models import resnet18

net = resnet18(weights=None).eval()
assert sum(p.numel() for p in net.parameters()) == IMAGENET_B14
net.fc = nn.Linear(net.fc.in_features, 10)
assert sum(p.numel() for p in net.parameters()) == NET10_B14
assert sum(p.numel() for p in net.layer4.parameters()) == LAYER4_B14
for size in (128, 32, 100):
    with torch.inference_mode():
        h = net.maxpool(net.relu(net.bn1(net.conv1(torch.zeros(16, 3, size, size)))))
        sides = {"stem": h.shape[-1]}
        for name in ("layer1", "layer2", "layer3", "layer4"):
            h = getattr(net, name)(h)
            sides[name] = h.shape[-1]
        logits = net(torch.zeros(16, 3, size, size))
    assert sides == resnet18_sides(size) and tuple(logits.shape) == (16, 10), (size, sides)

for k in range(1, 7):
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(E_B12, Y_B12)
    assert knn.predict(Q_B12[None])[0] == knn_cosine_b12(k), k

groups = [nn.Parameter(torch.zeros(1)) for _ in range(4)]
optimizer = torch.optim.AdamW([{"params": [p], "lr": 1e-3 * GAMMA_B13 ** d} for d, p in enumerate(groups)])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor_b13)
for t in range(60):
    assert np.allclose([pg["lr"] / 1e-5 for pg in optimizer.param_groups], [lr_b13(d, t) for d in range(4)])
    optimizer.step()
    scheduler.step()

assert np.allclose(torch.softmax(torch.exp(torch.tensor(4.6052)) * torch.tensor(COS_B15), 0).numpy(), P100_B15)
print("✅ the paper answers agree with PyTorch, torchvision and scikit-learn")
'''

exec(PAPER_CONTEXT)  # the builder computes a few mistakes with the same helpers
exec(LIBRARY_CHECKS)

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "B"'
LETTERS = 'the letters, e.g. "AC"'
WHOLE = "a whole number"
NUMBER = "a number"
DEC1 = "a number, 1 decimal"
DEC2 = "a number, 2 decimals"
DEC3 = "a number, 3 decimals"
DEC4 = "a number, 4 decimals"
SHAPE = 'a shape as a string, e.g. "(2, 3)"'
CLASS = 'a class name as a string, e.g. "chat"'
INDEX = "a vector number, from 0 to 5"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


def index_mistakes(answer: int, hints: dict) -> str:
    """Mistakes for a vector number of Ex B1.2: a message for every other number (else `wb.check` would say « one
    off »), and a hint for a number that is not whole."""
    entries = ", ".join(f'"{message}": {i}' for i, message in hints.items() if i != answer)
    return ("mistakes={" + entries + ', "il n\'y a que six vecteurs, numérotés de 0 à 5": 6}, '
            'fractional="on demande le numéro d\'un vecteur, de 0 à 5"')


PAPER = [
    Paper("B1.Q1", "Réutiliser plutôt que repartir de zéro : trois raisons et une limite", [
        ("a", LETTERS, '"ABD"', r'''mistakes={"un modèle pré-entraîné hérite des biais de ses données d'origine (fiche §201.7)": "ABCD",
          "une raison juste manque : relis ce que savent déjà faire les premières couches": "AB",
          "une raison juste manque : relis ce que coûte l'entraînement d'un grand réseau": "AD",
          "une raison juste manque : relis combien d'images étiquetées demande un ResNet entraîné de zéro": "BD"}'''),
        ("b", WHOLE, "400", r'''mistakes={"on demande un nombre de millions : 1 million s'écrit 1": 400_000_000,
          "142 millions, c'est le dataset de DINOv2": 142,
          "1,28 million, ce sont les images d'ImageNet-1k": 1},
          fractional="on demande un nombre entier de millions de paires"'''),
        ("c", TF, "False", r'''mistakes={"relis le §201.1 : ce que 500 images font à un réseau de millions de paramètres (ch. 9)": True}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "un modèle pré-entraîné tourne sur un CPU : tout le notebook du chapitre le fait",
                                           "C": "remplacer la tête est justement le premier geste du transfert"})),
        ("e", TF, "True", r'''mistakes={"relis le §201.1 : qui a payé les jours de calcul sur des GPU ?": False}'''),
    ]),
    Paper("B1.Q2", "Premières couches génériques, dernières couches spécialisées : vrai ou faux", [
        ("a", TF, "True", r'''mistakes={"relis la hiérarchie des features (§201.1) : on retrouve ces filtres dans presque tous les CNN": False}'''),
        ("b", TF, "False", r'''mistakes={"relis la hiérarchie des features : où se trouvent les features les plus liées aux 1 000 classes ?": True}'''),
        ("c", TF, "True", r'''mistakes={"relis la règle qui suit la hiérarchie des features (§201.1)": False}'''),
        ("d", TF, "False", r'''mistakes={"relis le §201.7 : quelles couches deviennent les premières inutiles sur un domaine lointain ?": True}'''),
        ("e", TF, "True", r'''mistakes={"relis d'où vient le mot « probe » (§201.3)": False}'''),
    ]),
    Paper("B1.Q3", "Où trouver un modèle pré-entraîné, et quoi lire avant de s'en servir", [
        ("a", LETTER, '"C"', letters("C", {"A": "torchvision n'a pas DINOv2 : il vient du Hugging Face Hub",
                                           "B": "torch.load relit un fichier déjà sur le disque, il ne télécharge rien"})),
        ("b", LETTERS, '"ABC"', r'''mistakes={"la fiche modèle décrit le modèle, pas ta machine ni ta connexion": "ABCD",
          "une rubrique de la fiche modèle manque : ce qu'on a le droit d'en faire": "BC",
          "une rubrique de la fiche modèle manque : d'où viennent ses données": "AC",
          "une rubrique de la fiche modèle manque : les usages hors du champ prévu (out of scope)": "AB"}'''),
        ("c", TF, "False", r'''mistakes={"relis l'encadré ⚠️ « public n'est pas libre » (§201.2)": True}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "c'était la licence de la sortie, en avril 2023 : relis le tableau du §201.2",
                                           "C": "c'est CLIP qui n'affiche aucune licence, pas DINOv2"})),
        ("e", TF, "True", r'''mistakes={"relis l'encadré sur l'API transformers : revision= désigne un commit du dépôt": False}'''),
    ]),
    Paper("B1.Q4", "weights.transforms() : pourquoi le prétraitement fait partie du modèle", [
        ("a", LETTER, '"C"', letters("C", {"A": "la normalisation se fait sur des valeurs déjà dans [0, 1] : elle ne peut pas venir en premier",
                                           "B": "on ne recadre pas avant d'avoir redimensionné : relis les quatre étapes de l'encadré"})),
        ("b", TF, "False", r'''mistakes={"les formes ne changent pas quand on oublie la normalisation : relis le §201.5": True}'''),
        ("c", WHOLE, "232", r'''mistakes={"224 est la taille du recadrage (crop_size), pas du redimensionnement": 224,
          "256 est le redimensionnement d'autres jeux de poids : relis l'encadré sur weights.transforms()": 256}'''),
        ("d", LETTER, '"A"', letters("A", {"B": "chaque modèle a ses statistiques : CLIP n'a pas été entraîné avec celles d'ImageNet",
                                           "C": "on normalise avec les statistiques du pré-entraînement, pas celles de tes données"})),
        ("e", TF, "True", r'''mistakes={"compare les deux jeux de poids de ResNet-50 cités dans l'encadré du §201.2": False}'''),
    ]),
    Paper("B1.Q5", "Extraction de features, linear probe, fine-tuning : qui s'entraîne ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "le backbone reste gelé en extraction de features",
                                           "C": "le backbone, batchnorms comprises, ne change pas en extraction de features"})),
        ("b", LETTER, '"C"', letters("C", {"A": "tout le réseau, c'est le fine-tuning complet",
                                           "B": "la tête seule, backbone gelé, c'est l'extraction de features"})),
        ("c", TF, "True", r'''mistakes={"un k-NN garde les exemples et compare : y a-t-il des poids à régler ?": False}'''),
        ("d", WHOLE, "768 * 7 + 7", r'''mistakes={"n'oublie pas les biais, un par classe": 768 * 7,
          "chaque classe a un poids par composante de l'embedding": 768 + 7}'''),
        ("e", TF, "False", r'''mistakes={"un embedding figé ne change jamais : relis le §201.3": True}'''),
    ]),
    Paper("B1.Q6", "Pourquoi un learning rate 10 à 100 fois plus petit pour fine-tuner ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "AdamW n'a pas de limite de ce genre : la raison tient aux poids de départ",
                                           "C": "un plus petit learning rate fait des pas plus courts : il ne converge pas plus vite"})),
        ("b", TF, "True", r'''mistakes={"relis le §201.4 : que vaut la loss quand la tête est encore aléatoire ?": False}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "LP-FT commence par la tête : relis l'ordre des deux lettres",
                                           "C": "LP-FT fait les deux étapes l'une après l'autre, une seule fois"})),
        ("d", TF, "False", r'''mistakes={"relis la formule des learning rates discriminatifs : vers quelle extrémité décroissent-ils ?": True}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "le warmup change le learning rate, pas la mémoire",
                                           "B": "le warmup ne touche pas aux batchnorms"})),
    ]),
    Paper("B1.Q7", "Batchnorm pendant le fine-tuning sur de tout petits batches", [
        ("a", LETTER, '"B"', letters("B", {"A": "les statistiques glissantes servent en mode eval(), pas en mode train()",
                                           "C": "une batchnorm calcule ses statistiques, elle ne les fixe pas"})),
        ("b", TF, "True", r'''mistakes={"relis les deux régimes de la batchnorm (§201.4)": False}'''),
        ("c", TF, "False", r'''mistakes={"relis la remarque sur l'ordre des appels sous freeze_batchnorm (§201.4)": True}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "les formes ne dépendent pas du mode de la batchnorm",
                                           "B": "des statistiques bruitées n'empêchent pas le calcul : le problème apparaît à l'évaluation"})),
        ("e", TF, "True", r'''mistakes={"eval() change la façon de normaliser, pas requires_grad : relis la parade du §201.4": False}'''),
    ]),
    Paper("B1.Q8", "Peu ou beaucoup de données, domaine proche ou lointain : la bonne stratégie", [
        ("a", LETTER, '"A"', letters("A", {"B": "le learning rate d'un entraînement de zéro détruit les features pré-entraînées",
                                           "C": "300 images ne suffisent pas à entraîner un ResNet de zéro"})),
        ("b", LETTER, '"B"', letters("B", {"A": "des pixels bruts ne forment pas un bon embedding",
                                           "C": "avec 200 000 images d'un domaine proche, fine-tuner rapporte davantage"})),
        ("c", LETTER, '"C"', letters("C", {"A": "des spectrogrammes sont loin des photos d'ImageNet : ce n'est pas le cas facile",
                                           "B": "150 images ne suffisent pas à entraîner un ResNet-50 de zéro"})),
        ("d", TF, "True", r'''mistakes={"relis la case « beaucoup de données, domaine lointain » de la matrice de décision": False}'''),
        ("e", TF, "False", r'''mistakes={"relis le protocole des courbes d'apprentissage (§201.6) : que vaut une mesure sur 10 images tirées une fois ?": True}'''),
    ]),
    Paper("B1.Q9", "Transfert négatif et oubli catastrophique : reconnaître les symptômes", [
        ("a", LETTER, '"B"', letters("B", {"A": "une accuracy qui baisse d'une epoch à l'autre, c'est de l'overfitting ou du bruit",
                                           "C": "le transfert négatif compare deux résultats finaux, pas des gradients"})),
        ("b", LETTER, '"A"', letters("A", {"B": "le modèle va bien sur sa nouvelle tâche : c'est l'ancienne qui est perdue",
                                           "C": "l'overfitting compare l'entraînement et le test d'une même tâche"})),
        ("c", TF, "True", r'''mistakes={"relis le §201.7 : à quoi compare-t-on un transfert pour dire qu'il nuit ?": False}'''),
        ("d", TF, "False", r'''mistakes={"relis les parades contre l'oubli (§201.7)": True}'''),
        ("e", LETTER, '"C"', letters("C", {"A": "L2-SP tire les poids vers leurs valeurs pré-entraînées",
                                           "B": "des learning rates discriminatifs bougent peu les premières couches"})),
    ]),
    Paper("B1.Q10", "Classer sans exemple d'entraînement : le zéro-shot de CLIP", [
        ("a", LETTER, '"A"', letters("A", {"B": "le zéro-shot n'utilise aucune image étiquetée",
                                           "C": "CLIP n'a pas de tête ImageNet : il fabrique sa tête à partir de mots"})),
        ("b", TF, "False", r'''mistakes={"relis le §201.8 : la valeur de config.json est celle du départ de l'entraînement": True}'''),
        ("c", TF, "True", r'''mistakes={"les cosinus d'une image avec les légendes sont très proches : que fait une softmax de nombres proches ?": False}'''),
        ("d", LETTER, '"C"', letters("C", {"A": "un ensemble de prompts porte sur les phrases des classes, pas sur les images",
                                           "B": "un seul CLIP suffit : on moyenne des embeddings de texte"})),
        ("e", TF, "False", r'''mistakes={"relis le §201.8 : sur quoi CLIP a-t-il été entraîné ?": True}'''),
    ]),
    Paper("B1.Q11", "Auto-supervisé (DINOv2) ou supervisé sur ImageNet : quels embeddings ?", [
        ("a", LETTER, '"B"', letters("B", {"A": "rapprocher une image de sa légende, c'est l'apprentissage contrastif de CLIP",
                                           "C": "DINOv2 n'utilise aucun label, ni ceux d'ImageNet ni d'autres"})),
        ("b", TF, "True", r'''mistakes={"relis le §201.8 : d'où viennent les modèles small, base et large ?": False}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "pixel_values est l'entrée du modèle, pas sa sortie",
                                           "C": "ces tokens donnent un vecteur par morceau d'image, pas un par image"})),
        ("d", TF, "False", r'''mistakes={"relis la fin du §201.9 : le classement ImageNet est un indice, pas une preuve": True}'''),
        ("e", WHOLE, "(448 // 14) ** 2 + 1", r'''mistakes={"n'oublie pas le token CLS": (448 // 14) ** 2,
          "257 tokens, c'est pour une image de 224 × 224 : celle-ci est plus grande": 257,
          "compte les carrés sur toute l'image, pas sur un seul côté": 448 // 14 + 1}'''),
    ]),
    Paper("B1.R1", "Ch. 29 : licence d'un dataset et décalage de distribution", [
        ("a", TF, "False", r'''mistakes={"NC, c'est « non commercial » (ch. 29)": True}'''),
        ("b", LETTER, '"A"', letters("A", {"B": "ImageNet ne vend pas de licence commerciale : relis l'encadré 🕰️ sur les licences des poids",
                                           "C": "certaines images d'ImageNet sont protégées par le droit d'auteur"})),
        ("c", LETTER, '"C"', letters("C", {"A": "un découpage non stratifié change les proportions des classes, pas la distribution d'usage",
                                           "B": "un planning de learning rate n'a rien à voir avec les données"})),
        ("d", TF, "True", r'''mistakes={"relis le §201.7, deuxième paragraphe": False}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "plus d'epochs ne disent rien des données d'usage",
                                           "C": "un découpage du même dataset suit la même distribution : il ne voit pas le décalage"})),
    ]),
    Paper("B1.R2", "Ch. 23 : geler un réseau et remplacer sa tête en PyTorch", [
        ("a", LETTER, '"B"', letters("B", {"A": "eval() change le comportement de la batchnorm et du dropout, pas requires_grad",
                                           "C": "no_grad autour de step() ne gèle rien : les gradients des poids sont déjà calculés"})),
        ("b", TF, "False", r'''mistakes={"eval() change le mode de calcul de certaines couches : les poids reçoivent encore leur gradient": True}'''),
        ("c", WHOLE, "1280 * 4 + 4", r'''mistakes={"n'oublie pas les 4 biais": 1280 * 4,
          "la nouvelle tête a 4 sorties, pas 1 000": 1280 * 1000 + 1000,
          "chaque sortie a un poids par entrée": 1280 + 4}'''),
        ("d", TF, "True", r'''mistakes={"une couche neuve ne sait rien du gel des autres : quelle est la valeur par défaut de requires_grad ?": False}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "weights_only=False laisse pickle exécuter du code caché dans le fichier",
                                           "C": "pickle.load exécute ce que le fichier lui demande : jamais sur un fichier reçu"})),
    ]),
    Paper("B1.R3", "Ch. 21 : ce que détectent les filtres des premières et des dernières couches", [
        ("a", LETTER, '"B"', letters("B", {"A": "des parties d'objets apparaissent dans les couches profondes, pas dans conv1",
                                           "C": "les filtres d'un réseau entraîné sont structurés : relis 📦 21.27"})),
        ("b", WHOLE, "64 * 3 * 7 * 7", r'''mistakes={"les convolutions de ResNet n'ont pas de biais": 64 * 3 * 7 * 7 + 64,
          "chaque filtre voit les 3 canaux de l'image": 64 * 7 * 7,
          "il y a 64 filtres": 3 * 7 * 7}'''),
        ("c", TF, "True", r'''mistakes={"chaque stride 2 divise le côté de la carte : combien y en a-t-il avant layer4 ?": False}'''),
        ("d", TF, "False", r'''mistakes={"un filtre de layer4 a des centaines de canaux d'entrée : peut-on l'afficher comme une image en couleur ?": True}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "le pooling moyen global fait la moyenne sur les positions, pas sur les canaux",
                                           "C": "le pooling moyen global ne garde qu'une valeur par canal"})),
    ]),
    Paper("B1.1", "Nouvelle tête : formes des tenseurs de l'image 32×32 redimensionnée jusqu'aux logits", [
        ("a", SHAPE, 'shape(16, 64, S128_B11["stem"], S128_B11["stem"])', r'''mistakes={"le maxpool du stem divise aussi le côté par 2": "(16, 64, 64, 64)",
          "n'oublie pas l'axe des exemples": "(64, 32, 32)",
          "conv1 a un stride 2, puis maxpool aussi : relis le stem": "(16, 64, 128, 128)"}'''),
        ("b", SHAPE, 'shape(16, 128, S128_B11["layer2"], S128_B11["layer2"])', r'''mistakes={"layer2 commence par une convolution de stride 2": "(16, 128, 32, 32)",
          "layer1 garde la taille : relis l'énoncé": "(16, 128, 8, 8)",
          "relis le nombre de canaux de layer2 dans l'énoncé": "(16, 256, 16, 16)"}'''),
        ("c", SHAPE, 'shape(16, 512, S128_B11["layer4"], S128_B11["layer4"])', r'''mistakes={"compte les strides 2 de layer2, layer3 et layer4": "(16, 512, 8, 8)",
          "layer1 ne divise pas le côté de la carte": "(16, 512, 2, 2)",
          "n'oublie pas l'axe des exemples": "(512, 4, 4)"}'''),
        ("d", SHAPE, 'shape(16, 512)', r'''mistakes={"c'est la sortie d'avgpool : l'aplatissement retire les deux axes de taille 1": "(16, 512, 1, 1)",
          "avgpool fait d'abord la moyenne de chaque canal sur toutes ses positions": "(16, 8192)",
          "n'oublie pas l'axe des exemples": "(512,)"}'''),
        ("e", SHAPE, 'shape(16, 10)', r'''mistakes={"la tête d'ImageNet a été remplacée par une tête à 10 classes": "(16, 1000)",
          "n'oublie pas l'axe des exemples": "(10,)",
          "relis la question : on demande la sortie de fc": "(16, 512)"}'''),
        ("f", WHOLE, 'resnet18_sides(32)["layer4"]', r'''mistakes={"compte toutes les étapes de stride 2, celles du stem comprises": 4,
          "une étape a été oubliée : conv1 et maxpool divisent chacun le côté": 2,
          "une convolution avec padding garde au moins une position : applique la formule à la dernière étape": 0},
          fractional="le côté d'une carte est un nombre entier : applique la formule à chaque étape"'''),
        ("g", WHOLE, 'resnet18_sides(100)["layer4"]', r'''mistakes={"ni 100/32, ni des divisions par 2 arrondies vers le bas : applique la formule de l'énoncé à chaque étape, avec son padding": 3,
          "compte toutes les étapes de stride 2, celles du stem comprises": 13,
          "une étape de stride 2 a été oubliée": 7},
          fractional="le côté d'une carte est un nombre entier : applique la formule à chaque étape"'''),
        ("h", TF, "True", r'''mistakes={"quelle que soit la taille de l'image, que rend le pooling moyen adaptatif ?": False}'''),
    ]),
    Paper("B1.2", "k plus proches voisins cosinus sur six embeddings donnés", [
        ("a", DEC3, "COS_B12[5]", r'''decimals=3, mistakes={"un cosinus ne dépasse pas 1 : divise aussi par la norme de e5": 4 / 3,
          "un cosinus ne dépasse pas 1 : divise aussi par la norme de q": 4 / np.sqrt(2),
          "le produit scalaire seul n'est pas le cosinus : divise par les normes": 4.0}'''),
        ("b", INDEX, "int(np.argmax(COS_B12))", index_mistakes(5, {
            0: "e0 n'a pas la même direction que q : recalcule son cosinus",
            1: "e1 est tout près : calcule les deux meilleurs cosinus à 3 décimales",
            2: "relis la question : le cosinus divise le produit scalaire par les normes",
            3: "c'est le plus proche en distance euclidienne : on demande le cosinus",
            4: "e4 a un petit cosinus avec q : recalcule-le"})),
        ("c", CLASS, '"' + knn_cosine_b12(2) + '"', r'''mistakes={"à égalité de votes, la règle de l'énoncé ne regarde pas la distance des voisins": "oiseau"}'''),
        ("d", CLASS, '"' + knn_cosine_b12(3) + '"', r'''mistakes={"avec k = 3, compte les votes des trois plus proches : y a-t-il encore égalité ?": "avion"}'''),
        ("e", INDEX, "int(np.argmax(E_B12 @ Q_B12))", index_mistakes(2, {
            0: "recalcule le produit scalaire de e0 avec q, et compare-le à tous les autres",
            1: "recalcule les six produits scalaires : lequel est le plus grand ?",
            3: "e3 a un produit scalaire moyen : compare les six",
            4: "e4 a le plus petit produit scalaire, pas le plus grand",
            5: "relis la question : on ne divise pas par les normes"})),
        ("f", DEC3, "float(((Q_B12 / 3 - E_B12[1] / 5) ** 2).sum())", r'''decimals=3, mistakes={"relis la relation de la fiche entre distance au carré et cosinus pour des vecteurs de norme 1 : un facteur manque": 1 - 14 / 15,
          "on demande la distance au carré, pas la distance": np.sqrt(2 - 2 * 14 / 15),
          "ramène les deux vecteurs à la norme 1 avant de soustraire": 6.0}'''),
        ("g", TF, "False", r'''mistakes={"après cette division, compare encore les produits scalaires des autres vecteurs avec q : sont-ils normalisés ?": True}'''),
    ]),
    Paper("B1.3", "Learning rates discriminatifs et warmup : le learning rate de chaque groupe, pas à pas", [
        ("a", DEC1, "lr_b13(3, T_WARMUP_B13)", r'''decimals=1, mistakes={"compte les puissances de γ entre la tête et layer2 : il y en a une de trop": 100 * 0.3 ** 4,
          "compte les puissances de γ entre la tête et layer2 : il en manque une": 100 * 0.3 ** 2,
          "donne la valeur en unités de 10⁻⁵ : 10⁻³ s'écrit 100": 2.7e-5}'''),
        ("b", WHOLE, "round(lr_b13(0, 2))", r'''mistakes={"les pas se comptent à partir de 0 : au pas 0, le facteur du warmup vaut 0": 60,
          "au pas 2, le warmup n'est pas fini": 100}'''),
        ("c", WHOLE, "round(lr_b13(1, 3))", r'''mistakes={"les pas se comptent à partir de 0 : au pas 0, le facteur du warmup vaut 0": 24,
          "au pas 3, le warmup n'est pas fini": 30,
          "relis la question : on demande layer4": 60}'''),
        ("d", DEC2, "lr_b13(0, 3) / lr_b13(3, 3)", r'''decimals=2, mistakes={"compte les puissances de γ entre la tête et layer2 : il y en a une de trop": 0.3 ** -4,
          "compte les puissances de γ entre la tête et layer2 : il en manque une": 0.3 ** -2,
          "le rapport est inversé : on divise le learning rate de la tête par celui de layer2": 0.3 ** 3}'''),
        ("e", DEC2, "lr_b13(2, 55)", r'''decimals=2, mistakes={"au pas 55, la décroissance en cosinus a commencé": 9.0,
          "la progression du cosinus se compte à partir de la fin du warmup": 9 * 0.5 * (1 + math.cos(math.pi * 55 / 105)),
          "relis la question : on demande layer3": 50.0}'''),
        ("f", DEC2, "lr_b13(0, 30)", r'''decimals=2, mistakes={"la progression du cosinus se compte à partir de la fin du warmup": 100 * 0.5 * (1 + math.cos(math.pi * 30 / 105)),
          "relis le signe devant le cosinus dans f(t)": 100 * 0.5 * (1 - math.cos(math.pi / 4)),
          "n'oublie pas le facteur ½ et le 1 de f(t)": 100 * math.cos(math.pi / 4)}'''),
        ("g", TF, "True", r'''mistakes={"relis f(0), puis comment AdamW applique son weight decay découplé (ch. 19)": False}'''),
        ("h", TF, "False", r'''mistakes={"relis la fiche : que fait LambdaLR de la valeur que rend la fonction ?": True}'''),
    ]),
    Paper("B1.4", "Compter les paramètres entraînables : tête seule, dernier bloc, réseau entier", [
        ("a", WHOLE, "HEAD10_B14", r'''mistakes={"n'oublie pas les 10 biais": 5120,
          "la tête d'ImageNet a été remplacée : la nouvelle a 10 sorties": 513000,
          "chaque sortie a un poids par composante de l'embedding": 522}'''),
        ("b", WHOLE, "NET10_B14", r'''mistakes={"retire l'ancienne tête avant d'ajouter la nouvelle": IMAGENET_B14 + HEAD10_B14,
          "ajoute la nouvelle tête au backbone": IMAGENET_B14 - 513000,
          "c'est le réseau avec sa tête ImageNet : remplace-la": IMAGENET_B14}'''),
        ("c", WHOLE, "conv_weights(512, 512, 3)", r'''mistakes={"il y a 512 filtres, un par canal de sortie": 3 * 3 * 512,
          "n'oublie pas la taille du noyau, 3 × 3": 512 * 512,
          "les convolutions de ResNet n'ont pas de biais": conv_weights(512, 512, 3) + 512}'''),
        ("d", WHOLE, "LAYER4_B14", r'''mistakes={"n'oublie pas le raccourci (downsample) du bloc 0": LAYER4_B14 - conv_weights(256, 512, 1) - bn_params(512),
          "n'oublie pas les batchnorms : deux paramètres par canal": LAYER4_B14 - 5 * bn_params(512),
          "les statistiques glissantes sont des buffers, pas des paramètres": LAYER4_B14 + 5 * bn_params(512),
          "la première convolution du bloc 0 reçoit 256 canaux, pas 512": LAYER4_B14 - conv_weights(256, 512, 3) + conv_weights(512, 512, 3),
          "layer4 a deux blocs": BLOCK1_B14}'''),
        ("e", WHOLE, "TRAIN_B14", r'''mistakes={"ajoute la tête, entraînée elle aussi": LAYER4_B14,
          "la tête d'ImageNet a été remplacée par une tête à 10 classes": LAYER4_B14 + 513000}'''),
        ("f", DEC1, "100 * TRAIN_B14 / NET10_B14", r'''decimals=1, mistakes={"divise par le nombre de paramètres du réseau muni de la nouvelle tête": 100 * TRAIN_B14 / IMAGENET_B14,
          "divise par le nombre de paramètres du réseau muni de sa nouvelle tête, pas de l'ancienne": 71.9,
          "on demande un pourcentage, pas une proportion": TRAIN_B14 / NET10_B14,
          "c'est la part des paramètres gelés": 100 - 100 * TRAIN_B14 / NET10_B14}'''),
        ("g", WHOLE, "TRAIN_B14 - 5 * bn_params(512)", r'''mistakes={"gèle aussi les paramètres des batchnorms de layer4": TRAIN_B14,
          "chaque batchnorm a deux paramètres par canal : ne les retire qu'une fois": TRAIN_B14 - 10 * bn_params(512),
          "n'oublie pas la batchnorm du raccourci": TRAIN_B14 - 4 * bn_params(512)}'''),
        ("h", TF, "False", r'''mistakes={"eval() change le calcul des batchnorms : change-t-il leur requires_grad ?": True}'''),
    ]),
    Paper("B1.5", "Zéro-shot CLIP à la main : cosinus, logit_scale et softmax", [
        ("a", DEC3, "P100_B15[1]", r'''decimals=3, mistakes={"multiplie les cosinus par s avant la softmax": zero_shot_b15(1.0)[1],
          "s est l'exponentielle de logit_scale": zero_shot_b15(4.6052)[1],
          "la softmax prend l'exponentielle de chaque logit": COS_B15[1] / COS_B15.sum(),
          "relis la question : s vient ici du paramètre entraîné": PINIT_B15[1]}'''),
        ("b", DEC3, "P100_B15[2]", r'''decimals=3, mistakes={"multiplie les cosinus par s avant la softmax": zero_shot_b15(1.0)[2],
          "s est l'exponentielle de logit_scale": zero_shot_b15(4.6052)[2],
          "la softmax prend l'exponentielle de chaque logit": COS_B15[2] / COS_B15.sum(),
          "relis la question : s vient ici du paramètre entraîné": PINIT_B15[2]}'''),
        ("c", DEC3, "P100_B15[1] / P100_B15[2]", r'''decimals=3, mistakes={"le rapport des probabilités n'est pas celui des cosinus": 0.30 / 0.28,
          "n'oublie pas l'exponentielle": 2.0,
          "le rapport est inversé : p(oiseau) au numérateur": P100_B15[2] / P100_B15[1],
          "multiplie l'écart des cosinus par s": math.exp(0.02)}'''),
        ("d", DEC4, "P100_B15[0]", r'''decimals=4, mistakes={"divise par la somme des exponentielles": math.exp(-6),
          "multiplie les cosinus par s avant la softmax": zero_shot_b15(1.0)[0],
          "relis la question : s vient ici du paramètre entraîné": PINIT_B15[0]}'''),
        ("e", DEC2, "PINIT_B15[1]", r'''decimals=2, mistakes={"relis la question : s vient ici de la valeur initiale": P100_B15[1],
          "s est l'exponentielle de la valeur initiale, pas 1": zero_shot_b15(1.0)[1],
          "s est l'exponentielle de logit_scale_init_value": zero_shot_b15(2.6592)[1]}'''),
        ("f", TF, "True", r'''mistakes={"écris le rapport des deux probabilités : que devient le dénominateur commun ?": False}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ B1.1 à B1.5)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"B\"`) ; "
               "plusieurs lettres : entre guillemets (`\"AC\"`) ; une forme de tenseur : entre guillemets "
               "(`\"(2, 3)\"`) ; un nom de classe : entre guillemets (`\"chat\"`). Arrondis comme l'énoncé le demande, "
               "et seulement à la fin du calcul. Les réponses pas encore remplies affichent ⏳. Les questions « dans "
               "ta copie », la dérivation ∂ B1.6, l'estimation 🧮 B1.7, la réflexion (🗣️ B1.8, ⚖️ B1.9, 📄 B1.10) et "
               "l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises B1.11 to B1.28): second generation session
# ---------------------------------------------------------------------------
PARTS: list = []
NEXT_SESSION = [
    ("A", "B1.11 à B1.16", "charger et sonder : trois modèles pré-entraînés, un ResNet face aux images de 32 pixels, "
                           "embeddings figés, linear probe et k plus proches voisins, résolution, projections, "
                           "sondage bloc par bloc"),
    ("B", "B1.17 à B1.19", "fine-tuner : le dernier bloc avec des learning rates par groupe, un fine-tuning qui "
                           "détruit le modèle, le fine-tuning complet"),
    ("C", "B1.20 à B1.23", "quand le transfert aide ou échoue : courbes d'apprentissage, domaine lointain, oubli "
                           "catastrophique"),
    ("D", "B1.24 à B1.28", "modèles de fondation, coût et livraison : zéro-shot avec CLIP, few-shot par prototypes, "
                           "latence sur CPU, publication du modèle dérivé, et le défi"),
]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | B1.Q1 à B1.Q11, B1.R1 à B1.R3, B1.1 à B1.5 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(ex.table_row(part.key if i == 0 else ""))
    if not PARTS:
        rows += [f"| {key} | {ids} | {title} (*prochaine session de génération*) | | | |"
                 for key, ids, title in NEXT_SESSION]
    if kind == "exercise":
        title = "# B1 · Transfer learning et modèles pré-entraînés — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/b1_transfer/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py B1`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# B1 · Transfer learning et modèles pré-entraînés — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Charger un modèle pré-entraîné avec son prétraitement (torchvision, Hugging Face Hub) et lire sa "
               "fiche modèle et sa licence.\n"
               "- Entraîner un linear probe et un k plus proches voisins sur des embeddings figés, et sonder les "
               "blocs d'un backbone.\n"
               "- Fine-tuner un réseau : dégel progressif, learning rates par groupe, warmup, batchnorm maîtrisée.\n"
               "- Mesurer le gain du pré-entraînement par des courbes d'apprentissage, et diagnostiquer transfert "
               "négatif et oubli catastrophique.\n"
               "- Classer en zéro-shot et en few-shot avec CLIP et DINOv2, et comparer des backbones sur CPU.\n\n"
               "**Rappel express.** Linear probe : $\\hat{\\mathbf{y}} = \\operatorname{softmax}(\\mathbf{W}\\,"
               "\\phi(\\mathbf{x}) + \\mathbf{b})$, $d \\times K + K$ paramètres. Learning rates discriminatifs : "
               "$\\eta_\\ell = \\eta_{\\text{tête}}\\,\\gamma^{L-\\ell}$ ; warmup : $\\eta_t = \\eta_{\\max}\\,"
               "\\min(1, t / t_{\\text{warmup}})$. L2-SP : $\\Omega = \\frac{\\lambda}{2}\\lVert\\theta - \\theta_0"
               "\\rVert^2$. Zéro-shot : $p(c \\mid \\mathbf{x}) \\propto \\exp(s \\cos(\\mathbf{u}_x, "
               "\\mathbf{v}_c))$. Prototype : $\\mathbf{m}_c = \\frac{1}{|S_c|}\\sum_{\\mathbf{x} \\in S_c} "
               "\\phi(\\mathbf{x})$.")]


def footer_cells(kind: str) -> list:
    later = ("" if PARTS else
             "\n\n*Les exercices de code B1.11 à B1.28 (parties A à D : charger et sonder, fine-tuner, quand le "
             "transfert aide ou échoue, modèles de fondation, coût et livraison) seront ajoutés à la prochaine "
             "session de génération ; relance alors `python tools/start_chapter.py B1` pour obtenir le notebook "
             "complet.*")
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu suivre les formes d'un mini-batch dans un ResNet pré-entraîné et compter ce que tu "
               "entraînes selon la stratégie ?\n"
               "2. Sais-tu écrire les groupes de paramètres et le planning d'un fine-tuning, et dire pourquoi les "
               "premières couches bougent le moins ?\n"
               "3. Sais-tu calculer les probabilités d'un zéro-shot de CLIP, et dire ce que fait l'échelle $s$ ?\n"
               "4. Sais-tu dire ce qu'un modèle dérivé hérite de son modèle de base (licence, biais, limites) ?\n\n"
               "**Pour aller plus loin** : le tutoriel de PyTorch sur le transfer learning et l'article de "
               "Yosinski et coll. (2014), cités dans la fiche. La suite : le bonus B2, tokenisation et embeddings."
               + later)]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT + LIBRARY_CHECKS, PAPER_INTRO)
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
