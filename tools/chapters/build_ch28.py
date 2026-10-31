#!/usr/bin/env python
"""Build the two notebooks of chapter 28 from a single source (used by Claude).

    python tools/chapters/build_ch28.py
    python tools/run_all_notebooks.py chapitres/ch28_creatif/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch28_creatif/03_notebook.ipynb

Part 0 checks the short answers of the quizzes Q1 to Q10, of the recalls R1 to R3 and of the ✏️ paper exercises
28.1, 28.2 and 28.4 (02_exercices.md). The derivations ∂ 28.3 and 28.5, the estimates 🧮 28.6 and 28.7, the
reflection (🗣️ 28.8, ⚖️ 28.9, 📄 28.10) and the interview are corrected with 05_solutions.md.
Parts A to D (exercises 28.11 to 28.29, tools/chapters/ch28_parts.py and ch28_code.py; experiments precomputed by
tools/chapters/history_ch28.py): filter and layer visualisation, deep dream, style transfer, and word-level text
generation on Verne, then the challenge.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, bug_letters_mistakes, check_unique_keys, code, guarded, md,  # noqa: E402,F401
                         paper_cells, part_cells, setup_cell, write_notebook)

CHAPTER = "28"
FOLDER = "chapitres/ch28_creatif"

# ---------------------------------------------------------------------------
# Part 0: short answers of the quizzes, the recalls and the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER_CONTEXT = '''# The short answers only use the numbers of their statements (02_exercices.md)
import numpy as np

# VGG16: (convolutions, filters) of the five blocks; a 2x2 max-pooling of stride 2 ends each block
VGG16_BLOCKS = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]


def side(block, image=224):
    """Side of the activation maps of the convolutions of a block (1 to 5)."""
    return image // 2 ** (block - 1)


F_28_2 = np.array([[1, 0, 2, 1], [0, 3, 1, 1], [2, 0, 4, 2]], dtype=float)
RAW_28_2 = F_28_2 @ F_28_2.T
G_28_2 = RAW_28_2 / F_28_2.size

FX_28_4, FC_28_4 = np.array([1, 2, 0, 1.0]), np.array([1, 0, 1, 1.0])
SX_28_4, SS_28_4 = np.array([[1, 1], [0, 2.0]]), np.array([[2, 0], [0, 2.0]])
GX_28_4, GS_28_4 = SX_28_4 @ SX_28_4.T / SX_28_4.size, SS_28_4 @ SS_28_4.T / SS_28_4.size
IMG_28_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 3.0]])
LC_28_4 = float(((FX_28_4 - FC_28_4) ** 2).sum())
LS_28_4 = float(((GX_28_4 - GS_28_4) ** 2).sum())
LTV_28_4 = float(((IMG_28_4[1:] - IMG_28_4[:-1]) ** 2).sum() + ((IMG_28_4[:, 1:] - IMG_28_4[:, :-1]) ** 2).sum())
'''

exec(PAPER_CONTEXT)  # the builder computes a few mistakes with the same helpers

TF = "True or False"
LETTER = 'the letter of your choice, e.g. "B"'
LETTERS = 'the letters, e.g. "ABD"'
WHOLE = "a whole number"
NUMBER = "a number"
DEC3 = "a number, 3 decimals"
DEC4 = "a number, 4 decimals"


def letters(answer: str, hints: dict) -> str:
    """Mistakes for a multiple-choice question: one message for each other letter."""
    return "mistakes={" + ", ".join(f'"{message}": "{letter}"' for letter, message in hints.items()
                                    if letter != answer) + "}"


PAPER = [
    Paper("28.Q1", "Pourquoi partir d'un réseau pré-entraîné comme VGG16", [
        ("a", TF, "False", r'''mistakes={"relis le §28.2.2 : que modifie-t-on à chaque pas, et que garde-t-on fixe ?": True}'''),
        ("b", WHOLE, "13 + 3", r'''mistakes={"les 13 convolutions ne sont pas tout : ajoute les couches denses": 13,
          "le max-pooling n'a pas de poids : ne compte que les convolutions et les couches denses": 21}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "VGG16 est un gros réseau, lent pour sa précision : ce n'est pas la raison",
                                           "C": "ImageNet contient des photos d'objets et d'animaux, pas des tableaux"})),
        ("d", LETTER, '"B"', letters("B", {"A": "c'est le chiffre du livre, corrigé par la fiche (§28.2.1)",
                                           "C": "14 millions est la taille de la base ImageNet complète, pas du jeu d'entraînement d'ILSVRC"})),
        ("e", WHOLE, "1000", r'''mistakes={"ImageNet-1k : le nom donne le nombre de classes": 21000}'''),
    ]),
    Paper("28.Q2", "Montée de gradient sur l'entrée, poids gelés", [
        ("a", LETTER, '"B"', letters("B", {"A": "les poids du réseau restent fixes pendant toute la visualisation",
                                           "C": "le learning rate est un réglage, pas ce qu'on optimise"})),
        ("b", TF, "True", r'''mistakes={"monter sur a, c'est descendre sur −a : le gradient change de signe, le pas aussi": False}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "c'est le deep dream qui part d'une photo, pas la visualisation d'un filtre",
                                           "C": "relis le §28.2.2 : des valeurs tirées au hasard entre −1 et 1"})),
        ("d", LETTER, '"B"', letters("B", {"A": "l'optimiseur ne reçoit que ce qu'on veut modifier : l'image"})),
        ("e", TF, "True", r'''mistakes={"geler les poids coupe leur gradient, pas celui de l'image, qui a requires_grad=True": False}'''),
    ]),
    Paper("28.Q3", "Visualiser une couche entière, pondérer ses filtres", [
        ("a", LETTER, '"B"', letters("B", {"A": "un seul filtre, c'est la visualisation d'un filtre (§28.2.2)",
                                           "C": "les couches denses ne servent pas ici : on regarde les convolutions"})),
        ("b", LETTER, '"A"', letters("A", {"B": "les parties d'objets apparaissent dans les couches profondes, pas au bloc 1"})),
        ("c", TF, "True", r'''mistakes={"avec un poids négatif, augmenter la somme pondérée, c'est diminuer l'activation de ce filtre": False}'''),
        ("d", WHOLE, "512", r'''mistakes={"les filtres doublent de bloc en bloc jusqu'au bloc 4, puis restent à 512": 256}'''),
        ("e", WHOLE, "side(4)", r'''mistakes={"le bloc 4 vient après trois poolings, pas quatre": 14,
          "le bloc 4 vient après trois poolings, et chacun divise le côté par 2": 56}'''),
    ]),
    Paper("28.Q4", "Deep dream : ce qui change par rapport à la visualisation d'une couche", [
        ("a", LETTERS, '"AB"', r'''mistakes={"les poids du réseau ne changent jamais dans ce chapitre": "ABC",
          "le deep dream fait monter les activations, comme la visualisation": "ABD",
          "il manque le changement d'image de départ": "B",
          "il manque la combinaison de plusieurs couches": "A"}'''),
        ("b", TF, "True", r'''mistakes={"la montée de gradient renforce ce qui excite déjà les filtres choisis": False}'''),
        ("c", LETTER, '"A"', letters("A", {"B": "relis le §28.3 : le nom du billet vient du réseau de Google qu'il utilisait",
                                           "C": "relis le §28.3 : le billet porte le nom d'un réseau de Google, pas d'une technique de peinture"})),
        ("d", LETTER, '"C"', letters("C", {"A": "couper le réseau change les couches, pas la zone de l'image",
                                           "B": "un pas plus petit ralentit le rêve partout, pas seulement dans une zone"})),
        ("e", TF, "False", r'''mistakes={"le transfert de style cherche à ressembler à des cibles : il fait baisser des écarts": True}'''),
    ]),
    Paper("28.Q5", "Des yeux partout : l'écho des données d'entraînement", [
        ("a", LETTER, '"C"', letters("C", {"A": "aucune couche n'est spécialisée à l'avance : les détecteurs sont appris",
                                           "B": "le deep dream part d'une photo, pas d'un bruit"})),
        ("b", TF, "False", r'''mistakes={"plus d'une centaine des 1 000 classes sont des races de chiens": True}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "un réseau ne rêve que de ce qu'il a appris"})),
        ("d", LETTER, '"B"', letters("B", {"A": "les traits et les textures, ce sont les couches précoces"})),
        ("e", TF, "True", r'''mistakes={"ce que le réseau fait apparaître dépend de ses données d'entraînement": False}'''),
    ]),
    Paper("28.Q6", "Image de base, référence de style, image générée : que minimise-t-on ?", [
        ("a", LETTER, '"C"', letters("C", {"A": "la photo de contenu est une cible, elle ne change pas",
                                           "B": "le tableau est une cible, il ne change pas",
                                           "D": "le réseau est gelé pendant tout le transfert"})),
        ("b", LETTER, '"A"', letters("A", {"B": "le style se compare par les matrices de Gram, le contenu par les activations"})),
        ("c", LETTER, '"B"', letters("B", {"A": "la photo fournit le contenu, pas le style"})),
        ("d", TF, "False", r'''mistakes={"le réseau sert d'instrument de mesure : ses poids ne bougent pas": True}'''),
        ("e", WHOLE, "1", r'''mistakes={"les cibles sont déjà calculées : seule l'image générée change d'un pas à l'autre": 3,
          "les cibles sont calculées une fois : seule l'image générée repasse": 2}'''),
    ]),
    Paper("28.Q7", "Ce que la matrice de Gram retient et ce qu'elle oublie", [
        ("a", LETTER, '"A"', letters("A", {"B": "on compare des filtres deux à deux, pas des positions",
                                           "C": "c'est la forme de F, la matrice des cartes mises à plat"})),
        ("b", LETTER, '"B"', letters("B", {"A": "les indices de G sont des filtres, pas des pixels",
                                           "C": "la somme porte sur toutes les positions : la position est oubliée"})),
        ("c", TF, "True", r'''mistakes={"chaque coefficient est une somme sur les positions : l'ordre des termes ne compte pas": False}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "G_ii additionne des carrés d'activations, pas des présences",
                                           "C": "G_ii additionne sur toutes les positions, pas seulement la plus forte"})),
        ("e", TF, "False", r'''mistakes={"relis l'encadré ⚠️ « personne ne sait vraiment » du §28.4.6 : Li et coll. nomment cette mesure": True}'''),
    ]),
    Paper("28.Q8", "Loss de contenu : couches précoces ou profondes ?", [
        ("a", LETTER, '"A"', letters("A", {"B": "les couches précoces gardent les détails : la reconstruction est presque exacte"})),
        ("b", LETTER, '"B"', letters("B", {"A": "plus on va profond, plus la reconstruction s'éloigne des pixels"})),
        ("c", LETTER, '"C"', letters("C", {"A": "c'est le choix du code de Chollet, pas celui du livre",
                                           "B": "toutes les couches, c'est le choix du livre pour le style"})),
        ("d", TF, "False", r'''mistakes={"Gatys et coll. prennent conv4_2, une couche profonde (fiche §28.4.3)": True}'''),
        ("e", LETTER, '"B"', letters("B", {"A": "le nombre de paramètres n'a rien à voir avec ce que la loss impose",
                                           "C": "le temps de calcul n'a rien à voir avec ce que la loss impose"})),
    ]),
    Paper("28.Q9", "Loss de style, poids relatifs et variation totale", [
        ("a", LETTER, '"B"', letters("B", {"A": "relis le §28.4.5 : le livre prend le style sur toutes les convolutions"})),
        ("b", WHOLE, "round(1 / 0.025)", r'''mistakes={"refais la division : 1 divisé par 0,025": 25}'''),
        ("c", LETTER, '"C"', letters("C", {"A": "la variation totale compare chaque pixel à ses voisins, pas à une plage de couleurs",
                                           "B": "l'écart à la photo, c'est la loss de contenu"})),
        ("d", TF, "True", r'''mistakes={"multiplie la loss de style par 1 000 en changeant de normalisation : le même β donne-t-il le même dosage ?": False}'''),
        ("e", TF, "False", r'''mistakes={"partir de la photo, c'est partir d'une image dont le contenu est déjà juste : c'est elle qui converge vite": True}'''),
    ]),
    Paper("28.Q10", "Générer du texte mot à mot : ce qu'implique « le mot le plus probable »", [
        ("a", LETTER, '"A"', letters("A", {"B": "relis le §28.5 : « le mot suivant le plus probable »"})),
        ("b", TF, "False", r'''mistakes={"le glouton ne tire rien au hasard : même entrée, même sortie": True}'''),
        ("c", LETTER, '"B"', letters("B", {"A": "un modèle mot à mot choisit dans son vocabulaire : il ne peut pas inventer de mot"})),
        ("d", TF, "False", r'''mistakes={"la couche de sortie a un neurone par mot du vocabulaire : rien d'autre ne peut sortir": True}'''),
        ("e", LETTER, '"A"', letters("A", {"B": "diviser par T < 1, c'est multiplier les écarts entre logits"})),
    ]),
    Paper("28.R1", "Un réseau qui sert de loss (le discriminateur)", [
        ("a", LETTER, '"A"', letters("A", {"B": "le générateur ne voit jamais de vraie image (ch. 27)"})),
        ("b", LETTER, '"B"', letters("B", {"A": "VGG16 est gelé : il n'apprend rien pendant le transfert"})),
        ("c", TF, "True", r'''mistakes={"dans les deux cas, le gradient de la loss traverse un réseau avant d'arriver à ce qu'on modifie": False}'''),
        ("d", LETTER, '"B"', letters("B", {"A": "chez Gatys, il n'y a pas de générateur : on modifie l'image elle-même"})),
        ("e", LETTER, '"B"', letters("B", {"A": "une loss adversariale vient d'un discriminateur entraîné en même temps"})),
    ]),
    Paper("28.R2", "Mélange de contenu contre mélange de représentations", [
        ("a", LETTER, '"B"', letters("B", {"A": "une moyenne de pixels superpose les deux traits, chacun à moitié d'intensité"})),
        ("b", LETTER, '"A"', letters("A", {"B": "le décodeur transforme un code en une image qui ressemble aux données"})),
        ("c", TF, "True", r'''mistakes={"le transfert de style ne moyenne pas de pixels : il impose des activations et des matrices de Gram": False}'''),
        ("d", NUMBER, "round(0.75 * 0 + 0.25 * 4)", r'''mistakes={"t pondère z₂, et 1 − t pondère z₁": 3}'''),
        ("e", TF, "False", r'''mistakes={"la matrice de Gram est quadratique en F : développe (F₁ + F₂)(F₁ + F₂)ᵀ / 4": True}'''),
    ]),
    Paper("28.R3", "Champ récepteur et hiérarchie des filtres de VGG16", [
        ("a", WHOLE, "3 + 2", r'''mistakes={"chaque convolution 3 × 3 de plus ajoute 2 pixels au champ récepteur, pas 3": 6,
          "deux convolutions ne voient pas seulement 3 pixels : la seconde voit 3 sorties de la première": 3}'''),
        ("b", WHOLE, "3 + 2 + 2", r'''mistakes={"chaque convolution 3 × 3 de plus ajoute 2 pixels, pas 3": 9}'''),
        ("c", TF, "True", r'''mistakes={"relis le ch. 21 : chaque couche voit une plus grande région de l'image": False}'''),
        ("d", WHOLE, "32 // 2 ** 5", r'''mistakes={"il y a cinq poolings, pas quatre": 2,
          "seuls les poolings réduisent la taille, et il y en a cinq": 0}'''),
        ("e", WHOLE, "64 * 64 * 9 + 64", r'''mistakes={"n'oublie pas les 64 biais": 36864,
          "un filtre 3 × 3 voit tous les canaux d'entrée : 64 × 3 × 3 poids par filtre": 640}'''),
    ]),
    Paper("28.1", "Tailles des activations et des matrices de Gram le long de VGG16", [
        ("a", WHOLE, "64 * side(1) ** 2", r'''mistakes={"n'oublie pas les 64 canaux": 50176,
          "la convolution garde la taille : 224 × 224, pas 112 × 112": 64 * 112 ** 2}'''),
        ("b", WHOLE, "512 * side(4) ** 2", r'''mistakes={"le bloc 4 vient après trois poolings : ses cartes font 28 × 28": 512 * 14 ** 2,
          "la première convolution du bloc 4 a déjà 512 filtres": 256 * 28 ** 2}'''),
        ("c", WHOLE, "512 ** 2", r'''mistakes={"la matrice de Gram est C × C : elle ne dépend pas de la taille des cartes": 401408,
          "c'est la moitié de la matrice : compte tous ses coefficients": 512 * 513 // 2}'''),
        ("d", WHOLE, "64 * side(1) ** 2 // 64 ** 2", r'''mistakes={"divise le nombre de valeurs de l'activation par 64 × 64, la taille de la matrice": 64 * 224 ** 2 // 64}'''),
        ("e", WHOLE, "512 * 513 // 2", r'''mistakes={"la diagonale compte aussi : n(n + 1)/2, pas n(n − 1)/2": 512 * 511 // 2,
          "la matrice est symétrique : les coefficients de part et d'autre de la diagonale sont égaux": 512 ** 2}'''),
        ("f", WHOLE, "side(5, image=128)", r'''mistakes={"le bloc 5 vient après quatre poolings, pas cinq": 4,
          "le bloc 5 vient après quatre poolings, et chacun divise le côté par 2": 16}'''),
        ("g", TF, "False", r'''mistakes={"la matrice de Gram est C × C : seul le nombre de filtres compte": True}'''),
    ]),
    Paper("28.2", "Une matrice de Gram à la main", [
        ("a", WHOLE, "int(RAW_28_2[0, 1])", r'''mistakes={"le coefficient (1, 2) combine deux lignes différentes de F, pas une ligne avec elle-même": 6}'''),
        ("b", WHOLE, "int(RAW_28_2[1, 1])", r'''mistakes={"le coefficient (2, 2) est la somme des carrés de la ligne 2, pas sa somme": 5}'''),
        ("c", NUMBER, "G_28_2[0, 2]", r'''decimals=3, mistakes={"normalise par C·H·W = 3 × 2 × 2 = 12": 12.0,
          "normalise par C·H·W = 12, pas seulement par H·W = 4": 3.0}'''),
        ("d", DEC3, "G_28_2[1, 1]", r'''decimals=3, mistakes={"normalise par C·H·W = 12": 11.0,
          "normalise par C·H·W = 12, pas seulement par H·W = 4": 2.75}'''),
        ("e", DEC3, "G_28_2.sum()", r'''decimals=3, mistakes={"compte chaque coefficient hors diagonale deux fois : la matrice en a neuf": 0.5 + 11 / 12 + 2 + 0.25 + 1 + 0.5,
          "normalise par C·H·W = 12": 83.0}'''),
        ("f", DEC3, "G_28_2[0, 1] / np.sqrt(G_28_2[0, 0] * G_28_2[1, 1])", r'''decimals=3, mistakes={"divise par la racine du produit des deux diagonales, pas par leur produit": 3 / 66,
          "divise par la racine des deux coefficients diagonaux, pas par leur somme": 3 / 17}'''),
        ("g", TF, "True", r'''mistakes={"retourner toutes les cartes de la même façon ne fait que réordonner les positions": False}'''),
    ]),
    Paper("28.4", "Losses de contenu, de style et de variation totale sur un mini-exemple", [
        ("a", NUMBER, "LC_28_4", r'''decimals=3, mistakes={"additionne les carrés des écarts, pas les écarts": 1.0,
          "additionne les carrés des écarts, pas leurs valeurs absolues": 3.0}'''),
        ("b", NUMBER, "LC_28_4 / 4", r'''decimals=3, mistakes={"F.mse_loss prend la moyenne des carrés : divise par 4": 5.0}'''),
        ("c", NUMBER, "GX_28_4[0, 1]", r'''decimals=3, mistakes={"normalise par C·H·W = 2 × 2 = 4": 2.0,
          "G multiplie les lignes de F (une par filtre) : F Fᵀ, pas Fᵀ F": 0.25}'''),
        ("d", NUMBER, "LS_28_4", r'''decimals=3, mistakes={"additionne les carrés des quatre écarts, sans racine": np.sqrt(0.75),
          "n'oublie pas les deux coefficients hors diagonale, qui comptent chacun": 0.5}'''),
        ("e", NUMBER, "LTV_28_4", r'''decimals=3, mistakes={"additionne les deux directions, horizontale et verticale": 7.0,
          "élève chaque écart au carré : la valeur absolue ne suffit pas": 10.0,
          "des écarts signés se compensent : élève-les au carré": 6.0}'''),
        ("f", DEC4, "0.025 * LC_28_4 + LS_28_4 + 1e-4 * LTV_28_4", r'''decimals=4, mistakes={"le contenu est pondéré par α = 0,025": 5 + 0.75 + 14e-4,
          "n'oublie pas la variation totale, même petite": 0.875}'''),
        ("g", NUMBER, "float(((SX_28_4 @ SX_28_4.T - SS_28_4 @ SS_28_4.T) ** 2).sum())", r'''decimals=3, mistakes={"sans normalisation, les écarts sont 4 fois plus grands, leurs carrés 16 fois": 3.0}'''),
    ]),
]
for _p in PAPER:
    for _sub in _p.subs:
        check_unique_keys(f"f({_sub[3]})", f"{_p.id}{_sub[0]}")

PAPER_INTRO = ("## Partie 0 · Vérifier tes réponses courtes (quiz, rappels, ✏️ 28.1, 28.2 et 28.4)\n\n"
               "Fais d'abord les quiz, les rappels et les exercices de `02_exercices.md` **sur papier**, dans ta "
               "copie de `06_mes_reponses.md`. Reporte ensuite chaque réponse courte ici : **la valeur** que tu as "
               "trouvée, pas l'expression Python, sinon tu ne vérifies rien. Un nombre : `42` ou `0.3` (en Python, "
               "le séparateur décimal est un **point** ; `0,3` sans guillemets serait un couple de deux nombres) ; "
               "un vrai ou faux : `True` ou `False` ; un choix : la lettre seule, entre guillemets (`\"B\"`) ; "
               "plusieurs lettres : entre guillemets (`\"ABD\"`). Arrondis comme l'énoncé le demande, et seulement "
               "à la fin du calcul. Les réponses pas encore remplies affichent ⏳. Les questions « dans ta copie », "
               "les dérivations ∂ 28.3 et 28.5, les estimations 🧮 28.6 et 28.7, la réflexion (🗣️ 28.8, ⚖️ 28.9, "
               "📄 28.10) et l'entretien se corrigent avec `05_solutions.md`.")

# ---------------------------------------------------------------------------
# Parts A to D (exercises 28.11 to 28.29): second generation session
# ---------------------------------------------------------------------------
from ch28_parts import PARTS, ROCKETS  # noqa: E402
import ch28_code  # noqa: E402


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 28.Q1 à 28.Q10, 28.R1 à 28.R3, 28.1, 28.2, 28.4 | vérifier tes réponses courtes | 🧠 🔁 ✏️ | ★ à ★★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            title = ex.title.replace("|", "\\|")
            rocket = " 🚀" if ex.id in ROCKETS else ""
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {title}{rocket} | {ex.type} | {STARS[ex.stars]} | "
                        f"{ex.minutes} |")
    if kind == "exercise":
        title = "# 28 · Applications créatives — notebook d'exercices"
        how = ("La partie 0 vérifie tes réponses courtes aux quiz, aux rappels et aux exercices papier. Chaque "
               "exercice de code : un énoncé, une cellule à compléter (les `...` et les `raise "
               "NotImplementedError`), puis une cellule de vérification (`wb.check`, ou des tests). « Tout "
               "exécuter » (*Run all*) va jusqu'au bout même si rien n'est rempli : ce qui n'est pas fait affiche "
               "⏳. Les questions « dans tes notes » qui n'ont pas de cellule 📝 se notent dans la section « Notes "
               "sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`. Tout tourne sur "
               "CPU : le notebook utilise SqueezeNet 1.1 (5 Mo de poids) sur de petites images. 🚀 : en mode complet "
               "(`FAST_MODE = False`), des images et des entraînements plus grands, et VGG16 (528 Mio) si tu le "
               "souhaites.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch28_creatif/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 28`) : ce fichier-ci est mis à jour par Claude. Sur Colab, le badge "
               "ouvre cette version du dépôt, qui n'est pas enregistrée : crée puis ouvre ta copie comme l'explique "
               "`00_setup/COLAB.md` §2.")
    else:
        title = "# 28 · Applications créatives — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Extraire les activations d'un réseau pré-entraîné gelé avec des hooks, et visualiser ce que "
               "cherchent ses filtres et ses couches par montée de gradient sur l'image.\n"
               "- Faire rêver un réseau sur une photo libre (deep dream).\n"
               "- Écrire la matrice de Gram et les losses de contenu, de style et de variation totale, puis un "
               "transfert de style complet sous budget CPU, et documenter une galerie (sources, licences, réglages).\n"
               "- Générer du texte mot à mot sur Verne et comparer les stratégies de décodage.\n\n"
               "**Rappel express.** Montée de gradient : $\\mathbf{x} \\leftarrow \\mathbf{x} + \\eta\\, "
               "\\nabla_{\\mathbf{x}}\\, a(\\mathbf{x})$, poids gelés. Gram : $G = \\frac{1}{CHW} F F^\\top$. "
               "Transfert de style : $L = \\alpha \\lVert F(\\mathbf{x}) - F(\\mathbf{c}) \\rVert^2 + \\beta \\sum_\\ell "
               "w_\\ell \\lVert G^\\ell(\\mathbf{x}) - G^\\ell(\\mathbf{s}) \\rVert_F^2 + \\lambda L_{TV}$. "
               "Température : $p_i \\propto e^{z_i / T}$.")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu écrire une montée de gradient sur l'image d'un réseau gelé, normalisation et plage des "
               "pixels comprises ?\n"
               "2. Sais-tu calculer une matrice de Gram et dire ce qu'elle garde et ce qu'elle oublie ?\n"
               "3. Sais-tu régler les trois poids d'un transfert de style en lisant la valeur de chaque terme, et "
               "choisir une stratégie de décodage de texte ?\n\n"
               "**Pour aller plus loin** : l'article de Gatys et coll. (2015) et l'article « Feature "
               "Visualization » de Distill (2017), cités dans la fiche. La suite : le ch. 29, les datasets et la "
               "préparation du projet final.")]


def build(kind: str) -> list:
    cells = header_cells(kind)
    cells.append(setup_cell(kind, chapter=CHAPTER) if kind == "exercise" else setup_cell(kind))
    cells += objectives_cell()
    cells += paper_cells(kind, PAPER, PAPER_CONTEXT, PAPER_INTRO)
    cells += [md("## Outils du notebook\n\nLa cellule suivante importe PyTorch, charge le réseau pré-entraîné gelé `NET` "
                 "(SqueezeNet 1.1) et définit les petits outils des quatre parties : `normalize`, `to_batch`, `to_image`, "
                 "`show_images`, `high_freq_share`, `dominant_period`… Lis leurs docstrings au besoin. Le premier "
                 "lancement télécharge les poids (5 Mo)."), code(ch28_code.TOOLS)]
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
