#!/usr/bin/env python
"""Build the two notebooks of chapter 1 from a single source (used by Claude).

    python tools/chapters/build_ch01.py
    python tools/run_all_notebooks.py chapitres/ch01_introduction/05_solutions.ipynb --inplace
    python tools/build_answers.py
    python tools/run_all_notebooks.py chapitres/ch01_introduction/03_notebook.ipynb

Part 0 checks the ✏️ paper exercises (1.1 to 1.4); parts A to D are the notebook
exercises 1.9 to 1.25 (no mylearn module in this chapter).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chapter_kit import (STARS, Ex, Paper, Part, badge, md, paper_cells, part_cells,  # noqa: E402
                         setup_cell, write_notebook)

CHAPTER = "1"
FOLDER = "chapitres/ch01_introduction"

# ---------------------------------------------------------------------------
# Part 0: checks of the paper exercises (02_exercices.md)
# ---------------------------------------------------------------------------
PAPER = [
    Paper("1.1", "Accuracy et erreurs à l'échelle d'un centre de tri", [
        ("a", "the accuracy of the book's network, a proportion (4 decimals)", "9905 / 10000", "decimals=4"),
        ("b", "its error rate, in % (2 decimals)", "100 * (1 - 9905 / 10000)",
         'decimals=2, mistakes={"c\'est l\'accuracy : on demande le taux d\'ERREUR": 0.99, '
         '"c\'est l\'accuracy en pourcentage : on demande le taux d\'ERREUR, en %": 99.05, '
         '"95, c\'est le NOMBRE d\'erreurs : exprime-le en pourcentage des 10 000 chiffres": 95}'),
        ("c", "the number of digits misread per day", "round(1_200_000 * (1 - 9905 / 10000))",
         'mistakes={"on demande les chiffres MAL lus : multiplie par le taux d\'erreur, pas par l\'accuracy": 1_188_600}'),
        ("d", "P(a 5-digit code is read without any error), 4 decimals", "(9905 / 10000) ** 5",
         'decimals=4, mistakes={"c\'est l\'approximation 1 − 5 × 0,0095 : calcule la probabilité EXACTE que les 5 chiffres soient tous justes (0B, indépendance)": 0.9525, '
         '"un code compte 5 chiffres, et chacun doit être lu juste": 0.9905}'),
        ("e", "the number of wrong codes among 240 000 envelopes", "round(240_000 * (1 - (9905 / 10000) ** 5))",
         'mistakes={"tu as arrondi trop tôt : garde la probabilité de d) sans l\'arrondir avant de multiplier": 11_280, '
         '"tu as arrondi d) à 4 décimales avant de multiplier : garde toutes les décimales de ta calculatrice": 11_184, '
         '"tu comptes les CHIFFRES faux, pas les CODES faux : un code avec deux chiffres faux ne compte qu\'une fois": 11_400}'),
    ]),
    Paper("1.2", "Concerts : la valeur manquante et celle de demain", [
        ("a", "the attendance of May 8, by linear interpolation between May 7 and May 9", "int((1290 + 1390) / 2)", ""),
        ("b", "the slope of the line through the first and the last day (people per day)",
         "int((1550 - 1200) / (12 - 5))",
         'mistakes={"du 5 au 12 mai, il y a 7 jours d\'écart (7 intervalles), pas 8": 44}'),
        ("c", "the value of this line on May 8", "int(1200 + 50 * (8 - 5))",
         'mistakes={"c\'est l\'interpolation de a) : on demande la valeur donnée par la DROITE de b)": 1340}'),
        ("d", "the prediction of this line for May 13", "int(1550 + 50 * (13 - 12))",
         'mistakes={"c\'est la prévision faite avec les deux derniers jours (question e) : utilise la droite de b)": 1580}'),
        ("e", "the prediction for May 13 from the last two days only", "int(1550 + (1550 - 1520))",
         'mistakes={"c\'est la prévision de la droite de b) : ici, prolonge seulement la tendance des deux derniers jours": 1600}'),
        ("f", "the band's income on May 13, in euros (with the prediction of d)", "round(1600 * 25 * 0.10)",
         'mistakes={"le groupe touche 10 % des ventes, pas la totalité": 40_000, "10 % s\'écrit 0,10 (ou 10/100), pas 10": 400_000}'),
    ]),
    Paper("1.3", "Compter les connexions d'un réseau en couches", [
        ("a", "the number of connections of the 4 -> 3 -> 2 network", "4 * 3 + 3 * 2",
         'mistakes={"compte les connexions de CHAQUE paire de couches voisines (4 → 3, puis 3 → 2) ; pour une paire, multiplie les tailles": 12, '
         '"on ne compte pas les neurones, mais les connexions entre deux couches voisines": 9}'),
        ("b", "its number of parameters (weights + biases)", "4 * 3 + 3 * 2 + 3 + 2",
         'mistakes={"un biais par NEURONE ; les 4 entrées ne sont pas des neurones": 4 * 3 + 3 * 2 + 4 + 3 + 2, '
         '"n\'oublie pas les biais : un par neurone": 18}'),
        ("c", "the number of parameters of the 784 -> 128 -> 10 network", "784 * 128 + 128 + 128 * 10 + 10",
         'mistakes={"n\'oublie pas un biais par neurone (128 + 10)": 784 * 128 + 128 * 10}'),
        ("d", "the same with 256 hidden neurons", "784 * 256 + 256 + 256 * 10 + 10",
         'mistakes={"n\'oublie pas un biais par neurone (256 + 10)": 784 * 256 + 256 * 10}'),
        ("e", "the number of parameters of the 784 -> 128 -> 128 -> 10 network",
         "784 * 128 + 128 + 128 * 128 + 128 + 128 * 10 + 10",
         'mistakes={"n\'oublie pas un biais par neurone (128 + 128 + 10)": 784 * 128 + 128 * 128 + 128 * 10}'),
    ]),
    Paper("1.4", "Moins de nombres pour dire la même chose", [
        ("a", "the number of features that carry information", "2",
         'mistakes={"une feature qui vaut toujours 0 ne distingue aucun jour d\'un autre": 3}'),
        ("b", "68 kg in pounds (1 decimal)", "68 * 2.2046", "decimals=1"),
        ("c", "the number of independent pieces of information in (kg, lb)", "1",
         'mistakes={"connaître l\'une des deux valeurs suffit pour calculer l\'autre": 2}'),
        ("d", "the distance of P from the start of the road, in m", "int(np.hypot(240, 320))",
         'mistakes={"la distance à vol d\'oiseau se calcule avec Pythagore, pas en additionnant les coordonnées": 560, '
         '"c\'est le CARRÉ de la distance : prends la racine": 160_000}'),
        ("e", "the map coordinates of the car at 750 m, a list", "[round(750 * 0.6), round(750 * 0.8)]", ""),
        ("f", "the position of Q along the road, in m", "round(250 * 0.6 + 300 * 0.8)",
         'mistakes={"c\'est la distance de Q au départ à vol d\'oiseau : on veut sa position LE LONG de la route (projection)": 391}'),
        ("g", "the distance lost (from Q to the centre line), in m", "round(abs(-250 * 0.8 + 300 * 0.6))",
         'mistakes={"c\'est le CARRÉ de la distance : prends la racine": 400}'),
    ]),
]

PAPER_CONTEXT = '''# The paper exercises only use the numbers of their statements (02_exercices.md)
import numpy as np'''

PAPER_INTRO = ("## Partie 0 · Vérifier tes exercices papier (✏️ de 1.1 à 1.4)\n\n"
               "Fais d'abord les exercices ✏️ de `02_exercices.md` **sur papier**, dans ta copie de "
               "`06_mes_reponses.md`. Reporte ensuite chaque réponse ici : **la valeur** que tu as trouvée "
               "(par exemple `42`, `0.125` ou `[7, -2]`), pas l'expression Python, sinon tu ne vérifies rien. "
               "Arrondis comme l'énoncé le demande. Les réponses pas encore remplies affichent ⏳.")

# ---------------------------------------------------------------------------
# Part A: the four common threads (1.9 to 1.13)
# ---------------------------------------------------------------------------
PART_A_GIVEN = r'''# Tools for the notebook exercises (parts A to D)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def verdict(ex_id, ok, success, failure):
    """Print ✅ with the success message, or ❌ with the failure message."""
    print(f"✅ Ex {ex_id} : {success}" if ok else f"❌ Ex {ex_id} : {failure}")'''

PART_A = Part("A", "Les quatre fils rouges du workbook",
              "Fiche §1.2.2 et « Les quatre fils rouges ». Tu ouvres les quatre datasets qui reviendront tout au "
              "long du workbook et tu les regardes avec le vocabulaire du chapitre : échantillons, features, labels, "
              "et le type de problème que chacun illustre.",
              given=PART_A_GIVEN, exercises=[
    Ex("1.9", "📦", 1, 10, "Penguins : échantillons, features et labels",
       "décrire un dataset tabulaire avec le vocabulaire du machine learning.",
       "0A (pandas) · fiche §1.2.2 · livre §1.2.2, §1.3.1", thread="Penguins", tracks="R, C",
       body=r"""Le dataset Palmer Penguins (déjà vu en 0A) décrit des manchots de trois espèces. Avec le vocabulaire du livre (§1.2.2) : chaque **ligne** est un **échantillon**, chaque **colonne** une **feature**, et la colonne que l'on cherche à prédire est le **label**. Ici, le label est l'espèce (`species`).

a) `n_samples` : le nombre d'échantillons de `penguins_all`.
b) `n_features` : le nombre de features qui décrivent chaque manchot quand le label est `species` (toutes les **autres** colonnes).
c) `species_counts` : le nombre de manchots de chaque espèce, **du plus fréquent au moins fréquent**, sous forme d'une liste de 3 entiers (`value_counts()`, puis `.tolist()`).
d) `n_complete` : le nombre de manchots qui n'ont **aucune** valeur manquante (`dropna()`).
e) `first_vector` : les 4 mesures du premier manchot, dans l'ordre de `MEASURES`, sous forme d'une liste (`penguins_all.loc[0, MEASURES]`). Ce manchot est désormais un **vecteur** (0B).
f) `label_for_mass` : si l'on voulait prédire la **masse** d'un manchot à partir de ses autres colonnes, quel serait le nom de la colonne label ?""",
       given=r'''penguins_all = wb.datasets.load_penguins()     # all the penguins, missing values included
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
penguins_all.head()''',
       todo=r'''n_samples = ...        # a)
n_features = ...       # b)
species_counts = ...   # c) a list of 3 integers, most frequent species first
n_complete = ...       # d)
first_vector = ...     # e) a list of 4 numbers
label_for_mass = ...   # f) a column name (a string)''',
       check=r'''wb.check("1.9a", n_samples)
wb.check("1.9b", n_features)
wb.check("1.9c", species_counts)
wb.check("1.9d", n_complete)
wb.check("1.9e", first_vector)
wb.check("1.9f", label_for_mass)''',
       solution=r'''n_samples = len(penguins_all)
n_features = penguins_all.shape[1] - 1            # every column except the label
species_counts = penguins_all["species"].value_counts().tolist()
n_complete = len(penguins_all.dropna())
first_vector = penguins_all.loc[0, MEASURES].astype(float).tolist()
label_for_mass = "body_mass_g"
print(n_samples, n_features, species_counts, n_complete, first_vector, label_for_mass)''',
       record=r'''wb.record("1.9a", n_samples)
wb.record("1.9b", n_features, mistakes={"le label n'est pas une feature : compte toutes les AUTRES colonnes": 8})
wb.record("1.9c", species_counts, mistakes={"compte tous les manchots de penguins_all, pas seulement les complets": [146, 119, 68]})
wb.record("1.9d", n_complete, mistakes={"dropna() retire une ligne dès qu'UNE valeur manque, sexe compris": 342})
wb.record("1.9e", first_vector, decimals=1)
wb.record("1.9f", label_for_mass, mistakes={"le label est ce que l'on veut PRÉDIRE : ici, la masse": "species"})''',
       note="Le mot « feature » dépend de la question posée : pour prédire l'espèce, la masse est une feature ; "
            "pour prédire la masse, c'est l'espèce qui devient une feature. 9 des 11 manchots au sexe inconnu "
            "ont toutes leurs mesures, mais `dropna()` les retire quand même ; pour ne regarder que certaines "
            "colonnes, `dropna(subset=MEASURES)` (0A)."),

    Ex("1.10", "📦", 1, 10, "MNIST : une image, 784 nombres",
       "voir qu'une image n'est, pour un ordinateur, qu'un tableau de nombres.",
       "0A (NumPy, `wb.plot.show_images`) · fiche « Les quatre fils rouges » · livre §1.1.1, §1.7", thread="MNIST",
       tracks="R, C",
       body=r"""MNIST contient 70 000 chiffres écrits à la main (60 000 pour l'entraînement, 10 000 pour le test), en niveaux de gris : chaque pixel est un entier, d'autant plus grand que l'encre est foncée. Le livre y revient souvent (§1.7), le workbook aussi.

a) `shape_mnist` : la forme de `X_mnist` (un tuple).
b) `n_pixels` : le nombre de pixels d'**une** image.
c) `pixel_range` : la liste `[plus petite valeur, plus grande valeur]` des pixels de tout `X_mnist`.
d) `first_label` : le label de la première image.
e) `n_ink` : le nombre de pixels **non nuls** de la première image (un masque booléen, 0A).
f) `most_common_digit` : le chiffre le plus fréquent parmi les labels `y_mnist` (`np.bincount`, puis `argmax`).

La vérification affiche ensuite 16 autres images avec leur label, et un morceau de la première image **en nombres**.""",
       given=r'''X_mnist, y_mnist = wb.datasets.load_mnist("train")   # 60 000 training images and their labels
print(type(X_mnist), X_mnist.dtype)''',
       todo=r'''shape_mnist = ...         # a)
n_pixels = ...            # b)
pixel_range = ...         # c) [min, max]
first_label = ...         # d)
n_ink = ...               # e)
most_common_digit = ...   # f)''',
       check=r'''wb.check("1.10a", shape_mnist)
wb.check("1.10b", n_pixels)
wb.check("1.10c", pixel_range)
wb.check("1.10d", first_label)
wb.check("1.10e", n_ink)
wb.check("1.10f", most_common_digit)
wb.plot.show_images(X_mnist[1:17], y_mnist[1:17], ncols=8)   # images 1 to 16 (the first one is yours to read)
plt.show()
print(X_mnist[0][4:24, 4:24])   # the centre of the first image, as numbers''',
       solution=r'''shape_mnist = X_mnist.shape
n_pixels = 28 * 28
pixel_range = [int(X_mnist.min()), int(X_mnist.max())]
first_label = int(y_mnist[0])
n_ink = int((X_mnist[0] > 0).sum())
most_common_digit = int(np.bincount(y_mnist).argmax())
print(shape_mnist, n_pixels, pixel_range, first_label, n_ink, most_common_digit, np.bincount(y_mnist))
wb.plot.show_images(X_mnist[1:17], y_mnist[1:17], ncols=8)
plt.show()
print(X_mnist[0][4:24, 4:24])''',
       record=r'''wb.record("1.10a", shape_mnist, mistakes={"une image est un tableau 28 × 28 : la forme compte trois nombres (images, lignes, colonnes)": [60000, 784]})
wb.record("1.10b", n_pixels)
wb.record("1.10c", pixel_range)
wb.record("1.10d", first_label)
wb.record("1.10e", n_ink, mistakes={"compte les pixels NON NULS (l'encre), pas les pixels de fond": 784 - n_ink})
wb.record("1.10f", most_common_digit)''',
       note="Pour un ordinateur, une image n'est qu'une grille de nombres. « Aplatie » (`reshape(784)`), elle "
            "devient un vecteur de 784 features : c'est ainsi que le réseau de 1.23 la verra. Les classes sont "
            "presque équilibrées : de 5 421 (chiffre 5) à 6 742 (chiffre 1) images."),

    Ex("1.11", "📦", 2, 15, "Holmes et Verne : le texte devient des nombres",
       "transformer un texte en nombres (codes des caractères, fréquences des lettres) et comparer deux textes avec une distance.",
       "0A (chaînes, `Counter`) · 0B (distance entre vecteurs) · fiche « Les quatre fils rouges » · livre §1.1.1",
       thread="Holmes/Verne", tracks="R, M, C",
       body=r"""Deux romans du domaine public servent de fils rouges « texte » : *The Adventures of Sherlock Holmes* (anglais) et *Le Tour du monde en quatre-vingts jours* (français). Pour un ordinateur, un texte est une suite de caractères, et chaque caractère a un numéro (son code Unicode, `ord`).

a) `n_chars` : la liste `[nombre de caractères de holmes, nombre de caractères de verne]`.
b) `codes` : la liste des codes (`ord`) des 4 caractères de la chaîne `"Aa é"` (l'espace compte).
c) `n_distinct` : la liste `[nombre de caractères différents dans holmes, … dans verne]` (`set`). Pourquoi le français en a-t-il davantage ?
d) et e) `letter_freq(text)` : le vecteur des **26 fréquences** des lettres de `a` à `z`. On passe le texte en minuscules (`text.lower()`), on ne garde que les caractères entre `"a"` et `"z"` (les lettres accentuées ne comptent pas), puis la fréquence d'une lettre est son nombre d'apparitions divisé par le nombre total de lettres gardées. La vérification regarde d) `letter_freq(holmes)` et e) `letter_freq(verne)` (3 décimales).
f) `distances` : la liste `[d_halves, d_languages]` (4 décimales), où `d_halves` est la distance (norme de la différence, 0B) entre les vecteurs de fréquences des deux moitiés de `holmes` (`holmes[:len(holmes) // 2]` et `holmes[len(holmes) // 2:]`), et `d_languages` la distance entre les vecteurs de `holmes` et de `verne`. Qu'en conclus-tu ?""",
       given=r'''holmes = wb.datasets.load_holmes()     # a str: the whole book, in English
verne = wb.datasets.load_verne()       # a str: the whole book, in French
print(holmes[:300])
print("-" * 60)
print(verne[:300])''',
       todo=r'''n_chars = ...      # a)
codes = ...        # b)
n_distinct = ...   # c)


def letter_freq(text):
    """The 26 frequencies of the letters a to z among the letters a-z of text (in lower case), a NumPy array."""
    raise NotImplementedError("letter_freq() is not written yet")


distances = ...    # f) [d_halves, d_languages]''',
       check=r'''wb.check("1.11a", n_chars)
wb.check("1.11b", codes)
wb.check("1.11c", n_distinct)
with wb.attempt("1.11d"):
    wb.check("1.11d", letter_freq(holmes))
    wb.check("1.11e", letter_freq(verne))
    letters = [chr(k) for k in range(ord("a"), ord("z") + 1)]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(np.arange(26) - 0.2, letter_freq(holmes), width=0.4, label="Holmes (EN)")
    ax.bar(np.arange(26) + 0.2, letter_freq(verne), width=0.4, label="Verne (FR)")
    ax.set_xticks(np.arange(26), letters)
    ax.set_ylabel("frequency")
    ax.legend()
    plt.show()
wb.check("1.11f", distances)''',
       solution=r'''from collections import Counter

n_chars = [len(holmes), len(verne)]
codes = [ord(c) for c in "Aa é"]
n_distinct = [len(set(holmes)), len(set(verne))]


def letter_freq(text):
    """The 26 frequencies of the letters a to z among the letters a-z of text (in lower case), a NumPy array."""
    counts = Counter(c for c in text.lower() if "a" <= c <= "z")
    total = sum(counts.values())
    return np.array([counts[chr(k)] / total for k in range(ord("a"), ord("z") + 1)])


half = len(holmes) // 2
d_halves = np.linalg.norm(letter_freq(holmes[:half]) - letter_freq(holmes[half:]))
d_languages = np.linalg.norm(letter_freq(holmes) - letter_freq(verne))
distances = [d_halves, d_languages]
print(n_chars, codes, n_distinct, np.round(distances, 4))
letters = [chr(k) for k in range(ord("a"), ord("z") + 1)]
fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(np.arange(26) - 0.2, letter_freq(holmes), width=0.4, label="Holmes (EN)")
ax.bar(np.arange(26) + 0.2, letter_freq(verne), width=0.4, label="Verne (FR)")
ax.set_xticks(np.arange(26), letters)
ax.set_ylabel("frequency")
ax.legend()
plt.show()''',
       record=r'''wb.record("1.11a", n_chars)
wb.record("1.11b", codes)
wb.record("1.11c", n_distinct)
wb.record("1.11d", letter_freq(holmes), decimals=3)
wb.record("1.11e", letter_freq(verne), decimals=3)
wb.record("1.11f", distances, decimals=4, mistakes={"garde l'ordre demandé : d'abord les deux moitiés de Holmes, puis Holmes contre Verne": distances[::-1]})''',
       note="Le français utilise des lettres accentuées (é, è, à, ç…) : 103 caractères différents contre 88. "
            "Les deux moitiés de *Holmes* ont presque le même vecteur de fréquences, alors que Holmes et Verne sont "
            "environ vingt fois plus éloignés : les fréquences des lettres sont une signature de la langue (un `w` "
            "sur 40 lettres en anglais, presque aucun en français). Un classifieur de langue s'en sert (ch. 6 et 13)."),

    Ex("1.12", "📦", 2, 20, "Taches solaires : tracer, lisser, repérer le cycle",
       "explorer une série temporelle réelle, la débruiter par une moyenne mobile et mesurer son cycle.",
       "0A (pandas) · 0B.4 (moyenne mobile) · fiche §1.3.2, §1.4.2 · livre §1.3.2, §1.4.2",
       thread="taches solaires", tracks="C",
       body=r"""Le nombre mensuel de taches solaires a été reconstitué depuis 1749 et il est tenu à jour par le SILSO, à l'observatoire royal de Belgique. La série est très bruitée, mais cache un cycle célèbre. Deux idées du livre s'y rencontrent : le **débruitage** (§1.4.2) et la **régression** (§1.3.2 : prévoir la valeur du mois suivant, que tu feras au ch. 22).

a) `n_months` : le nombre de mois de la série `sun` (déjà limitée aux mois définitifs).
b) `record_year` : l'année du mois qui compte le plus de taches (`idxmax`, puis la colonne `year`).
c) `smooth` : la moyenne mobile sur 13 mois, **exactement** `sun["sunspots"].rolling(13).mean()` (la moyenne du mois et des 12 précédents, 0B.4) ; puis `n_missing`, le nombre de valeurs manquantes (`NaN`) de `smooth`. D'où viennent-elles ?
d) `peak_year` : l'année où la série **lissée** est la plus haute.
e) `peaks` : les positions des sommets de la série lissée, données par l'outil fourni `local_maxima(smooth.to_numpy(), 60)` (un mois est un sommet s'il est le plus haut à 60 mois près, avant comme après) ; puis `n_peaks`, leur nombre.
f) `mean_period` : la durée moyenne, en années (1 décimale), entre deux sommets successifs ; utilise la colonne `decimal_year` aux positions `peaks` et `np.diff`.

La vérification trace la série brute, la série lissée et les sommets.""",
       given=r'''sun = wb.datasets.load_sunspots(definitive_only=True)   # monthly sunspot numbers, final values only
print(sun[["date", "sunspots"]].head(3))


def local_maxima(values, half_window):
    """Positions t where values[t] is the largest value of values[t - half_window : t + half_window + 1]."""
    positions = []
    for t in range(half_window, len(values) - half_window):
        window = values[t - half_window: t + half_window + 1]
        if not np.isnan(window).any() and values[t] == window.max():
            positions.append(t)
    return positions''',
       todo=r'''n_months = ...      # a)
record_year = ...   # b)
smooth = ...        # c) the 13-month moving average
n_missing = ...     # c)
peak_year = ...     # d)
peaks = ...         # e) a list of positions
n_peaks = ...       # e)
mean_period = ...   # f) in years''',
       check=r'''wb.check("1.12a", n_months)
wb.check("1.12b", record_year)
wb.check("1.12c", n_missing)
wb.check("1.12d", peak_year)
wb.check("1.12e", n_peaks)
wb.check("1.12f", mean_period)
if smooth is not ... and peaks is not ...:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(sun["decimal_year"], sun["sunspots"], color="0.75", lw=0.6, label="monthly")
    ax.plot(sun["decimal_year"], smooth, lw=1.8, label="13-month moving average")
    ax.plot(sun["decimal_year"].to_numpy()[peaks], smooth.to_numpy()[peaks], "o", label="peaks")
    ax.set_xlabel("year")
    ax.set_ylabel("sunspots")
    ax.legend(loc="upper left")
    plt.show()''',
       solution=r'''n_months = len(sun)
record_year = int(sun.loc[sun["sunspots"].idxmax(), "year"])
smooth = sun["sunspots"].rolling(13).mean()
n_missing = int(smooth.isna().sum())
peak_year = int(sun.loc[smooth.idxmax(), "year"])
peaks = local_maxima(smooth.to_numpy(), 60)
n_peaks = len(peaks)
mean_period = float(np.diff(sun["decimal_year"].to_numpy()[peaks]).mean())
print(n_months, record_year, n_missing, peak_year, n_peaks, round(mean_period, 2))
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(sun["decimal_year"], sun["sunspots"], color="0.75", lw=0.6, label="monthly")
ax.plot(sun["decimal_year"], smooth, lw=1.8, label="13-month moving average")
ax.plot(sun["decimal_year"].to_numpy()[peaks], smooth.to_numpy()[peaks], "o", label="peaks")
ax.set_xlabel("year")
ax.set_ylabel("sunspots")
ax.legend(loc="upper left")
plt.show()''',
       record=r'''wb.record("1.12a", n_months)
wb.record("1.12b", record_year, mistakes={"c'est le sommet de la série LISSÉE (question d) : ici, le mois brut record": peak_year})
wb.record("1.12c", n_missing, mistakes={"le mois t a besoin des 12 mois précédents : combien de mois au début n'en ont pas assez ?": 13})
wb.record("1.12d", peak_year, mistakes={"c'est le mois brut record (question b) : ici, le sommet de la série LISSÉE": record_year})
wb.record("1.12e", n_peaks)
years_12 = sun["decimal_year"].to_numpy()[peaks]
wb.record("1.12f", mean_period, decimals=1, mistakes={"entre 24 sommets, il y a 23 intervalles : c'est la moyenne des ÉCARTS (np.diff)": (years_12[-1] - years_12[0]) / n_peaks})''',
       note="Le mois record (1778) n'est pas l'année du plus haut sommet lissé (1958) : un mois isolé peut être "
            "exceptionnel sans que le cycle le soit. Les 12 premiers mois n'ont pas assez d'historique pour une "
            "moyenne sur 13 mois. Le cycle solaire dure en moyenne environ 11 ans, mais chaque cycle varie de "
            "9 à 14 ans environ, et son amplitude aussi : c'est ce qui rend la prévision difficile (ch. 22)."),

    Ex("1.13", "🛠️", 2, 15, "Lire les data cards des quatre fils rouges",
       "lire la fiche d'un dataset (provenance, licence, biais, limites) avant de s'en servir.",
       "Ex 1.9 à 1.12 · `data/cards/` · livre §1.8", tracks="C",
       body=r"""Un professionnel ne se sert jamais d'un dataset sans lire sa **data card** (*fiche de données*) : d'où viennent les données, sous quelle licence, quels biais et quelles limites. Le workbook en a une par dataset, dans `data/cards/`. La cellule suivante affiche le catalogue et les fiches des quatre fils rouges.

Réponds dans la cellule **📝 Mes réponses** :
1. Pour chacun des cinq datasets (Penguins, MNIST, Holmes, Verne, taches solaires) : sa **licence**, et peux-tu t'en servir dans un **produit vendu** ?
2. Quelle limite de Penguins pourrait pousser un modèle à « tricher » en regardant une autre colonne que la morphologie ?
3. Pourquoi la fiche de MNIST dit-elle qu'un modèle médiocre peut y paraître bon ?
4. En quoi Holmes et Verne diffèrent-ils des textes sur lesquels on entraîne un LLM ?
5. Quelles précautions prendre pour les taches solaires avant d'enregistrer des résultats (colonne `definitive`) ?""",
       given=r'''print(wb.datasets.list_datasets()[["dataset", "type", "versionné", "fiche"]].to_string(index=False))
for name in ["penguins", "mnist", "holmes", "verne", "sunspots"]:
    print("=" * 80)
    wb.datasets.dataset_card(name)''',
       after=[("todo_md", "📝 **Mes réponses** (1.13) :\n\n1. …\n2. …\n3. …\n4. …\n5. …"),
              ("solution_md", "**Réponses (1.13)** :\n\n"
               "1. Penguins : CC0 (domaine public), oui. MNIST : CC BY-SA 3.0, oui en citant les auteurs et en "
               "partageant dans les mêmes conditions tout dataset dérivé. Holmes et Verne : domaine public "
               "(Project Gutenberg), oui (la licence Gutenberg ne porte que sur les fichiers qui gardent son nom et "
               "son en-tête). Taches solaires : **CC BY-NC 4.0**, attribution obligatoire et **pas d'usage "
               "commercial**.\n"
               "2. Les espèces sont liées aux îles (les Gentoo ne viennent que de Biscoe) : un modèle qui voit "
               "`island` peut apprendre l'île au lieu de la morphologie, et se tromper dès qu'on change d'île.\n"
               "3. MNIST est « résolu » : les meilleurs modèles dépassent 99,7 % d'accuracy, et même un modèle "
               "simple dépasse 90 %. Un bon score sur MNIST prouve donc peu de chose ; Fashion-MNIST est plus "
               "exigeant.\n"
               "4. Un seul auteur, un seul genre, une langue du XIXᵉ siècle, quelques centaines de milliers de "
               "caractères, contre des milliers de milliards de tokens de sources variées pour un LLM ; et les "
               "représentations sociales de l'époque.\n"
               "5. Les derniers mois sont provisoires et peuvent être révisés : on les exclut "
               "(`definitive_only=True`) et on note la date de téléchargement ; les mesures du XVIIIᵉ siècle sont "
               "beaucoup plus bruitées.")],
       note="Au ch. 29, tu écriras la data card du dataset de ton projet final sur ce modèle : provenance, "
            "licence, taille, variables, biais et limites."),
])

# ---------------------------------------------------------------------------
# Part B: learning from labelled examples (1.14 to 1.20)
# ---------------------------------------------------------------------------
PART_B_GIVEN = r'''# Penguins, split once and for all: a training set and a test set set aside
penguins = wb.datasets.load_penguins(dropna=True)     # the 333 complete penguins
MEASURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
rng_split = np.random.default_rng(42)                  # the random draw of the penguins set aside
order = rng_split.permutation(len(penguins))
test = penguins.iloc[order[:100]].reset_index(drop=True)    # 100 penguins set aside: the test set
train = penguins.iloc[order[100:]].reset_index(drop=True)   # 233 penguins to learn from: the training set
X_train, y_train = train[MEASURES], train["species"]
X_test, y_test = test[MEASURES], test["species"]


def accuracy(y_true, y_pred):
    """Proportion of correct predictions."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def predict_with(rule, X):
    """Apply rule(bill_length, bill_depth, flipper_length, body_mass) to every row of X."""
    return np.array([rule(*row) for row in X[MEASURES].to_numpy()])


print(f"training set: {len(train)} penguins · test set: {len(test)} penguins")'''

PART_B = Part("B", "Apprendre à partir d'exemples étiquetés",
              "Fiche §1.1.2 et §1.2. Les 333 manchots complets sont tirés au sort une fois pour toutes : 233 pour "
              "apprendre (le **jeu d'entraînement**) et 100 mis de côté (le **jeu de test**), qui ne servent qu'à "
              "mesurer si ce qui a été appris se **généralise**. Tu compares un modèle qui apprend par cœur, des "
              "règles écrites à la main, une vraie boucle d'entraînement et un modèle qui apprend ses règles tout seul.",
              given=PART_B_GIVEN, exercises=[
    Ex("1.14", "🔮", 2, 15, "Mémoriser n'est pas apprendre",
       "voir qu'un score parfait sur les exemples appris ne dit rien de la généralisation.",
       "Ex 1.9 · fiche §1.2.1, §1.2.3 · livre §1.2.1 à §1.2.3", thread="Penguins", tracks="R, C", hypothesis=True,
       body=r"""Le `Memorizer` ci-dessous apprend « par cœur », comme l'école absurde du livre (§1.2.1) : pendant l'entraînement (`fit`), il range chaque manchot dans une table (ses 4 mesures → son espèce). Pour prédire (`predict`), il cherche les mesures du manchot dans sa table ; s'il ne les trouve pas, il répond l'espèce la plus fréquente de l'entraînement. Il a la même interface que les modèles de scikit-learn : `fit`, puis `predict` (encadré 🧮 de la fiche).

**Sans rien exécuter**, classe l'accuracy qu'il obtiendra dans une catégorie : `"parfaite"` (100 %), `"bonne"` (de 80 % à 100 % exclu), `"moyenne"` (de 50 % à 80 % exclu) ou `"mauvaise"` (moins de 50 %).

a) sur le **jeu d'entraînement** (les manchots qu'il a appris) ;
b) sur le **jeu de test** (100 manchots qu'il n'a jamais vus).

Écris ton hypothèse (cellule 📝), puis tes deux prédictions ; la vérification ne regarde que tes prédictions. Ensuite seulement, exécute l'**Expérience**.""",
       given=r'''class Memorizer:
    """Learns by heart: a table (4 measures -> species). Unknown penguin: the most frequent species."""

    def fit(self, X, y):
        self.table_ = {tuple(row): label for row, label in zip(X.to_numpy(), y)}
        self.default_ = y.value_counts().idxmax()
        return self

    def predict(self, X):
        return np.array([self.table_.get(tuple(row), self.default_) for row in X.to_numpy()])''',
       todo=r'''prediction_1_14a = ...   # a) on the training set: "parfaite", "bonne", "moyenne" or "mauvaise"
prediction_1_14b = ...   # b) on the test set''',
       check=r'''wb.check("1.14a", prediction_1_14a)
wb.check("1.14b", prediction_1_14b)''',
       solution=r'''prediction_1_14a = "parfaite"
prediction_1_14b = "mauvaise"''',
       record=r'''wb.record("1.14a", prediction_1_14a, mistakes={"chaque manchot d'entraînement est DANS la table : le mémoriseur ne peut pas se tromper sur eux": "bonne"})
wb.record("1.14b", prediction_1_14b, mistakes={"un manchot du test a-t-il exactement les mêmes 4 mesures qu'un manchot appris ? Que répond alors le mémoriseur ?": "bonne",
                                                "retrouver un manchot appris est facile ; mais les manchots du test sont nouveaux": "parfaite",
                                                "regarde ce que le mémoriseur répond pour un manchot inconnu, et la part de cette espèce dans le test": "moyenne"})''',
       after=[("md", "**Expérience** : exécute la cellule et compare avec tes prédictions."),
              ("code", r'''memorizer = Memorizer().fit(X_train, y_train)
print("accuracy on the training set:", accuracy(y_train, memorizer.predict(X_train)))
print("accuracy on the test set:    ", accuracy(y_test, memorizer.predict(X_test)))'''),
              ("md", "c) `n_seen` : combien de manchots du jeu de test figurent dans la table du mémoriseur (`tuple(row) in memorizer.table_`) ?  \n"
                     "d) `default_species` : l'espèce qu'il répond alors pour eux. Compare l'accuracy sur le test avec la part de cette espèce dans `y_test`."),
              ("todo", r'''n_seen = ...            # c)
default_species = ...   # d)'''),
              ("check", r'''wb.check("1.14c", n_seen)
wb.check("1.14d", default_species)'''),
              ("solution", r'''n_seen = sum(tuple(row) in memorizer.table_ for row in X_test.to_numpy())
default_species = memorizer.default_
print(n_seen, default_species, (y_test == default_species).mean())'''),
              ("record", r'''wb.record("1.14c", n_seen)
wb.record("1.14d", default_species)''')],
       note="Sur l'entraînement, le mémoriseur est parfait : il retrouve chaque réponse dans sa table. Aucun manchot "
            "du test n'a exactement les mêmes mesures qu'un manchot appris, donc il répond toujours « Adelie », "
            "l'espèce la plus fréquente, et son accuracy (40 %) est simplement la part des Adélie dans le test. "
            "Un score sur les données d'entraînement ne mesure pas la **généralisation** : c'est tout l'intérêt "
            "du jeu de test mis de côté."),

    Ex("1.15", "🔨", 2, 20, "Un système expert pour les manchots",
       "programmer des règles d'expert, mesurer leur accuracy et repérer les cas qu'elles ratent.",
       "Ex 1.9 · 0A (`if`/`elif`, `pd.crosstab`) · fiche §1.1.2 · livre §1.1.2", thread="Penguins", tracks="C",
       body=r"""Une biologiste te donne ses règles pour reconnaître l'espèce d'un manchot :
1. « Un manchot de plus de 4 700 g est un Gentoo. »
2. « Sinon, un bec de plus de 45 mm de long signe un Chinstrap. »
3. « Sinon, c'est un Adélie. »

C'est un petit **système expert** (§1.1.2) : des règles écrites à la main, sans rien apprendre des données.

Écris `expert_rule(bill_length, bill_depth, flipper_length, body_mass)`, qui renvoie `"Gentoo"`, `"Chinstrap"` ou `"Adelie"` en appliquant **exactement** ces trois règles (« plus de » : strictement plus grand). L'outil fourni `predict_with(expert_rule, X)` l'applique à chaque manchot d'un tableau.

a) `expert_train_acc` : l'accuracy de ces règles sur le jeu d'entraînement (3 décimales).
b) `expert_test_acc` : leur accuracy sur le jeu de test (2 décimales).
c) `worst_species` : l'espèce dont les manchots d'entraînement sont **le plus souvent** mal classés (regarde `pd.crosstab(y_train, predictions)`, affiché par la vérification). Pourquoi ces manchots-là sont-ils mal classés ?""",
       given=r'''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for species in ["Adelie", "Chinstrap", "Gentoo"]:
    part = train[train["species"] == species]
    axes[0].scatter(part["body_mass_g"], part["bill_length_mm"], s=14, label=species)
    axes[1].scatter(part["flipper_length_mm"], part["bill_length_mm"], s=14, label=species)
axes[0].axvline(4700, color="0.3", ls="--")
axes[0].axhline(45, color="0.3", ls=":")
axes[0].set_xlabel("body_mass_g")
axes[1].set_xlabel("flipper_length_mm")
for ax in axes:
    ax.set_ylabel("bill_length_mm")
    ax.legend()
fig.suptitle("Training set: the expert's thresholds (dashed: 4 700 g, dotted: 45 mm)")
plt.show()''',
       todo=r'''def expert_rule(bill_length, bill_depth, flipper_length, body_mass):
    """The biologist's three rules: returns "Gentoo", "Chinstrap" or "Adelie"."""
    raise NotImplementedError("expert_rule() is not written yet")


expert_train_acc = ...   # a)
expert_test_acc = ...    # b)
worst_species = ...      # c)''',
       check=r'''with wb.attempt("1.15"):
    print(pd.crosstab(y_train, predict_with(expert_rule, X_train), rownames=["true"], colnames=["predicted"]))
wb.check("1.15a", expert_train_acc)
wb.check("1.15b", expert_test_acc)
wb.check("1.15c", worst_species)''',
       solution=r'''def expert_rule(bill_length, bill_depth, flipper_length, body_mass):
    """The biologist's three rules: returns "Gentoo", "Chinstrap" or "Adelie"."""
    if body_mass > 4700:
        return "Gentoo"
    if bill_length > 45:
        return "Chinstrap"
    return "Adelie"


expert_pred_train = predict_with(expert_rule, X_train)
expert_train_acc = accuracy(y_train, expert_pred_train)
expert_test_acc = accuracy(y_test, predict_with(expert_rule, X_test))
table_15 = pd.crosstab(y_train, expert_pred_train, rownames=["true"], colnames=["predicted"])
print(table_15)
errors_by_species = (expert_pred_train != y_train).groupby(y_train).mean()
worst_species = errors_by_species.idxmax()
print(round(expert_train_acc, 3), expert_test_acc, errors_by_species.round(3).to_dict())''',
       record=r'''STRICT_15 = "« plus de » veut dire STRICTEMENT plus grand (>) : des manchots pèsent exactement 4 700 g"
wb.record("1.15a", expert_train_acc, decimals=3, mistakes={STRICT_15: 0.871})
wb.record("1.15b", expert_test_acc, decimals=2, mistakes={"c'est l'accuracy sur l'ENTRAÎNEMENT : applique les règles à X_test": round(expert_train_acc, 2),
                                                         STRICT_15: 0.91})
wb.record("1.15c", worst_species, mistakes={"regarde les LIGNES du tableau (les vraies espèces) : quelle espèce a le plus d'erreurs, proportionnellement ?": "Adelie"})''',
       note="La règle 1 confond « gros » et « Gentoo » : les Gentoo les plus légers (souvent des femelles) pèsent "
            "moins de 4 700 g et tombent dans les règles 2 et 3, où ils deviennent Chinstrap ou Adélie. Il faudrait "
            "une règle de plus pour ce cas particulier, puis une autre pour le suivant… C'est le problème des "
            "systèmes experts (§1.1.2), et l'occasion de remarquer que la nageoire (graphique de droite) sépare "
            "bien mieux les Gentoo que la masse."),

    Ex("1.16", "🔨", 2, 30, "La boucle d'entraînement à la main",
       "coder une boucle d'entraînement complète (prédire, mesurer l'erreur, corriger) pour une droite.",
       "0B (fonction affine, loss, 101.5.3) · Rappel 1.R3 · fiche §1.2.2 · livre §1.2.2, §1.2.4 (fig. 1.8)",
       thread="synthétique", tracks="R, M, C",
       body=r"""Les points `(x_line, y_line)` suivent à peu près une droite. Le modèle est une droite $\hat{y} = w\,x + b$ : ses deux **paramètres** sont $w$ et $b$. Pour l'entraîner, on part de $w = b = 0$, puis on montre les 50 échantillons **un par un, dans l'ordre** ; pour chacun, on fait l'étape de la figure 1.8 du livre :
1. **prédire** $\hat{y} = w\,x + b$ ;
2. **comparer** : l'erreur est $y - \hat{y}$ ;
3. **corriger** les paramètres, avec la règle (admise ici, justifiée aux ch. 5 et 19) :
$$w \leftarrow w + \eta\,(y - \hat{y})\,x \qquad b \leftarrow b + \eta\,(y - \hat{y})$$
où $\eta$ est le **learning rate**. Un passage complet sur les 50 échantillons est une **epoch**. Pour suivre l'entraînement, on calcule après chaque epoch la **loss** : l'erreur quadratique moyenne $L = \frac{1}{n}\sum_i (\hat{y}_i - y_i)^2$ sur tous les échantillons (0B, 101.1.3).

Écris les quatre fonctions, puis vérifie :
a) `mse(0, 0, x_line, y_line)` : la loss avant tout entraînement (3 décimales) ;
b) `train_step(0.0, 0.0, x_line[0], y_line[0], 0.01)` : les nouveaux $(w, b)$ après **une seule** correction (4 décimales) ;
c) $(w, b)$ après **une** epoch avec $\eta = 0{,}01$ (4 décimales) ;
d) $(w, b)$ après **20** epochs avec $\eta = 0{,}01$ (3 décimales) ;
e) la loss après ces 20 epochs (4 décimales).

La vérification trace la droite après 0, 1 et 20 epochs, et la loss epoch par epoch. Les données ont été fabriquées avec $w = 2$ et $b = 1$, plus du bruit : retrouves-tu ces valeurs ?""",
       given=r'''X_line, y_line = wb.synth.make_linear(n=50, w=(2.0,), b=1.0, noise=0.5, seed=0)   # y ≈ 2x + 1 + noise
x_line = X_line[:, 0]                                                              # a 1-D array of 50 numbers
print(np.round(x_line[:3], 3), np.round(y_line[:3], 3))''',
       todo=r'''def predict_line(w, b, x):
    """The prediction y_hat = w x + b (x can be a number or a NumPy array)."""
    raise NotImplementedError("predict_line() is not written yet")


def mse(w, b, x, y):
    """Mean squared error of the line (w, b) on the samples (x, y)."""
    raise NotImplementedError("mse() is not written yet")


def train_step(w, b, x_i, y_i, eta):
    """One correction on one sample (x_i, y_i): returns the new (w, b)."""
    raise NotImplementedError("train_step() is not written yet")


def train_line(x, y, eta, n_epochs):
    """Start from w = b = 0, then n_epochs passes over the samples, in order.

    Returns (w, b, losses): losses[k] is the MSE after k epochs (losses[0]: before training).
    """
    raise NotImplementedError("train_line() is not written yet")''',
       check=r'''with wb.attempt("1.16"):
    wb.check("1.16a", mse(0, 0, x_line, y_line))
    wb.check("1.16b", list(train_step(0.0, 0.0, x_line[0], y_line[0], 0.01)))
    wb.check("1.16c", list(train_line(x_line, y_line, 0.01, 1)[:2]))
    w_16, b_16, losses_16 = train_line(x_line, y_line, 0.01, 20)
    wb.check("1.16d", [w_16, b_16])
    wb.check("1.16e", losses_16[-1])
    verdict("1.16", len(losses_16) == 21 and abs(losses_16[0] - mse(0, 0, x_line, y_line)) < 1e-9,
            "losses contient bien la loss avant l'entraînement, puis une loss par epoch (21 valeurs).",
            "losses doit contenir 21 valeurs : la loss AVANT l'entraînement (w = b = 0), puis une par epoch.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].scatter(x_line, y_line, s=14, color="0.5")
    grid = np.linspace(-3, 3, 2)
    for epochs in (0, 1, 20):
        w, b, _ = train_line(x_line, y_line, 0.01, epochs)
        axes[0].plot(grid, predict_line(w, b, grid), label=f"after {epochs} epoch(s)")
    axes[0].legend()
    axes[1].plot(losses_16, marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss (MSE)")
    plt.show()''',
       solution=r'''def predict_line(w, b, x):
    """The prediction y_hat = w x + b (x can be a number or a NumPy array)."""
    return w * x + b


def mse(w, b, x, y):
    """Mean squared error of the line (w, b) on the samples (x, y)."""
    return float(np.mean((predict_line(w, b, x) - y) ** 2))


def train_step(w, b, x_i, y_i, eta):
    """One correction on one sample (x_i, y_i): returns the new (w, b)."""
    error = y_i - predict_line(w, b, x_i)
    return w + eta * error * x_i, b + eta * error


def train_line(x, y, eta, n_epochs):
    """Start from w = b = 0, then n_epochs passes over the samples, in order.

    Returns (w, b, losses): losses[k] is the MSE after k epochs (losses[0]: before training).
    """
    w, b = 0.0, 0.0
    losses = [mse(w, b, x, y)]
    for _ in range(n_epochs):
        for x_i, y_i in zip(x, y):
            w, b = train_step(w, b, x_i, y_i, eta)
        losses.append(mse(w, b, x, y))
    return w, b, losses


w_16, b_16, losses_16 = train_line(x_line, y_line, 0.01, 20)
print(mse(0, 0, x_line, y_line), train_step(0.0, 0.0, x_line[0], y_line[0], 0.01))
print(train_line(x_line, y_line, 0.01, 1)[:2], (w_16, b_16), losses_16[-1])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].scatter(x_line, y_line, s=14, color="0.5")
grid = np.linspace(-3, 3, 2)
for epochs in (0, 1, 20):
    w, b, _ = train_line(x_line, y_line, 0.01, epochs)
    axes[0].plot(grid, predict_line(w, b, grid), label=f"after {epochs} epoch(s)")
axes[0].legend()
axes[1].plot(losses_16, marker="o")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("loss (MSE)")
plt.show()''',
       record=r'''wb.record("1.16a", mse(0, 0, x_line, y_line), decimals=3)
step_16 = train_step(0.0, 0.0, x_line[0], y_line[0], 0.01)
wb.record("1.16b", list(step_16), decimals=4, mistakes={"le signe de l'erreur : c'est y − ŷ (la vraie valeur MOINS la prédiction)": [-v for v in step_16]})
wb.record("1.16c", list(train_line(x_line, y_line, 0.01, 1)[:2]), decimals=4)
wb.record("1.16d", [w_16, b_16], decimals=3)
wb.record("1.16e", losses_16[-1], decimals=4)''',
       note="Après une epoch, la droite est déjà proche ; après 20, $w \\approx 2{,}04$ et $b \\approx 1{,}01$ : pas "
            "exactement 2 et 1, car le bruit déplace un peu la meilleure droite. Remarque : quand la prédiction est "
            "juste, l'erreur vaut 0 et la correction aussi ; plus l'erreur est grande, plus on corrige. La règle "
            "vaut $-\\frac{\\eta}{2}$ fois la dérivée de $(\\hat{y} - y)^2$ calculée en 0B (101.5.3) : c'est déjà "
            "une descente de gradient (ch. 5 et 19)."),

    Ex("1.17", "🔬", 2, 20, "Learning rate : trop prudent, trop pressé",
       "observer l'effet du learning rate sur la vitesse et la stabilité de l'entraînement.",
       "Ex 1.16 · fiche §1.2.2 · livre §1.2.2, §1.2.4", thread="synthétique", tracks="C",
       body=r"""Reprends `train_line` de 1.16 et entraîne la droite pendant 20 epochs avec quatre learning rates : $\eta = 0{,}001$, $0{,}01$, $0{,}1$ et $0{,}5$.

Écris `losses_by_eta()`, qui renvoie le dictionnaire `{eta: liste des losses}` pour les quatre valeurs de `ETAS_17` (une ligne de compréhension suffit). La vérification trace la loss epoch par epoch : à gauche les quatre courbes (échelle logarithmique), à droite les trois premières seules, pour les voir de près ; elle affiche aussi, pour chaque $\eta$, la loss après 1 et après 20 epochs, et la valeur finale de $w$. Réponds ensuite dans la cellule **📝 Mes réponses** :
1. Quel learning rate est « trop prudent » ? À quoi le vois-tu ?
2. Que se passe-t-il avec $\eta = 0{,}5$ ? Que vaut $w$ après 20 epochs ?
3. Entre $0{,}01$ et $0{,}1$, lequel fait baisser la loss le plus vite au début ? Lequel finit le plus bas ?
4. Le livre propose de réduire le learning rate au fil de l'entraînement (§1.2.2). Pourquoi est-ce une bonne idée, d'après tes courbes ?""",
       given=r'''ETAS_17 = [0.001, 0.01, 0.1, 0.5]


def plot_losses_17(results):
    """Left: the four loss curves (log scale). Right: the first three only (a closer look)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, etas in zip(axes, [ETAS_17, ETAS_17[:3]]):
        for eta in etas:
            ax.plot(results[eta], marker="o", ms=3, label=f"eta = {eta}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (MSE, log scale)")
        ax.legend()
    axes[1].set_title("without eta = 0.5")
    plt.show()''',
       todo=r'''def losses_by_eta():
    """A dict {eta: the list of losses of train_line(x_line, y_line, eta, 20)} for every eta of ETAS_17."""
    raise NotImplementedError("losses_by_eta() is not written yet")''',
       check=r'''with wb.attempt("1.17"):
    results_17 = losses_by_eta()
    plot_losses_17(results_17)
    for eta in ETAS_17:
        w, b, losses = train_line(x_line, y_line, eta, 20)
        print(f"eta = {eta:<6} loss after 1 epoch: {losses[1]:10.4g}   after 20: {losses[-1]:10.4g}   final w = {w:.4g}")
    verdict("1.17", results_17[0.001][-1] > results_17[0.01][-1] and results_17[0.5][-1] > 1e6,
            "tes courbes montrent bien un learning rate trop prudent et un autre qui fait exploser la loss.",
            "tes courbes ne ressemblent pas à celles attendues : vérifie train_line (1.16) et le dictionnaire.")''',
       solution=r'''def losses_by_eta():
    """A dict {eta: the list of losses of train_line(x_line, y_line, eta, 20)} for every eta of ETAS_17."""
    return {eta: train_line(x_line, y_line, eta, 20)[2] for eta in ETAS_17}


results_17 = losses_by_eta()
plot_losses_17(results_17)
for eta in ETAS_17:
    w, b, losses = train_line(x_line, y_line, eta, 20)
    print(f"eta = {eta:<6} loss after 1 epoch: {losses[1]:10.4g}   after 20: {losses[-1]:10.4g}   final w = {w:.4g}")''',
       after=[("todo_md", "📝 **Mes réponses** (1.17) :\n\n1. …\n2. …\n3. …\n4. …"),
              ("solution_md", "**Réponses (1.17)** :\n\n"
               "1. $\\eta = 0{,}001$ : la loss baisse, mais lentement (sur le graphique de droite, c'est la courbe la "
               "plus haute à la fin) ; après 20 epochs, elle est encore loin du minimum (environ 0,38 contre 0,25).\n"
               "2. Avec $\\eta = 0{,}5$, chaque correction dépasse la bonne valeur et en rajoute : la loss est "
               "multipliée à chaque epoch et atteint environ $10^{53}$ ; $w$ vaut quelque $10^{26}$. "
               "L'entraînement **diverge**.\n"
               "3. $\\eta = 0{,}1$ descend le plus vite au début (dès la première epoch, la loss tombe à 0,28, "
               "son niveau final), mais elle reste au-dessus de celle de $\\eta = 0{,}01$ (0,25) : les grands pas font "
               "osciller la droite autour de la meilleure position sans jamais s'y poser. $\\eta = 0{,}01$ finit le plus bas.\n"
               "4. Il faut de grands pas au début pour avancer vite, et de petits pas à la fin pour se poser au "
               "fond : réduire $\\eta$ au fil des epochs combine les deux (les plannings de learning rate du ch. 19).")],
       note="Le learning rate est un **hyperparamètre** : c'est toi qui le choisis, par essais successifs. Trop petit, "
            "l'entraînement traîne ; trop grand, il oscille ou diverge. Il n'existe pas de valeur universelle : elle "
            "dépend du modèle et des données (ici, de la taille des $x$)."),

    Ex("1.18", "📦", 2, 20, "Un arbre de décision apprend les règles à ta place",
       "entraîner un modèle de scikit-learn en boîte noire, l'évaluer sur le jeu de test et lire les règles qu'il a apprises.",
       "Ex 1.9, Ex 1.14 · fiche 🧮 « scikit-learn en boîte noire » · livre §1.1.2, §1.2.3, §1.3.1",
       thread="Penguins", tracks="R, C",
       body=r"""Au lieu d'écrire des règles, laissons un algorithme les **apprendre** à partir des exemples d'entraînement. Un **arbre de décision** (détaillé au ch. 13) pose des questions du type « bec ≤ 40 mm ? » et les choisit lui-même. Les trois lignes de scikit-learn sont fournies : on crée le modèle en fixant ses **hyperparamètres** (ici, au plus deux questions à la suite : `max_depth=2`), puis `fit` apprend ses **paramètres** (les questions et les seuils) sur le jeu d'entraînement.

a) `tree_train_acc` : son accuracy sur le jeu d'entraînement (`tree.score(X, y)`, 3 décimales).
b) `tree_test_acc` : son accuracy sur le jeu de test (2 décimales).
c) `root_feature` : le nom de la feature de la **première** question de l'arbre (lis `rules`, affiché par la cellule fournie).
d) `root_threshold` : le seuil de cette première question (1 décimale).
e) `gain_vs_expert` : combien de manchots du jeu de test l'arbre classe-t-il correctement **en plus** des règles de la biologiste (1.15) ? (un entier)""",
       given=r'''from sklearn.tree import DecisionTreeClassifier, export_text

tree = DecisionTreeClassifier(max_depth=2, random_state=0)   # hyperparameters: chosen by you
tree.fit(X_train, y_train)                                    # parameters: learned from the training set
rules = export_text(tree, feature_names=MEASURES)
print(rules)''',
       todo=r'''tree_train_acc = ...   # a)
tree_test_acc = ...    # b)
root_feature = ...     # c) a column name
root_threshold = ...   # d)
gain_vs_expert = ...   # e) an integer''',
       check=r'''wb.check("1.18a", tree_train_acc)
wb.check("1.18b", tree_test_acc)
wb.check("1.18c", root_feature)
wb.check("1.18d", root_threshold)
wb.check("1.18e", gain_vs_expert)''',
       solution=r'''tree_train_acc = tree.score(X_train, y_train)
tree_test_acc = tree.score(X_test, y_test)
root_feature = MEASURES[tree.tree_.feature[0]]      # or read the first line of `rules`
root_threshold = float(tree.tree_.threshold[0])
gain_vs_expert = round((tree_test_acc - expert_test_acc) * len(test))
print(round(tree_train_acc, 3), tree_test_acc, root_feature, root_threshold, gain_vs_expert)''',
       record=r'''wb.record("1.18a", tree_train_acc, decimals=3)
wb.record("1.18b", tree_test_acc, decimals=2)
wb.record("1.18c", root_feature, mistakes={"la première question est la PREMIÈRE ligne de rules (la moins indentée)": "bill_length_mm"})
wb.record("1.18d", root_threshold, decimals=1)
wb.record("1.18e", gain_vs_expert)''',
       note="L'arbre a trouvé seul des règles proches de celles d'un expert… mais meilleures : sa première question "
            "porte sur la nageoire, qui sépare nettement les Gentoo, et non sur la masse. Il fait 97 % sur le test "
            "contre 92 % pour la biologiste. On n'a jamais écrit une seule règle : on a fourni des exemples "
            "étiquetés. C'est la promesse du machine learning (§1.1.2)."),

    Ex("1.19", "🔮", 2, 15, "Un manchot d'une espèce jamais vue",
       "prévoir, en lisant ses règles, ce que répond un classifieur face à un cas hors de ses classes.",
       "Ex 1.18 · fiche §1.3.1 · livre §1.3.1 (fig. 1.10)", thread="Penguins", tracks="C", hypothesis=True,
       body=r"""L'arbre de 1.18 ne connaît que trois espèces. Deux visiteurs se présentent :
- un **manchot empereur**, la plus grande espèce (jusqu'à environ 40 kg, un bec d'environ 8 cm) : on prend un bec de 80 mm de long, une masse de 30 000 g, et, pour les deux mesures que personne n'a relevées ici, des valeurs **imaginées** pour l'exercice : 22 mm d'épaisseur de bec et 340 mm de nageoire ;
- un **poussin imaginaire**, minuscule : bec de 20 mm sur 9 mm, nageoire de 90 mm, 900 g.

**Sans rien exécuter**, en lisant seulement les règles de l'arbre (`rules`, affiché en 1.18), prévois sa réponse : `"Adelie"`, `"Chinstrap"`, `"Gentoo"` ou `"inconnu"` (s'il peut refuser de répondre).

a) pour l'empereur ; b) pour le poussin.

Écris ton hypothèse (cellule 📝), puis tes deux prédictions, et seulement ensuite l'**Expérience**.""",
       given=r'''visitors = pd.DataFrame([[80.0, 22.0, 340.0, 30000.0],     # an emperor penguin (2 made-up measures)
                         [20.0, 9.0, 90.0, 900.0]],           # a tiny imaginary chick
                        columns=MEASURES, index=["emperor", "chick"])
visitors''',
       todo=r'''prediction_1_19a = ...   # a) the emperor: "Adelie", "Chinstrap", "Gentoo" or "inconnu"
prediction_1_19b = ...   # b) the chick''',
       check=r'''wb.check("1.19a", prediction_1_19a)
wb.check("1.19b", prediction_1_19b)''',
       solution=r'''prediction_1_19a = "Chinstrap"   # flipper > 206.5, then bill depth > 17.65
prediction_1_19b = "Adelie"      # flipper <= 206.5, then bill length <= 43.35''',
       record=r'''wb.record("1.19a", prediction_1_19a, mistakes={"un classifieur ne peut répondre qu'avec les classes vues à l'entraînement": "inconnu",
                                                "suis les règles jusqu'au bout : après la nageoire, l'arbre pose une seconde question": "Gentoo"})
wb.record("1.19b", prediction_1_19b, mistakes={"un classifieur ne peut répondre qu'avec les classes vues à l'entraînement": "inconnu"})''',
       after=[("md", "**Expérience** : exécute la cellule. `predict_proba` donne, pour chaque visiteur, la « confiance » "
                     "de l'arbre pour chaque espèce (dans l'ordre de `tree.classes_`)."),
              ("code", r'''print(tree.predict(visitors))
print(tree.classes_)
print(tree.predict_proba(visitors).round(3))'''),
              ("md", "c) `proba_emperor` : la probabilité que l'arbre donne à sa réponse pour l'empereur (2 décimales)."),
              ("todo", r'''proba_emperor = ...   # c)'''),
              ("check", r'''wb.check("1.19c", proba_emperor)'''),
              ("solution", r'''proba_emperor = float(tree.predict_proba(visitors)[0].max())
print(proba_emperor)'''),
              ("record", r'''wb.record("1.19c", proba_emperor, decimals=2)''')],
       note="L'arbre ne peut pas dire « je ne sais pas » : il répond toujours l'une des classes qu'il connaît, comme "
            "le classifieur du livre qui prend une cuillère pour autre chose (§1.3.1). L'empereur devient un "
            "Chinstrap « à 67 % », le poussin un Adélie « à 96 % » : ces probabilités ne sont que les proportions "
            "d'espèces dans la feuille de l'arbre, pas une vraie mesure de confiance. Repérer une entrée qui ne "
            "ressemble à rien de connu est un problème à part (détection d'anomalies, hors distribution)."),

    Ex("1.20", "🐛", 2, 20, "Le score trop beau pour être vrai",
       "repérer et corriger deux erreurs qui faussent une évaluation : une fuite du label et un test vu à l'entraînement.",
       "Ex 1.18 · fiche §1.2.3, ⚠️ pièges · livre §1.2.3", thread="Penguins", tracks="C",
       body=r"""Un collègue annonce fièrement 100 % d'accuracy sur le jeu de test avec un arbre de décision. Voici son code (cellule suivante). Ce score parfait cache **deux** erreurs. Trouve-les (indice : relis la fiche, §1.2.3 et les pièges ⚠️), puis écris `train_and_evaluate_20()`, qui fait **correctement** la même chose : un `DecisionTreeClassifier(random_state=0)` entraîné **seulement** sur le jeu d'entraînement, avec des features honnêtes (`features_20`), et évalué sur le jeu de test. Elle renvoie `(model, test_accuracy)`.

Réponds ensuite dans la cellule **📝 Mes réponses** : quelles sont les deux erreurs ? Suffit-il d'en corriger une seule ? Quelle accuracy obtiens-tu après correction (une proportion entre 0 et 1), et pourquoi est-ce une meilleure nouvelle que 100 % ?""",
       given=r'''# The colleague's code (buggy!)
penguins_coded = penguins.copy()
penguins_coded["species_code"] = penguins_coded["species"].map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})  # handy for colours
FEATURES_20 = MEASURES + ["species_code"]
model_20 = DecisionTreeClassifier(random_state=0)
model_20.fit(penguins_coded[FEATURES_20], penguins_coded["species"])
test_coded = penguins_coded.iloc[order[:100]]
print("accuracy on the test set:", model_20.score(test_coded[FEATURES_20], test_coded["species"]))''',
       todo=r'''features_20 = ...   # the features you keep (a list of column names)


def train_and_evaluate_20():
    """Train a DecisionTreeClassifier(random_state=0) on the training set only; return (model, test_accuracy)."""
    raise NotImplementedError("train_and_evaluate_20() is not written yet")''',
       check=r'''with wb.attempt("1.20"):
    model_fixed, accuracy_fixed = train_and_evaluate_20()
    n_fit = int(model_fixed.tree_.n_node_samples[0])       # number of penguins the model was trained on
    used_20 = [str(c) for c in getattr(model_fixed, "feature_names_in_", features_20)]  # the columns it was trained on
    print(f"trained on {n_fit} penguins with the features {used_20}, test accuracy: {accuracy_fixed:.3f}")
    verdict("1.20", n_fit == len(train),
            "le modèle n'a vu que les 233 manchots d'entraînement.",
            f"le modèle a été entraîné sur {n_fit} manchots, pas sur les {len(train)} du jeu d'entraînement.")
    verdict("1.20", "species_code" not in used_20 and "species" not in used_20
            and model_fixed.n_features_in_ == len(features_20),
            "aucune feature ne contient la réponse.",
            "une de tes features contient encore la réponse (le label, sous une autre forme), ou le modèle "
            "n'a pas été entraîné sur features_20.")
    verdict("1.20", 0 < accuracy_fixed < 1,
            "un score réaliste, mesuré sur des manchots jamais vus.",
            "l'accuracy attendue est une proportion entre 0 et 1, et un score de 1 (100 %) cache encore une erreur.")''',
       solution=r'''features_20 = MEASURES          # species_code is the label in disguise: it must go


def train_and_evaluate_20():
    """Train a DecisionTreeClassifier(random_state=0) on the training set only; return (model, test_accuracy)."""
    model = DecisionTreeClassifier(random_state=0)
    model.fit(train[features_20], train["species"])
    return model, model.score(test[features_20], test["species"])


model_fixed, accuracy_fixed = train_and_evaluate_20()
print(model_fixed.tree_.n_node_samples[0], list(model_fixed.feature_names_in_), accuracy_fixed)''',
       after=[("todo_md", "📝 **Mes réponses** (1.20) : …"),
              ("solution_md", "**Réponses (1.20)** : 1. **Fuite du label** : `species_code` n'est que l'espèce écrite "
               "en chiffres ; l'arbre n'a qu'à la lire. 2. **Test vu à l'entraînement** : `fit` reçoit les 333 "
               "manchots, dont les 100 du test ; un arbre sans limite de profondeur apprend chaque exemple par cœur "
               "(1.14) et « retrouve » donc ceux du test. Corriger une seule erreur ne suffit pas : chacune donne "
               "encore 100 % à elle seule. Une fois les deux corrigées, l'accuracy tombe à 95 % : c'est une "
               "estimation **honnête** de ce que le modèle fera sur de nouveaux manchots, alors que 100 % était une "
               "illusion qui se serait effondrée une fois le modèle déployé.")],
       note="Un score trop beau doit toujours éveiller un soupçon. Deux vérifications à faire par réflexe : aucune "
            "feature ne contient la réponse, et le jeu de test n'a servi à rien d'autre qu'à la mesure finale. On "
            "retrouvera ces fuites de données (*data leakage*) aux ch. 8 et 12."),
])

# ---------------------------------------------------------------------------
# Part C: beyond supervised learning (1.21 to 1.24)
# ---------------------------------------------------------------------------
PART_C = Part("C", "Au-delà du supervisé : regrouper, récompenser, empiler, générer",
              "Fiche §1.4 à §1.7. Quatre familles en boîte noire ou en version miniature : un clustering sans "
              "étiquettes, un agent qui apprend par la récompense, un réseau de neurones sur MNIST et un générateur "
              "de texte.", exercises=[
    Ex("1.21", "📦", 2, 20, "Regrouper les manchots sans leurs étiquettes",
       "faire tourner un clustering sans labels et comparer les groupes trouvés aux espèces.",
       "Ex 1.9 · 0A.52 (standardisation) · fiche §1.4.1 · livre §1.4, §1.4.1 (fig. 1.13)", thread="Penguins", tracks="C",
       body=r"""Oublions les espèces : l'algorithme **k-means** (détaillé au ch. 7) ne reçoit que les 4 mesures des 333 manchots et cherche 3 groupes de manchots proches les uns des autres (proches au sens de la distance entre vecteurs, 0B). On ne regarde les espèces **qu'après**, pour juger les groupes. Les trois lignes de scikit-learn sont dans l'outil fourni `cluster(X)`.

1. Écris `purity(groups, labels)` : pour chaque groupe, on compte les manchots de l'espèce la plus représentée dans ce groupe ; la pureté est la somme de ces nombres divisée par le nombre total de manchots (1 si chaque groupe ne contient qu'une espèce). Indice : `pd.crosstab(groups, labels)`, puis `.max(axis=1)`.
2. Écris `X_scaled` : les 4 mesures **mises à la même échelle**, `(X - X.mean()) / X.std()` avec `X = penguins[MEASURES]` (chaque colonne est centrée et divisée par son écart-type : la standardisation de 0A.52 ; tu verras au ch. 12 quand et pourquoi la faire).

La vérification affiche les tableaux groupes × espèces et la pureté, sur les mesures brutes puis sur les mesures mises à l'échelle. Réponds dans la cellule **📝 Mes réponses** : pourquoi les groupes sont-ils si mauvais sur les mesures brutes (pense aux unités ; voir aussi le rappel 1.R1) ? Quelles espèces restent mélangées ? Une biologiste aurait-elle pu découvrir les trois espèces de cette façon ?""",
       given=r'''from sklearn.cluster import KMeans


def cluster(X):
    """Groups found by k-means (3 groups) from the measures X only: no species given."""
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
    return kmeans.fit_predict(X)''',
       todo=r'''def purity(groups, labels):
    """Sum over the groups of the size of their most frequent label, divided by the number of samples."""
    raise NotImplementedError("purity() is not written yet")


X_scaled = ...   # the 4 measures of penguins, each column centred and divided by its standard deviation''',
       check=r'''with wb.attempt("1.21"):
    groups_raw = cluster(penguins[MEASURES])
    print(pd.crosstab(groups_raw, penguins["species"], rownames=["group"], colnames=["species"]))
    print(f"purity with the raw measures: {purity(groups_raw, penguins['species']):.3f}\n")
    if X_scaled is ...:
        raise NotImplementedError("X_scaled is not written yet")
    groups_scaled = cluster(X_scaled)
    print(pd.crosstab(groups_scaled, penguins["species"], rownames=["group"], colnames=["species"]))
    print(f"purity with the scaled measures: {purity(groups_scaled, penguins['species']):.3f}")
    verdict("1.21", abs(purity(np.zeros(4), np.array(["a", "a", "b", "c"])) - 0.5) < 1e-9,
            "ta pureté donne 0,5 sur un petit exemple calculable à la main (1 groupe : 2 « a » sur 4).",
            "sur un petit exemple (un seul groupe contenant a, a, b, c), la pureté devrait valoir 2/4 = 0,5.")''',
       solution=r'''def purity(groups, labels):
    """Sum over the groups of the size of their most frequent label, divided by the number of samples."""
    table = pd.crosstab(np.asarray(groups), np.asarray(labels))
    return float(table.max(axis=1).sum() / len(labels))


X_scaled = (penguins[MEASURES] - penguins[MEASURES].mean()) / penguins[MEASURES].std()
groups_raw = cluster(penguins[MEASURES])
groups_scaled = cluster(X_scaled)
print(pd.crosstab(groups_raw, penguins["species"], rownames=["group"], colnames=["species"]))
print(f"purity with the raw measures: {purity(groups_raw, penguins['species']):.3f}\n")
print(pd.crosstab(groups_scaled, penguins["species"], rownames=["group"], colnames=["species"]))
print(f"purity with the scaled measures: {purity(groups_scaled, penguins['species']):.3f}")''',
       after=[("todo_md", "📝 **Mes réponses** (1.21) : …"),
              ("solution_md", "**Réponses (1.21)** : sur les mesures brutes, la masse (des milliers de grammes) "
               "écrase les autres mesures (quelques dizaines de millimètres) dans le calcul des distances : les "
               "groupes sont surtout des tranches de masse, et les Gentoo sont coupés en deux (pureté ≈ 0,68). Une "
               "fois les mesures à la même échelle, chaque mesure compte autant : la pureté monte à environ 0,92, "
               "les Gentoo forment un groupe à eux seuls, et seuls des Adélie et des Chinstrap restent mélangés "
               "(ils ont des tailles voisines). Oui, une biologiste qui ignorerait les espèces verrait apparaître "
               "des groupes naturels ; mais c'est à elle de décider ce qu'ils signifient : le clustering ne donne "
               "pas de noms.")],
       note="Sans étiquettes, l'algorithme ne peut que regrouper ce qui se ressemble ; ce qu'il appelle « se "
            "ressembler » dépend entièrement de la distance choisie, donc des unités des features. Les groupes "
            "n'ont pas de nom : le groupe 0 n'est « l'Adélie » que si nous le décidons après coup."),

    Ex("1.22", "🔬", 2, 30, "L'agent cuisinier : apprendre par la récompense",
       "simuler un agent qui apprend par essais et récompenses, et mesurer l'effet de la part d'exploration.",
       "0A (boucles, `np.random.default_rng`) · fiche §1.6 · livre §1.6 (fig. 1.20)", thread="bandit", tracks="C",
       body=r"""Comme dans l'histoire du livre (§1.6), un cuisinier doit nourrir un enfant pendant un an, sans savoir ce qu'il aime. Chaque soir, il choisit une recette parmi cinq ; la seule information qu'il reçoit est si l'enfant mange (récompense 1) ou non (récompense 0). Les goûts de l'enfant sont des probabilités **cachées** (`LIKES`) : le cuisinier ne les connaît pas, il doit les découvrir en cuisinant.

Stratégie du cuisinier :
- les soirs 0 à 4 : il essaie chaque recette une fois, dans l'ordre ;
- ensuite, chaque soir, il tire `rng.random()` : si le résultat est plus petit que `explore`, il **explore** (une recette au hasard, `int(rng.integers(5))`) ; sinon il **exploite** : la recette qui a le meilleur taux de repas mangés jusqu'ici (`np.argmax`) ;
- la réaction de l'enfant est donnée par l'outil fourni `child_eats(recipe, rng)`.

Écris `cook_year(explore, rng)`, qui simule 365 soirs et renvoie `(nombre de repas mangés, liste des recettes cuisinées)`, en tirant les nombres aléatoires **exactement dans cet ordre**. La vérification contrôle d'abord une année précise (`explore = 0.1`, `rng = np.random.default_rng(0)`) : le nombre de repas mangés doit tomber pile. L'expérience fournie fait ensuite vivre 200 années à chaque valeur de `explore` ; elle trace la part des repas mangés et la part des années où, sur les 30 derniers soirs, le cuisinier cuisine le plus souvent la recette préférée de l'enfant.

Réponds dans la cellule **📝 Mes réponses** : qui est l'agent, l'environnement, l'action, la récompense ? Pourquoi `explore = 0` ne trouve-t-il pas toujours la recette préférée ? Pourquoi `explore = 1` est-il mauvais ? Quelle valeur choisirais-tu ?""",
       given=r'''RECIPES = ["butter pasta", "cheese pasta", "pesto pasta", "tomato pasta", "baked potato"]
LIKES = np.array([0.6, 0.8, 0.1, 0.2, 0.5])   # hidden tastes of the child: the cook does NOT know them


def child_eats(recipe, rng):
    """The environment: 1 if the child eats the meal, else 0."""
    return int(rng.random() < LIKES[recipe])


def year_statistics(explore, n_years=200, seed=0):
    """Mean share of meals eaten, and share of years ending on the favourite recipe (last 30 evenings)."""
    rng = np.random.default_rng(seed)
    eaten, found = [], []
    for _ in range(n_years):
        n_eaten, cooked = cook_year(explore, rng)
        eaten.append(n_eaten / 365)
        found.append(np.bincount(cooked[-30:], minlength=5).argmax() == LIKES.argmax())
    return float(np.mean(eaten)), float(np.mean(found))''',
       todo=r'''def cook_year(explore, rng, n_evenings=365):
    """Simulate one year of dinners; returns (number of meals eaten, list of the recipes cooked)."""
    raise NotImplementedError("cook_year() is not written yet")''',
       check=r'''with wb.attempt("1.22"):
    wb.check("1.22", cook_year(0.1, np.random.default_rng(0))[0])
    EXPLORES = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    stats_22 = {explore: year_statistics(explore) for explore in EXPLORES}
    for explore, (eaten, found) in stats_22.items():
        print(f"explore = {explore:<5} meals eaten: {eaten:.1%}   favourite found: {found:.0%}")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(EXPLORES, [s[0] for s in stats_22.values()], marker="o", label="share of meals eaten")
    ax.plot(EXPLORES, [s[1] for s in stats_22.values()], marker="s", label="share of years ending on the favourite")
    ax.set_xlabel("explore")
    ax.legend()
    plt.show()
    verdict("1.22", abs(stats_22[1.0][0] - LIKES.mean()) < 0.02 and stats_22[0.1][1] > 0.8,
            "en explorant tout le temps, on mange au hasard (la moyenne des goûts) ; avec 10 % d'exploration, on trouve presque toujours la recette préférée.",
            "avec explore = 1, on devrait trouver la moyenne des goûts (44 %), et avec 0.1 la recette préférée presque "
            "toujours : vérifie ton tirage au hasard et ton choix de la meilleure recette (le TAUX de repas mangés).")''',
       solution=r'''def cook_year(explore, rng, n_evenings=365):
    """Simulate one year of dinners; returns (number of meals eaten, list of the recipes cooked)."""
    n_tried = np.zeros(len(LIKES))
    n_eaten_by_recipe = np.zeros(len(LIKES))
    n_eaten, cooked = 0, []
    for evening in range(n_evenings):
        if evening < len(LIKES):
            recipe = evening                                        # first, try each recipe once
        elif rng.random() < explore:
            recipe = int(rng.integers(len(LIKES)))                  # explore: a random recipe
        else:
            recipe = int(np.argmax(n_eaten_by_recipe / n_tried))    # exploit: the best rate so far
        reward = child_eats(recipe, rng)
        n_tried[recipe] += 1
        n_eaten_by_recipe[recipe] += reward
        n_eaten += reward
        cooked.append(recipe)
    return n_eaten, cooked


meals_22 = cook_year(0.1, np.random.default_rng(0))[0]
print("one year with explore = 0.1 and default_rng(0):", meals_22, "meals eaten")
EXPLORES = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
stats_22 = {explore: year_statistics(explore) for explore in EXPLORES}
for explore, (eaten, found) in stats_22.items():
    print(f"explore = {explore:<5} meals eaten: {eaten:.1%}   favourite found: {found:.0%}")
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(EXPLORES, [s[0] for s in stats_22.values()], marker="o", label="share of meals eaten")
ax.plot(EXPLORES, [s[1] for s in stats_22.values()], marker="s", label="share of years ending on the favourite")
ax.set_xlabel("explore")
ax.legend()
plt.show()''',
       record=r'''wb.record("1.22", meals_22)''',
       after=[("todo_md", "📝 **Mes réponses** (1.22) : …"),
              ("solution_md", "**Réponses (1.22)** : l'**agent** est le cuisinier, l'**environnement** l'enfant (tout "
               "le reste), l'**action** la recette du soir, la **récompense** 1 si l'enfant mange, 0 sinon. Aucun "
               "label ne dit « la bonne recette était… » : seulement un retour sur l'action choisie. Avec "
               "`explore = 0`, un premier essai malchanceux (les pâtes au fromage refusées le soir 1) fait tomber "
               "la meilleure recette à 0 %, et le cuisinier ne la recuisine plus jamais : près d'une année sur "
               "trois, il reste bloqué sur une recette moins bonne. Avec `explore = 1`, il ne profite jamais de "
               "ce qu'il a appris : 44 %, la moyenne des goûts. Un peu d'exploration (5 à 10 % des soirs) trouve "
               "presque toujours la recette préférée tout en mangeant autant ou plus : c'est le compromis "
               "**exploration contre exploitation**, au cœur de l'apprentissage par renforcement (ch. 11 et 26).")],
       note="Ce petit problème s'appelle un **bandit manchot** (*multi-armed bandit*) : chaque recette est un bras de "
            "machine à sous dont on ignore le gain moyen. Tu le reprendras au ch. 11 avec la stratégie "
            "« ε-greedy », qui est exactement celle de `cook_year`."),

    Ex("1.23", "📦", 2, 25, "Un réseau de neurones en boîte noire sur MNIST",
       "entraîner un petit réseau de neurones avec scikit-learn, compter ses paramètres et examiner ses erreurs.",
       "Ex 1.10 · Quiz 1.Q10 · fiche §1.7 · livre §1.7 (fig. 1.21 et 1.23)", thread="MNIST", tracks="R, C",
       body=r"""Voici un vrai réseau de neurones (détaillé aux ch. 10 et 16) : 784 entrées (les pixels), une couche cachée de 128 neurones, 10 sorties (une par chiffre). Les trois lignes de scikit-learn sont fournies ; l'entraînement fait 30 passages (epochs) sur 5 000 images en `FAST_MODE` (60 000 sinon) et prend quelques secondes. Un avertissement `ConvergenceWarning` peut s'afficher : l'entraînement a été arrêté exprès au bout de 30 passages pour aller vite. On le laisse s'afficher : un avertissement se lit toujours (0B.37).

a) `n_params` : le nombre total de paramètres du réseau, poids et biais. Les poids sont dans la liste de matrices `mlp.coefs_`, les biais dans la liste de vecteurs `mlp.intercepts_` (additionne les `.size`). Si tu as fait l'exercice papier 1.3, compare avec ta réponse à 1.3 c.
b) `n_layers` : le nombre de couches que compte scikit-learn (`mlp.n_layers_`) ; lesquelles compte-t-il ?
c) `test_accuracy` : l'accuracy sur les 10 000 images de test (`mlp.score`, une proportion entre 0 et 1) ; et `n_errors`, le nombre d'images de test mal classées. Compare avec les 95 erreurs du petit réseau du livre (§1.7).

La vérification affiche des images de test mal classées, titrées « ✗ prédiction (vrai label) » : les aurais-tu reconnues ?""",
       given=r'''from sklearn.neural_network import MLPClassifier

n_train = 5000 if FAST_MODE else 60000
X_tr, y_tr = wb.datasets.load_mnist("train", n=n_train, seed=0, flatten=True, normalize=True)   # (n_train, 784)
X_te, y_te = wb.datasets.load_mnist("test", flatten=True, normalize=True)                      # (10000, 784)

mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, random_state=0)   # 784 -> 128 -> 10
mlp.fit(X_tr, y_tr)                                                            # training: a few seconds
predictions_23 = mlp.predict(X_te)''',
       todo=r'''n_params = ...        # a) weights + biases
n_layers = ...        # b)
test_accuracy = ...   # c)
n_errors = ...        # c)''',
       check=r'''wb.check("1.23a", n_params)
wb.check("1.23b", n_layers)
if test_accuracy is not ... and n_errors is not ...:
    print(f"test accuracy: {test_accuracy:.2%} · errors: {n_errors} out of {len(y_te)}")
    verdict("1.23", 0.9 <= test_accuracy <= 1 and n_errors == int((predictions_23 != y_te).sum()),
            "plus de 90 % de chiffres reconnus, sans avoir écrit une seule règle.",
            "vérifie test_accuracy (mlp.score sur X_te, y_te : une proportion entre 0 et 1) et n_errors (les prédictions différentes des labels).")
wrong = np.flatnonzero(predictions_23 != y_te)[:16]
wb.plot.show_images(X_te[wrong].reshape(-1, 28, 28), y_te[wrong], predictions_23[wrong], ncols=8)
plt.show()''',
       solution=r'''n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
n_layers = mlp.n_layers_
test_accuracy = mlp.score(X_te, y_te)
n_errors = int((predictions_23 != y_te).sum())
print(n_params, [w.shape for w in mlp.coefs_], n_layers, f"{test_accuracy:.2%}", n_errors)
wrong = np.flatnonzero(predictions_23 != y_te)[:16]
wb.plot.show_images(X_te[wrong].reshape(-1, 28, 28), y_te[wrong], predictions_23[wrong], ncols=8)
plt.show()''',
       record=r'''wb.record("1.23a", n_params, mistakes={"n'oublie pas les biais (mlp.intercepts_)": sum(w.size for w in mlp.coefs_)})
wb.record("1.23b", n_layers, mistakes={"scikit-learn compte aussi la couche d'ENTRÉE (les pixels)": 2})''',
       note="101 770 paramètres, exactement le calcul de 1.3 c : $784 \\times 128 + 128 + 128 \\times 10 + 10$. "
            "scikit-learn compte 3 couches : l'entrée, la couche cachée et la sortie. Avec 5 000 images, le "
            "réseau fait environ 93,6 % (644 erreurs ici ; un peu plus ou un peu moins selon la machine) ; avec les 60 000 images (`FAST_MODE = False`, "
            "environ 30 s), environ 97,8 %. Pour passer les 99 % du petit réseau du livre (95 erreurs), il faut "
            "plus d'entraînement ou des réseaux convolutifs, conçus pour les images (ch. 21). Beaucoup d'images "
            "mal classées sont réellement ambiguës."),

    Ex("1.24", "🔨", 3, 35, "Fabriquer du faux Holmes et du faux Verne",
       "écrire un petit générateur de texte (bigrammes de caractères) et le relier aux modèles de langage actuels.",
       "Ex 1.11 · 0A (`dict`, `Counter`) · fiche §1.5, 🕰️ panorama 2026 · livre §1.5", thread="Holmes/Verne",
       tracks="C",
       body=r"""Un **générateur** (§1.5) produit de nouvelles données qui ressemblent aux exemples. Le plus simple des générateurs de texte : pour chaque caractère, on compte quels caractères le suivent dans le livre (les **bigrammes**, paires de caractères consécutifs), puis on écrit un faux texte caractère par caractère, en tirant chaque caractère suivant au hasard, avec des probabilités proportionnelles à ces comptes.

1. `bigram_counts(text)` : un dictionnaire `{caractère: Counter des caractères qui le suivent}`. Par exemple, dans `"abab"`, `"a"` est suivi 2 fois de `"b"`, et `"b"` 1 fois de `"a"`.
2. `generate(counts, start, length, rng)` : un texte de `length` caractères qui commence par le caractère `start`. À chaque pas, les candidats sont les caractères qui suivent le caractère courant, **triés** (`sorted(counts[current])`), avec les poids `counts[current][c]` ; on tire le suivant avec `rng.choice(followers, p=weights / weights.sum())`.

La vérification contrôle tes comptes (par exemple, le nombre de « qu » de *Holmes*), puis écrit 300 caractères de faux Holmes et de faux Verne. Réponds dans la cellule **📝 Mes réponses** : qu'est-ce qui ressemble à de l'anglais et à du français ? Qu'est-ce qui manque ? En quoi un grand modèle de langage (LLM) fait-il « la même chose en beaucoup plus grand » ?""",
       todo=r'''def bigram_counts(text):
    """For each character, a Counter of the characters that follow it in text."""
    raise NotImplementedError("bigram_counts() is not written yet")


def generate(counts, start, length, rng):
    """A text of `length` characters starting with `start`, each next character drawn from the followers
    of the current one (sorted), with probabilities proportional to their counts."""
    raise NotImplementedError("generate() is not written yet")''',
       check=r'''with wb.attempt("1.24"):
    counts_holmes, counts_verne = bigram_counts(holmes), bigram_counts(verne)
    verdict("1.24", counts_holmes["q"]["u"] == holmes.count("qu")
            and sum(sum(c.values()) for c in counts_holmes.values()) == len(holmes) - 1,
            "tes comptes de bigrammes sont justes (autant de paires que de caractères moins un).",
            "tes comptes ne correspondent pas au texte : chaque paire de caractères consécutifs compte une fois.")
    fake_holmes = generate(counts_holmes, "T", 300, np.random.default_rng(0))
    fake_verne = generate(counts_verne, "L", 300, np.random.default_rng(0))
    verdict("1.24", len(fake_holmes) == 300 and fake_holmes[0] == "T",
            "le texte généré a la bonne longueur et commence par le bon caractère.",
            "le texte généré doit faire exactement `length` caractères et commencer par `start`.")
    print(fake_holmes)
    print("-" * 60)
    print(fake_verne)''',
       solution=r'''from collections import Counter, defaultdict


def bigram_counts(text):
    """For each character, a Counter of the characters that follow it in text."""
    counts = defaultdict(Counter)
    for current, following in zip(text, text[1:]):
        counts[current][following] += 1
    return counts


def generate(counts, start, length, rng):
    """A text of `length` characters starting with `start`, each next character drawn from the followers
    of the current one (sorted), with probabilities proportional to their counts."""
    characters = [start]
    for _ in range(length - 1):
        followers = sorted(counts[characters[-1]])
        weights = np.array([counts[characters[-1]][c] for c in followers], dtype=float)
        characters.append(str(rng.choice(followers, p=weights / weights.sum())))
    return "".join(characters)


counts_holmes, counts_verne = bigram_counts(holmes), bigram_counts(verne)
print(counts_holmes["q"].most_common(3), counts_verne["q"].most_common(3))
fake_holmes = generate(counts_holmes, "T", 300, np.random.default_rng(0))
fake_verne = generate(counts_verne, "L", 300, np.random.default_rng(0))
print(fake_holmes)
print("-" * 60)
print(fake_verne)''',
       after=[("todo_md", "📝 **Mes réponses** (1.24) : …"),
              ("solution_md", "**Réponses (1.24)** : le faux Holmes a l'air anglais de loin : des « th », des « he », "
               "des mots courts, des guillemets et des majuscules après les points ; le faux Verne place des accents "
               "et des « qu » au bon endroit. Mais presque aucun mot n'existe, et il n'y a ni grammaire ni sens : le "
               "modèle ne voit qu'**un** caractère en arrière. Un LLM fait la même chose, prédire la suite, mais avec "
               "des **tokens** (des morceaux de mots) au lieu de caractères, un **contexte** de milliers de tokens "
               "au lieu d'un seul caractère, et des **milliards de paramètres** appris sur une grande partie du web "
               "au lieu d'un tableau de comptes tiré d'un seul livre ; il est ensuite aligné sur des préférences "
               "humaines (fiche, 🕰️ panorama 2026). Pour aller plus loin : un contexte de 2 ou 3 caractères "
               "(trigrammes) donne déjà beaucoup plus de vrais mots.")],
       note="Ce générateur n'a besoin d'aucun label : le texte lui-même fournit la « bonne réponse » (le caractère "
            "suivant). C'est l'idée de l'apprentissage **auto-supervisé**, le mode de pré-entraînement de tous les "
            "LLM (fiche, 🕰️ familles d'apprentissage)."),
])

# ---------------------------------------------------------------------------
# Part D: the challenge (1.25)
# ---------------------------------------------------------------------------
PART_D = Part("D", "Défi final", "Tout le chapitre : règles écrites à la main, jeu de test honnête, comparaison "
              "avec un modèle appris.", exercises=[
    Ex("1.25", "🏆", 3, 40, "Battre l'expert : 95 % avec tes propres règles",
       "écrire ses propres règles de classification et atteindre 95 % d'accuracy sur le jeu de test, sans tricher.",
       "Ex 1.15, Ex 1.18 · fiche §1.1.2, §1.2.3", thread="Penguins", tracks="C",
       body=r"""Les règles de la biologiste (1.15) restent sous les 95 % sur le jeu de test. À ton tour d'être l'expert : écris `my_rule(bill_length, bill_depth, flipper_length, body_mass)`, avec **au plus 4 tests** `if`/`elif` (un test peut combiner deux comparaisons avec `and`), qui atteint **au moins 95 %** d'accuracy sur le jeu de test.

**Règle du jeu (honnêteté)** : tu construis tes règles en regardant **uniquement le jeu d'entraînement** (les graphiques ci-dessous, et `accuracy(y_train, predict_with(my_rule, X_train))` autant de fois que tu veux). Tu n'évalues sur le jeu de test **qu'à la fin**, une fois tes règles fixées. Si tu ajustes tes seuils pour gagner des points sur le test, ton score ne mesure plus la généralisation (c'est l'erreur de 1.20, si tu l'as faite).

Pistes : quelle mesure sépare le mieux les Gentoo (1.15, 1.18) ? Puis, parmi les autres, laquelle sépare les Chinstrap des Adélie ?""",
       given=r'''pairs = [("flipper_length_mm", "bill_length_mm"), ("bill_depth_mm", "bill_length_mm"), ("flipper_length_mm", "bill_depth_mm")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (x_name, y_name) in zip(axes, pairs):
    for species in ["Adelie", "Chinstrap", "Gentoo"]:
        part = train[train["species"] == species]
        ax.scatter(part[x_name], part[y_name], s=12, label=species)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
axes[0].legend()
fig.suptitle("Training set only")
plt.show()''',
       todo=r'''def my_rule(bill_length, bill_depth, flipper_length, body_mass):
    """Your own rules (at most 4 conditions): returns "Adelie", "Chinstrap" or "Gentoo"."""
    raise NotImplementedError("my_rule() is not written yet")''',
       check=r'''with wb.attempt("1.25"):
    my_train_acc = accuracy(y_train, predict_with(my_rule, X_train))
    my_test_acc = accuracy(y_test, predict_with(my_rule, X_test))
    print(f"your rules: {my_train_acc:.1%} on the training set, {my_test_acc:.1%} on the test set")
    print(pd.crosstab(y_test, predict_with(my_rule, X_test), rownames=["true"], colnames=["predicted"]))
    verdict("1.25", my_test_acc >= 0.95,
            "🏆 défi réussi : au moins 95 % sur des manchots que tes règles n'avaient jamais vus.",
            "pas encore 95 % sur le test : reviens aux graphiques du jeu d'entraînement (pas au test !).")''',
       solution=r'''def my_rule(bill_length, bill_depth, flipper_length, body_mass):
    """Your own rules (at most 4 conditions): returns "Adelie", "Chinstrap" or "Gentoo"."""
    if flipper_length > 206 and bill_depth < 17.5:
        return "Gentoo"          # long flippers and a thin bill
    if bill_length > 44:
        return "Chinstrap"       # among the others, the long bills
    return "Adelie"


my_train_acc = accuracy(y_train, predict_with(my_rule, X_train))
my_test_acc = accuracy(y_test, predict_with(my_rule, X_test))
print(f"these rules: {my_train_acc:.1%} on the training set, {my_test_acc:.1%} on the test set")
print(pd.crosstab(y_test, predict_with(my_rule, X_test), rownames=["true"], colnames=["predicted"]))''',
       note="Une solution parmi d'autres : les Gentoo ont une longue nageoire **et** un bec fin ; parmi les autres, "
            "un bec long signe le Chinstrap. Trois règles lues sur les graphiques d'entraînement dépassent les 95 % "
            "sur le test. Tu as fait le travail d'un expert… que l'arbre de 1.18 a fait seul, en une seconde. Sur "
            "784 pixels (MNIST), personne ne saurait écrire ces règles à la main : c'est là que le machine "
            "learning devient indispensable."),
])

PARTS = [PART_A, PART_B, PART_C, PART_D]


def header_cells(kind: str) -> list:
    rel = f"{FOLDER}/{'03_notebook' if kind == 'exercise' else '05_solutions'}.ipynb"
    rows = ["| Partie | ID | Titre | Type | ★ | ⏱️ |", "|---|---|---|---|---|---|",
            "| 0 | 1.1 à 1.4 | vérifier tes exercices papier ✏️ | ✏️ | ★ | |"]
    for part in PARTS:
        for i, ex in enumerate(part.exercises):
            rows.append(f"| {part.key if i == 0 else ''} | {ex.id} | {ex.title} | {ex.type} | {STARS[ex.stars]} | {ex.minutes} |")
    if kind == "exercise":
        title = "# 1 · Introduction au machine learning et au deep learning — notebook d'exercices"
        how = ("La partie 0 vérifie tes exercices papier. Chaque exercice de code : un énoncé, une cellule à "
               "compléter (les `...` et les `raise NotImplementedError`), puis une cellule de vérification "
               "(`wb.check`, ou des ✅/❌ pour les expériences). « Exécuter tout » va jusqu'au bout même si rien "
               "n'est rempli : ce qui n'est pas fait affiche ⏳. Les questions « pourquoi ? » qui n'ont pas de cellule 📝 se notent dans la section "
               "« Notes sur le notebook » de ta copie de `06_mes_reponses.md`. Bloqué 15 minutes ? `04_indices.md`.\n\n"
               "> Travaille dans **ta copie** (`mon_travail/ch01_introduction/03_notebook.ipynb`, créée par "
               "`python tools/start_chapter.py 1`) : ce fichier-ci est mis à jour par Claude.")
    else:
        title = "# 1 · Introduction au machine learning et au deep learning — solutions (notebook exécuté)"
        how = ("Les réponses des exercices, exécutées. Les démarches détaillées (le *pourquoi*, les erreurs "
               "fréquentes, les variantes) sont dans `05_solutions.md`. Les cellules marquées `answer` "
               "enregistrent les réponses vérifiées par `wb.check` (`tools/build_answers.py`).")
    return [md(f"{title}\n\n{badge(rel)}\n\n{how}\n\n" + "\n".join(rows))]


def objectives_cell() -> list:
    return [md("## Objectifs et rappel express\n\n"
               "- Décrire les quatre fils rouges du workbook avec le vocabulaire du ML : échantillons, features, labels.\n"
               "- Voir la différence entre mémoriser et généraliser, et évaluer honnêtement sur un jeu de test.\n"
               "- Écrire des règles à la main, puis une vraie boucle d'entraînement, et comparer avec un modèle qui "
               "apprend seul ses règles.\n"
               "- Faire tourner en boîte noire un clustering, un agent qui apprend par la récompense, un réseau de "
               "neurones et un générateur de texte.\n\n"
               "**Rappel express.** Un dataset est un tableau : une ligne par **échantillon**, une colonne par "
               "**feature**, plus le **label** à prédire. Entraîner : prédire, mesurer l'erreur (la **loss**), "
               "corriger les **paramètres** d'un pas réglé par le **learning rate**. On mesure la **généralisation** "
               "sur un **jeu de test** mis de côté, jamais utilisé pour apprendre. $\\text{accuracy} = "
               "\\frac{\\text{prédictions correctes}}{\\text{nombre d'échantillons}}$. En scikit-learn : "
               "`model.fit(X_train, y_train)`, puis `model.predict(X_new)` ou `model.score(X_test, y_test)`.")]


def footer_cells(kind: str) -> list:
    return [md("## ✅ Bilan\n\n"
               "**Auto-évaluation** (note-toi de 0 à 3 dans `mon_travail/suivi/auto_evaluation.md`) :\n"
               "1. Sais-tu décrire un dataset (échantillons, features, label) et dire à quelle famille appartient une "
               "tâche (classification, régression, clustering, renforcement, génération) ?\n"
               "2. Sais-tu écrire une boucle d'entraînement minimale et expliquer ce que fait le learning rate ?\n"
               "3. Sais-tu repérer une évaluation faussée (mémorisation, fuite du label, test vu à l'entraînement) ?\n\n"
               "**Pour aller plus loin** : le *Machine Learning Crash Course* de Google et les vidéos de 3Blue1Brown "
               "sur les réseaux de neurones, cités dans la fiche. La suite : le ch. 2 (hasard et statistiques) "
               "donne les outils pour décrire les données et leurs fluctuations ; tu retrouveras le jeu de test au "
               "ch. 8, la régression au ch. 9, les neurones au ch. 10, le bandit au ch. 11, les arbres au ch. 13.")]


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
